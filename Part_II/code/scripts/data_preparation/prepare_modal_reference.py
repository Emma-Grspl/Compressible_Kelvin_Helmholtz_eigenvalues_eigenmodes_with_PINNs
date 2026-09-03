#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[2]

SOURCE_DIR = Path(__file__).resolve().parents[1] / "data" / "modal" / "source"

OUTPUT_DIR = REPO / "classic_supersonic" / "data" / "modal"

SPECTRAL = (
    REPO / 'assets/classic_supersonic/csv/pinn_direct/spectral/table_supersonic_reference_v2_spectral.csv'
)

SAMPLE_OUTPUT = OUTPUT_DIR / "modal_sample_for_tests.parquet"
METADATA_OUTPUT = OUTPUT_DIR / "modal_reference_metadata.json"
CHECKSUM_OUTPUT = OUTPUT_DIR / "SHA256SUMS.txt"

EXPECTED_ROWS = 838_847
EXPECTED_POINTS = 183

NUMERIC_COLUMNS = {
    "Mach",
    "M",
    "alpha",
    "y",
    "p_real",
    "p_imag",
    "rho_real",
    "rho_imag",
    "u_real",
    "u_imag",
    "v_real",
    "v_imag",
    "cr",
    "ci",
    "omega_i",
    "reference_cr",
    "reference_ci",
    "reference_omega_i",
    "tail_polish_lambda_real",
    "tail_polish_lambda_imag",
    "tail_polish_y_join",
}

DATASETS = {
    "raw_confirmed": {
        "source": SOURCE_DIR
        / "supersonic_sparse_PINN_reference_v2_FINAL_modal_fields_raw_confirmed.csv",
        "output": OUTPUT_DIR / "supersonic_reference_v2_modal_raw.parquet",
    },
    "tail_polished": {
        "source": SOURCE_DIR
        / "supersonic_sparse_PINN_reference_v2_FINAL_modal_fields_tail_polished_v1.csv",
        "output": OUTPUT_DIR
        / "supersonic_reference_v2_modal_tail_polished.parquet",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def canonicalize_table(table: pa.Table) -> pa.Table:
    names = ["Mach" if name == "M" else name for name in table.column_names]

    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate columns after normalization: {names}")

    table = table.rename_columns(names)

    required = {"Mach", "alpha", "y"}
    missing = required.difference(table.column_names)

    if missing:
        raise ValueError(
            f"Missing required modal columns: {sorted(missing)}; "
            f"available={table.column_names}"
        )

    for name in sorted(NUMERIC_COLUMNS.intersection(table.column_names)):
        index = table.schema.get_field_index(name)

        try:
            values = pc.cast(
                table[name],
                pa.float64(),
                safe=False,
            )
        except Exception as exc:
            raise ValueError(
                f"Cannot convert numeric modal column {name!r} "
                f"to float64"
            ) from exc

        table = table.set_column(index, name, values)

    spectral_fallbacks = {
        "cr": ["reference_cr"],
        "ci": ["reference_ci"],
        "omega_i": ["reference_omega_i"],
    }

    for target, fallbacks in spectral_fallbacks.items():
        if target not in table.column_names:
            continue

        arrays = [table[target]]

        for fallback in fallbacks:
            if fallback in table.column_names:
                arrays.append(table[fallback])

        if (
            target == "omega_i"
            and "alpha" in table.column_names
            and "ci" in table.column_names
        ):
            arrays.append(
                pc.multiply(
                    table["alpha"],
                    table["ci"],
                )
            )

        if len(arrays) > 1:
            index = table.schema.get_field_index(target)
            values = pc.coalesce(*arrays)
            table = table.set_column(index, target, values)

    return table


def selected_sample_points() -> pd.DataFrame:
    spectral = pd.read_csv(SPECTRAL, low_memory=False)

    if "Mach" not in spectral.columns and "M" in spectral.columns:
        spectral = spectral.rename(columns={"M": "Mach"})

    spectral["Mach"] = pd.to_numeric(spectral["Mach"], errors="raise")
    spectral["alpha"] = pd.to_numeric(spectral["alpha"], errors="raise")

    selected = []

    for mach, group in spectral.groupby("Mach", sort=True):
        group = group.sort_values("alpha").reset_index(drop=True)
        row = group.iloc[len(group) // 2]
        selected.append(
            {
                "Mach": float(row["Mach"]),
                "alpha": float(row["alpha"]),
            }
        )

    result = pd.DataFrame(selected)

    if len(result) != 11:
        raise ValueError(
            f"Expected one sample point for each of 11 Mach values, got {len(result)}"
        )

    return result


def key_array(mach: pd.Series, alpha: pd.Series) -> list[tuple[float, float]]:
    return list(
        zip(
            pd.to_numeric(mach, errors="raise").round(10),
            pd.to_numeric(alpha, errors="raise").round(10),
        )
    )


def convert_dataset(
    variant: str,
    source: Path,
    output: Path,
    target_keys: set[tuple[float, float]],
) -> tuple[dict, list[pd.DataFrame]]:
    if not source.exists():
        raise FileNotFoundError(source)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    temporary = output.with_suffix(".parquet.tmp")

    if temporary.exists():
        temporary.unlink()

    header = pd.read_csv(
        source,
        nrows=0,
    ).columns.tolist()

    column_types = {
        name: pa.string()
        for name in header
    }

    reader = pacsv.open_csv(
        source,
        read_options=pacsv.ReadOptions(
            block_size=64 * 1024 * 1024,
            use_threads=True,
        ),
        convert_options=pacsv.ConvertOptions(
            column_types=column_types,
            strings_can_be_null=True,
            null_values=[
                "",
                "NA",
                "NaN",
                "nan",
                "NULL",
                "null",
                "None",
            ],
        ),
    )

    writer: pq.ParquetWriter | None = None
    canonical_schema: pa.Schema | None = None

    row_count = 0
    point_keys: set[tuple[float, float]] = set()
    sample_frames: list[pd.DataFrame] = []

    try:
        for batch_number, batch in enumerate(reader, start=1):
            table = canonicalize_table(pa.Table.from_batches([batch]))

            if writer is None:
                canonical_schema = table.schema
                writer = pq.ParquetWriter(
                    temporary,
                    canonical_schema,
                    compression="zstd",
                    use_dictionary=True,
                    write_statistics=True,
                )
            elif table.schema != canonical_schema:
                table = table.cast(canonical_schema, safe=False)

            writer.write_table(table)
            row_count += table.num_rows

            coordinates = table.select(["Mach", "alpha"]).to_pandas()
            batch_keys = key_array(
                coordinates["Mach"],
                coordinates["alpha"],
            )
            point_keys.update(batch_keys)

            mask = np.fromiter(
                (key in target_keys for key in batch_keys),
                dtype=bool,
                count=len(batch_keys),
            )

            if mask.any():
                sampled = table.filter(pa.array(mask)).to_pandas()
                sampled.insert(0, "dataset_variant", variant)
                sample_frames.append(sampled)

            print(
                f"[{variant}] batch={batch_number:02d} "
                f"rows_written={row_count}"
            )

    finally:
        if writer is not None:
            writer.close()

    if row_count != EXPECTED_ROWS:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"{variant}: expected {EXPECTED_ROWS} rows, got {row_count}"
        )

    if len(point_keys) != EXPECTED_POINTS:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"{variant}: expected {EXPECTED_POINTS} modal points, "
            f"got {len(point_keys)}"
        )

    temporary.replace(output)

    metadata = {
        "variant": variant,
        "source": str(source.relative_to(REPO)),
        "output": str(output.relative_to(REPO)),
        "row_count": row_count,
        "unique_spectral_points": len(point_keys),
        "source_size_bytes": source.stat().st_size,
        "output_size_bytes": output.stat().st_size,
        "source_sha256": sha256(source),
        "output_sha256": sha256(output),
        "columns": list(canonical_schema.names if canonical_schema else []),
    }

    return metadata, sample_frames


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_points = selected_sample_points()
    target_keys = set(
        key_array(sample_points["Mach"], sample_points["alpha"])
    )

    all_metadata = []
    all_sample_frames = []

    for variant, paths in DATASETS.items():
        metadata, sample_frames = convert_dataset(
            variant=variant,
            source=paths["source"],
            output=paths["output"],
            target_keys=target_keys,
        )

        all_metadata.append(metadata)
        all_sample_frames.extend(sample_frames)

    sample = pd.concat(
        all_sample_frames,
        ignore_index=True,
        sort=False,
    )

    sample["Mach"] = pd.to_numeric(sample["Mach"], errors="raise")
    sample["alpha"] = pd.to_numeric(sample["alpha"], errors="raise")
    sample["y"] = pd.to_numeric(sample["y"], errors="raise")

    sample = sample.sort_values(
        ["dataset_variant", "Mach", "alpha", "y"]
    ).reset_index(drop=True)

    observed = (
        sample[["dataset_variant", "Mach", "alpha"]]
        .drop_duplicates()
        .groupby("dataset_variant")
        .size()
        .to_dict()
    )

    expected_sample_counts = {
        "raw_confirmed": 11,
        "tail_polished": 11,
    }

    if observed != expected_sample_counts:
        raise ValueError(
            f"Unexpected sample point counts: {observed}"
        )

    sample.to_parquet(
        SAMPLE_OUTPUT,
        index=False,
        compression="zstd",
    )

    metadata_document = {
        "reference_version": "2.0.0",
        "expected_rows_per_dataset": EXPECTED_ROWS,
        "expected_unique_points_per_dataset": EXPECTED_POINTS,
        "sample_strategy": "median-alpha point for each Mach value",
        "sample_points": sample_points.to_dict(orient="records"),
        "sample_path": str(SAMPLE_OUTPUT.relative_to(REPO)),
        "sample_rows": int(len(sample)),
        "sample_sha256": sha256(SAMPLE_OUTPUT),
        "datasets": all_metadata,
    }

    METADATA_OUTPUT.write_text(
        json.dumps(metadata_document, indent=2) + "\n"
    )

    checksum_paths = [
        DATASETS["raw_confirmed"]["source"],
        DATASETS["raw_confirmed"]["output"],
        DATASETS["tail_polished"]["source"],
        DATASETS["tail_polished"]["output"],
        SAMPLE_OUTPUT,
        METADATA_OUTPUT,
    ]

    with CHECKSUM_OUTPUT.open("w") as stream:
        for path in checksum_paths:
            stream.write(
                f"{sha256(path)}  {path.relative_to(REPO)}\n"
            )

    print()
    print("Modal reference prepared")
    print(f"metadata: {METADATA_OUTPUT.relative_to(REPO)}")
    print(f"sample:   {SAMPLE_OUTPUT.relative_to(REPO)}")
    print(f"sample rows: {len(sample)}")

    for item in all_metadata:
        size_mib = item["output_size_bytes"] / 1024**2
        print(
            f"{item['variant']}: "
            f"{item['row_count']} rows, "
            f"{item['unique_spectral_points']} points, "
            f"{size_mib:.2f} MiB"
        )


if __name__ == "__main__":
    main()
