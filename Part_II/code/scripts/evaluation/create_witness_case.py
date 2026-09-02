#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent

MACH = 1.50
ALPHA = 0.1625
CASE_NAME = "witness_M150_a01625"

SPECTRAL = PACKAGE / "data/spectral/supersonic_reference_v2_spectral.csv"
RAW_MODAL = PACKAGE / "data/modal/supersonic_reference_v2_modal_raw.parquet"
MODAL_METADATA = PACKAGE / "data/modal/modal_reference_metadata.json"

SAMPLE = PACKAGE / f"data/samples/{CASE_NAME}_raw.parquet"
EXPECTED = PACKAGE / f"data/samples/{CASE_NAME}_expected.json"
CONFIG = PACKAGE / f"configs/reproducibility/{CASE_NAME}.yaml"
CHECKSUMS = PACKAGE / "data/samples/SHA256SUMS.txt"

PHYSICAL_COLUMNS = [
    "p_real",
    "p_imag",
    "rho_real",
    "rho_imag",
    "u_real",
    "u_imag",
    "v_real",
    "v_imag",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def finite_scalar(row: pd.Series, names: list[str]) -> float | None:
    for name in names:
        if name not in row.index:
            continue

        value = pd.to_numeric(
            pd.Series([row[name]]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(value) and np.isfinite(value):
            return float(value)

    return None


def main() -> None:
    if not SPECTRAL.exists():
        raise FileNotFoundError(SPECTRAL)

    if not RAW_MODAL.exists():
        raise FileNotFoundError(
            f"Missing local full modal Parquet: {RAW_MODAL}"
        )

    spectral = pd.read_csv(SPECTRAL, low_memory=False)

    mask = (
        np.isclose(spectral["Mach"], MACH, atol=1e-12)
        & np.isclose(spectral["alpha"], ALPHA, atol=1e-12)
    )

    selected = spectral.loc[mask]

    if len(selected) != 1:
        raise ValueError(
            f"Expected one spectral point for M={MACH}, "
            f"alpha={ALPHA}; found {len(selected)}"
        )

    row = selected.iloc[0]

    cr = finite_scalar(row, ["cr", "reference_cr"])
    ci = finite_scalar(row, ["ci", "reference_ci"])
    omega_i = finite_scalar(
        row,
        ["omega_i", "reference_omega_i"],
    )

    if cr is None or ci is None:
        raise ValueError("Missing canonical cr or ci")

    if omega_i is None:
        omega_i = ALPHA * ci

    dataset = ds.dataset(
        RAW_MODAL,
        format="parquet",
    )

    tolerance = 1e-12

    condition = (
        (ds.field("Mach") >= MACH - tolerance)
        & (ds.field("Mach") <= MACH + tolerance)
        & (ds.field("alpha") >= ALPHA - tolerance)
        & (ds.field("alpha") <= ALPHA + tolerance)
    )

    modal = dataset.to_table(filter=condition).to_pandas()

    modal = modal[
        np.isclose(modal["Mach"], MACH, atol=tolerance)
        & np.isclose(modal["alpha"], ALPHA, atol=tolerance)
    ].copy()

    if modal.empty:
        raise ValueError("No modal rows found for witness point")

    missing = [
        name for name in PHYSICAL_COLUMNS
        if name not in modal.columns
    ]

    if missing:
        raise ValueError(
            f"Missing physical modal columns: {missing}"
        )

    numeric_columns = [
        "Mach",
        "alpha",
        "y",
        "cr",
        "ci",
        "omega_i",
        *PHYSICAL_COLUMNS,
    ]

    for name in numeric_columns:
        if name in modal.columns:
            modal[name] = pd.to_numeric(
                modal[name],
                errors="raise",
            )

    if not np.isfinite(
        modal[["Mach", "alpha", "y", *PHYSICAL_COLUMNS]]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite coordinates or physical modal values"
        )

    modal = modal.sort_values("y").reset_index(drop=True)

    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)

    modal.to_parquet(
        SAMPLE,
        index=False,
        compression="zstd",
    )

    source_metadata = {}

    if MODAL_METADATA.exists():
        metadata = json.loads(MODAL_METADATA.read_text())

        raw_entry = next(
            (
                item for item in metadata.get("datasets", [])
                if item.get("variant") == "raw_confirmed"
            ),
            None,
        )

        if raw_entry:
            source_metadata = {
                "path": raw_entry.get("output"),
                "sha256": raw_entry.get("output_sha256"),
                "row_count": raw_entry.get("row_count"),
            }

    sample_sha = sha256(SAMPLE)

    expected = {
        "name": CASE_NAME,
        "reference_version": "2.0.0",
        "point": {
            "Mach": MACH,
            "alpha": ALPHA,
        },
        "spectral": {
            "cr": cr,
            "ci": ci,
            "omega_i": omega_i,
            "validation_status": str(
                row.get("validation_status", "")
            ),
        },
        "modal": {
            "variant": "raw_confirmed",
            "sample_path": str(SAMPLE.relative_to(REPO)),
            "sample_sha256": sample_sha,
            "n_rows": int(len(modal)),
            "y_min": float(modal["y"].min()),
            "y_max": float(modal["y"].max()),
            "columns": list(modal.columns),
        },
        "source_modal_dataset": source_metadata,
        "numerical_reproduction": {
            "status": "solver_adapter_pending",
            "spectral_tolerances": None,
            "modal_tolerances": None,
            "note": (
                "Numerical tolerances must be calibrated from "
                "repeated solver runs and must not be guessed."
            ),
        },
    }

    EXPECTED.write_text(
        json.dumps(expected, indent=2) + "\n"
    )

    config = {
        "case": {
            "name": CASE_NAME,
            "Mach": MACH,
            "alpha": ALPHA,
        },
        "reference": {
            "spectral_csv": str(SPECTRAL.relative_to(REPO)),
            "modal_sample": str(SAMPLE.relative_to(REPO)),
            "expected_json": str(EXPECTED.relative_to(REPO)),
            "modal_variant": "raw_confirmed",
        },
        "expected_spectral": {
            "cr": cr,
            "ci": ci,
            "omega_i": omega_i,
        },
        "comparison": {
            "phase_alignment": "complex_least_squares",
            "spectral_tolerances": None,
            "modal_core_tolerances": None,
            "tail_comparison": "reported_separately",
        },
        "solver_adapter": {
            "status": "pending_api_inventory",
        },
    }

    CONFIG.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        )
    )

    checksum_files = [
        SAMPLE,
        EXPECTED,
        CONFIG,
    ]

    with CHECKSUMS.open("w") as stream:
        for path in checksum_files:
            stream.write(
                f"{sha256(path)}  {path.relative_to(REPO)}\n"
            )

    print("Created witness case")
    print("case:", CASE_NAME)
    print("Mach:", MACH)
    print("alpha:", ALPHA)
    print("cr:", cr)
    print("ci:", ci)
    print("omega_i:", omega_i)
    print("modal rows:", len(modal))
    print("y range:", modal["y"].min(), modal["y"].max())
    print("sample:", SAMPLE.relative_to(REPO))
    print("config:", CONFIG.relative_to(REPO))
    print("expected:", EXPECTED.relative_to(REPO))


if __name__ == "__main__":
    main()
