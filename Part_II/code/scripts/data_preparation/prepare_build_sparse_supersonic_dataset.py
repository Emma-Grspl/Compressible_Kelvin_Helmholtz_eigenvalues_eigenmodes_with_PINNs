#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ID_COLUMN = "final_reference_id"

REQUIRED_SPECTRAL_COLUMNS = {
    ID_COLUMN,
    "Mach",
    "alpha",
    "cr",
    "ci",
}

REQUIRED_MODAL_COLUMNS = {
    ID_COLUMN,
    "Mach",
    "alpha",
    "cr",
    "ci",
    "y",
    "kappa",
    "q",
    "p_real",
    "p_imag",
}

OPTIONAL_MODAL_COLUMNS = [
    "coordinate_index",
    "omega_i",
    "rho_real",
    "rho_imag",
    "u_real",
    "u_imag",
    "v_real",
    "v_imag",
]


def canonical_id(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    try:
        numeric = float(text)

        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass

    return text


def canonical_id_series(series: pd.Series) -> pd.Series:
    return series.map(canonical_id).astype("string")


def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def require_columns(
    dataframe: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    missing = sorted(
        set(required)
        - set(dataframe.columns)
    )

    if missing:
        raise RuntimeError(
            f"{label}: missing columns {missing}"
        )


def median_finite(
    dataframe: pd.DataFrame,
    column: str,
) -> float | None:
    if column not in dataframe.columns:
        return None

    values = (
        pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if values.empty:
        return None

    return float(
        np.median(
            values.to_numpy(float)
        )
    )


def determine_neutral_alpha(
    group: pd.DataFrame,
    allow_proxy: bool,
) -> tuple[float, float, str]:
    estimate_candidates = [
        "neutral_alpha_estimate",
        "alpha_neutral_estimate",
        "alpha_neutral_selected",
        "neutral_alpha",
    ]

    for column in estimate_candidates:
        value = median_finite(
            group,
            column,
        )

        if value is not None:
            uncertainty = median_finite(
                group,
                "neutral_alpha_uncertainty",
            )

            if uncertainty is None:
                uncertainty = np.nan

            return (
                value,
                uncertainty,
                column,
            )

    lower_candidates = [
        "neutral_alpha_lower",
        "alpha_neutral_lower",
    ]

    upper_candidates = [
        "neutral_alpha_upper",
        "alpha_neutral_upper",
    ]

    lower = None
    upper = None

    for column in lower_candidates:
        lower = median_finite(
            group,
            column,
        )

        if lower is not None:
            break

    for column in upper_candidates:
        upper = median_finite(
            group,
            column,
        )

        if upper is not None:
            break

    if (
        lower is not None
        and upper is not None
        and upper >= lower
    ):
        return (
            0.5 * (lower + upper),
            0.5 * (upper - lower),
            "midpoint_of_neutral_bracket",
        )

    if allow_proxy:
        maximum_alpha = float(
            numeric_series(
                group,
                "alpha",
            ).max()
        )

        return (
            maximum_alpha,
            np.nan,
            "max_available_alpha_proxy",
        )

    Mach = float(
        numeric_series(
            group,
            "Mach",
        ).iloc[0]
    )

    raise RuntimeError(
        "No neutral-alpha information for "
        f"Mach={Mach:.6f}. "
        "Use --allow-neutral-proxy only after "
        "explicit scientific review."
    )


def add_neutral_coordinates(
    spectral: pd.DataFrame,
    allow_proxy: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    rows: list[dict[str, Any]] = []

    output_parts: list[pd.DataFrame] = []

    for Mach, group in spectral.groupby(
        "Mach",
        sort=True,
    ):
        group = group.copy()

        (
            alpha_neutral,
            uncertainty,
            source,
        ) = determine_neutral_alpha(
            group,
            allow_proxy=allow_proxy,
        )

        group["alpha_neutral_used"] = (
            alpha_neutral
        )

        group[
            "alpha_neutral_uncertainty_used"
        ] = uncertainty

        group[
            "alpha_neutral_source"
        ] = source

        group["s"] = (
            group["alpha"]
            / alpha_neutral
        )

        output_parts.append(group)

        rows.append(
            {
                "Mach": float(Mach),
                "alpha_neutral_used": (
                    alpha_neutral
                ),
                "alpha_neutral_uncertainty_used": (
                    uncertainty
                ),
                "alpha_neutral_source": source,
                "alpha_min_available": float(
                    group["alpha"].min()
                ),
                "alpha_max_available": float(
                    group["alpha"].max()
                ),
                "s_min_available": float(
                    group["s"].min()
                ),
                "s_max_available": float(
                    group["s"].max()
                ),
                "n_points": int(len(group)),
            }
        )

    output = pd.concat(
        output_parts,
        ignore_index=True,
    )

    neutral_table = pd.DataFrame(rows)

    return output, neutral_table


def choose_unique_nearest_rows(
    group: pd.DataFrame,
    target_values: np.ndarray,
) -> list[pd.Series]:
    selected: list[pd.Series] = []

    used_indices: set[int] = set()

    for target in target_values:
        candidates = group.copy()

        candidates[
            "_distance_to_target"
        ] = np.abs(
            candidates["s"] - target
        )

        sort_columns = [
            "_distance_to_target",
        ]

        if "residual_norm" in candidates.columns:
            candidates[
                "_residual_sort"
            ] = pd.to_numeric(
                candidates["residual_norm"],
                errors="coerce",
            ).fillna(np.inf)

            sort_columns.append(
                "_residual_sort"
            )

        candidates = candidates.sort_values(
            sort_columns,
            kind="stable",
        )

        chosen = None

        for index, row in candidates.iterrows():
            if int(index) not in used_indices:
                chosen = row
                used_indices.add(int(index))
                break

        if chosen is None:
            raise RuntimeError(
                "Unable to choose four unique "
                "anchors for one Mach."
            )

        selected.append(chosen)

    return selected


def build_spectral_anchors(
    spectral: pd.DataFrame,
    quantiles: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for Mach, group in spectral.groupby(
        "Mach",
        sort=True,
    ):
        group = group.sort_values(
            "alpha"
        ).copy()

        s_values = group["s"].to_numpy(float)

        target_s = np.quantile(
            s_values,
            quantiles,
        )

        selected = choose_unique_nearest_rows(
            group,
            target_s,
        )

        for rank, (
            quantile,
            target,
            row,
        ) in enumerate(
            zip(
                quantiles,
                target_s,
                selected,
            ),
            start=1,
        ):
            record = row.to_dict()

            record.update(
                {
                    "anchor_rank": rank,
                    "anchor_quantile": (
                        quantile
                    ),
                    "anchor_target_s": (
                        float(target)
                    ),
                    "anchor_distance_s": abs(
                        float(row["s"])
                        - float(target)
                    ),
                }
            )

            rows.append(record)

    anchors = pd.DataFrame(rows)

    anchors = anchors.sort_values(
        ["Mach", "anchor_rank"]
    ).reset_index(drop=True)

    duplicated = anchors.duplicated(
        ["Mach", ID_COLUMN]
    )

    if duplicated.any():
        raise RuntimeError(
            "Duplicate anchor selected within "
            "the same Mach."
        )

    return anchors


def parse_mach_values(
    values: list[float],
) -> set[float]:
    return {
        round(float(value), 8)
        for value in values
    }


def assign_mach_split(
    dataframe: pd.DataFrame,
    validation_mach: set[float],
    test_mach: set[float],
) -> pd.Series:
    def classify(value: float) -> str:
        rounded = round(
            float(value),
            8,
        )

        if rounded in validation_mach:
            return "validation"

        if rounded in test_mach:
            return "test"

        return "train"

    return dataframe["Mach"].map(
        classify
    )


def build_alpha_split(
    spectral: pd.DataFrame,
    anchor_ids: set[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for Mach, group in spectral.groupby(
        "Mach",
        sort=True,
    ):
        group = group.sort_values(
            "alpha"
        ).copy()

        group[
            "is_spectral_anchor"
        ] = group[
            "_id_key"
        ].isin(anchor_ids)

        roles: list[str] = []

        audit_counter = 0

        for is_anchor in group[
            "is_spectral_anchor"
        ]:
            if bool(is_anchor):
                roles.append(
                    "train_anchor"
                )
            else:
                roles.append(
                    "validation"
                    if audit_counter % 2 == 0
                    else "test"
                )

                audit_counter += 1

        group[
            "alpha_interpolation_role"
        ] = roles

        parts.append(group)

    return pd.concat(
        parts,
        ignore_index=True,
    )


def merge_modal_index(
    manifest: pd.DataFrame,
    modal_index: pd.DataFrame,
) -> pd.DataFrame:
    modal_index = modal_index.copy()

    require_columns(
        modal_index,
        {ID_COLUMN},
        "modal index",
    )

    modal_index["_id_key"] = (
        canonical_id_series(
            modal_index[ID_COLUMN]
        )
    )

    duplicate_ids = modal_index[
        "_id_key"
    ].duplicated(keep=False)

    if duplicate_ids.any():
        duplicates = (
            modal_index.loc[
                duplicate_ids,
                "_id_key",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise RuntimeError(
            "Modal index contains duplicate "
            f"IDs: {duplicates[:10]}"
        )

    extra_columns = [
        column
        for column in modal_index.columns
        if column
        not in {
            ID_COLUMN,
            "_id_key",
        }
        and column
        not in manifest.columns
    ]

    index_subset = modal_index[
        ["_id_key", *extra_columns]
    ].copy()

    index_subset = index_subset.rename(
        columns={
            column: f"index_{column}"
            for column in extra_columns
        }
    )

    merged = manifest.merge(
        index_subset,
        on="_id_key",
        how="left",
        validate="one_to_one",
    )

    missing = merged[
        [
            column
            for column in merged.columns
            if column.startswith("index_")
        ]
    ].isna().all(axis=1)

    if missing.any():
        missing_ids = merged.loc[
            missing,
            ID_COLUMN,
        ].astype(str).tolist()

        raise RuntimeError(
            "Selected modal anchors are absent "
            f"from the modal index: {missing_ids}"
        )

    return merged


def read_selected_modal_rows(
    modal_path: Path,
    selected_ids: set[str],
    chunksize: int,
) -> dict[str, pd.DataFrame]:
    header = pd.read_csv(
        modal_path,
        nrows=0,
    )

    require_columns(
        header,
        REQUIRED_MODAL_COLUMNS,
        "modal reference",
    )

    use_columns = sorted(
        REQUIRED_MODAL_COLUMNS
        | {
            column
            for column in OPTIONAL_MODAL_COLUMNS
            if column in header.columns
        }
    )

    storage: dict[
        str,
        list[pd.DataFrame],
    ] = {
        reference_id: []
        for reference_id in selected_ids
    }

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            modal_path,
            usecols=use_columns,
            chunksize=chunksize,
            dtype={
                ID_COLUMN: "string",
            },
        ),
        start=1,
    ):
        chunk["_id_key"] = (
            canonical_id_series(
                chunk[ID_COLUMN]
            )
        )

        selected = chunk[
            chunk["_id_key"].isin(
                selected_ids
            )
        ].copy()

        if not selected.empty:
            for reference_id, group in (
                selected.groupby(
                    "_id_key",
                    sort=False,
                )
            ):
                storage[
                    str(reference_id)
                ].append(group)

        print(
            f"modal chunk {chunk_number}: "
            f"selected rows={len(selected)}",
            flush=True,
        )

    result: dict[str, pd.DataFrame] = {}

    missing: list[str] = []

    for reference_id in sorted(
        selected_ids
    ):
        pieces = storage.get(
            reference_id,
            [],
        )

        if not pieces:
            missing.append(reference_id)
            continue

        mode = pd.concat(
            pieces,
            ignore_index=True,
        )

        if "coordinate_index" in mode.columns:
            mode = mode.sort_values(
                "coordinate_index"
            )
        else:
            mode = mode.sort_values("y")

        mode = mode.reset_index(drop=True)

        result[reference_id] = mode

    if missing:
        raise RuntimeError(
            "No modal rows found for selected "
            f"IDs: {missing}"
        )

    return result


def prepare_one_mode(
    dataframe: pd.DataFrame,
    center_safe_threshold: float,
) -> dict[str, np.ndarray | float | bool]:
    y = numeric_series(
        dataframe,
        "y",
    ).to_numpy(float)

    kappa = numeric_series(
        dataframe,
        "kappa",
    ).to_numpy(float)

    q = numeric_series(
        dataframe,
        "q",
    ).to_numpy(float)

    p = (
        numeric_series(
            dataframe,
            "p_real",
        ).to_numpy(float)
        + 1j
        * numeric_series(
            dataframe,
            "p_imag",
        ).to_numpy(float)
    )

    finite = (
        np.isfinite(y)
        & np.isfinite(kappa)
        & np.isfinite(q)
        & np.isfinite(p.real)
        & np.isfinite(p.imag)
    )

    if not np.all(finite):
        raise RuntimeError(
            "Non-finite values in modal fields."
        )

    amplitude = np.abs(p)

    peak_index = int(
        np.argmax(amplitude)
    )

    center_index = int(
        np.argmin(np.abs(y))
    )

    peak_amplitude = float(
        amplitude[peak_index]
    )

    center_amplitude = float(
        amplitude[center_index]
    )

    if (
        not math.isfinite(peak_amplitude)
        or peak_amplitude <= 0.0
    ):
        raise RuntimeError(
            "Invalid pressure amplitude."
        )

    epsilon = max(
        1.0e-300,
        1.0e-12 * peak_amplitude,
    )

    log_amplitude = np.log(
        np.maximum(
            amplitude,
            epsilon,
        )
    )

    log_center = (
        log_amplitude
        - log_amplitude[center_index]
    )

    log_peak = (
        log_amplitude
        - log_amplitude[peak_index]
    )

    center_ratio = (
        center_amplitude
        / peak_amplitude
    )

    result: dict[
        str,
        np.ndarray | float | bool,
    ] = {
        "y": y,
        "kappa": kappa,
        "q": q,
        "p": p,
        "logabs_p_center_gauge": (
            log_center
        ),
        "logabs_p_peak_gauge": log_peak,
        "center_y": float(
            y[center_index]
        ),
        "peak_y": float(
            y[peak_index]
        ),
        "center_amplitude_ratio": (
            center_ratio
        ),
        "center_gauge_safe": bool(
            center_ratio
            >= center_safe_threshold
        ),
    }

    complex_fields = {
        "rho": (
            "rho_real",
            "rho_imag",
        ),
        "u": (
            "u_real",
            "u_imag",
        ),
        "v": (
            "v_real",
            "v_imag",
        ),
    }

    for name, (
        real_column,
        imaginary_column,
    ) in complex_fields.items():
        if (
            real_column in dataframe.columns
            and imaginary_column
            in dataframe.columns
        ):
            result[name] = (
                numeric_series(
                    dataframe,
                    real_column,
                ).to_numpy(float)
                + 1j
                * numeric_series(
                    dataframe,
                    imaginary_column,
                ).to_numpy(float)
            )

    if "coordinate_index" in dataframe.columns:
        result["coordinate_index"] = (
            numeric_series(
                dataframe,
                "coordinate_index",
            ).to_numpy(float)
        )

    return result


def unicode_array(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(
        [str(value) for value in values],
        dtype="U",
    )


def write_modal_bank(
    path: Path,
    manifest: pd.DataFrame,
    prepared_modes: dict[
        str,
        dict[str, np.ndarray | float | bool],
    ],
) -> dict[str, Any]:
    manifest = manifest.sort_values(
        ["Mach", "anchor_rank"]
    ).reset_index(drop=True)

    pointers = [0]

    y_values: list[np.ndarray] = []
    kappa_values: list[np.ndarray] = []
    q_values: list[np.ndarray] = []
    p_values: list[np.ndarray] = []
    center_log_values: list[np.ndarray] = []
    peak_log_values: list[np.ndarray] = []

    rho_values: list[np.ndarray] = []
    u_values: list[np.ndarray] = []
    v_values: list[np.ndarray] = []

    has_rho = True
    has_u = True
    has_v = True

    center_y: list[float] = []
    peak_y: list[float] = []
    center_ratio: list[float] = []
    center_safe: list[bool] = []

    row_counts: list[int] = []

    for _, row in manifest.iterrows():
        reference_id = str(
            row["_id_key"]
        )

        mode = prepared_modes[
            reference_id
        ]

        y = np.asarray(
            mode["y"],
            dtype=float,
        )

        n_rows = int(len(y))

        row_counts.append(n_rows)

        pointers.append(
            pointers[-1] + n_rows
        )

        y_values.append(y)

        kappa_values.append(
            np.asarray(
                mode["kappa"],
                dtype=float,
            )
        )

        q_values.append(
            np.asarray(
                mode["q"],
                dtype=float,
            )
        )

        p_values.append(
            np.asarray(
                mode["p"],
                dtype=np.complex128,
            )
        )

        center_log_values.append(
            np.asarray(
                mode[
                    "logabs_p_center_gauge"
                ],
                dtype=float,
            )
        )

        peak_log_values.append(
            np.asarray(
                mode[
                    "logabs_p_peak_gauge"
                ],
                dtype=float,
            )
        )

        center_y.append(
            float(mode["center_y"])
        )

        peak_y.append(
            float(mode["peak_y"])
        )

        center_ratio.append(
            float(
                mode[
                    "center_amplitude_ratio"
                ]
            )
        )

        center_safe.append(
            bool(
                mode[
                    "center_gauge_safe"
                ]
            )
        )

        if "rho" in mode:
            rho_values.append(
                np.asarray(
                    mode["rho"],
                    dtype=np.complex128,
                )
            )
        else:
            has_rho = False

        if "u" in mode:
            u_values.append(
                np.asarray(
                    mode["u"],
                    dtype=np.complex128,
                )
            )
        else:
            has_u = False

        if "v" in mode:
            v_values.append(
                np.asarray(
                    mode["v"],
                    dtype=np.complex128,
                )
            )
        else:
            has_v = False

    payload: dict[str, np.ndarray] = {
        "mode_ptr": np.asarray(
            pointers,
            dtype=np.int64,
        ),
        "mode_row_count": np.asarray(
            row_counts,
            dtype=np.int64,
        ),
        "mode_final_reference_id": (
            unicode_array(
                manifest[ID_COLUMN]
            )
        ),
        "mode_id_key": unicode_array(
            manifest["_id_key"]
        ),
        "Mach": manifest[
            "Mach"
        ].to_numpy(float),
        "alpha": manifest[
            "alpha"
        ].to_numpy(float),
        "s": manifest[
            "s"
        ].to_numpy(float),
        "cr": manifest[
            "cr"
        ].to_numpy(float),
        "ci": manifest[
            "ci"
        ].to_numpy(float),
        "anchor_rank": manifest[
            "anchor_rank"
        ].to_numpy(int),
        "anchor_quantile": manifest[
            "anchor_quantile"
        ].to_numpy(float),
        "anchor_target_s": manifest[
            "anchor_target_s"
        ].to_numpy(float),
        "mach_split": unicode_array(
            manifest["mach_split"]
        ),
        "y": np.concatenate(y_values),
        "kappa": np.concatenate(
            kappa_values
        ),
        "q": np.concatenate(q_values),
        "p": np.concatenate(p_values),
        "logabs_p_center_gauge": (
            np.concatenate(
                center_log_values
            )
        ),
        "logabs_p_peak_gauge": (
            np.concatenate(
                peak_log_values
            )
        ),
        "center_y": np.asarray(
            center_y,
            dtype=float,
        ),
        "peak_y": np.asarray(
            peak_y,
            dtype=float,
        ),
        "center_amplitude_ratio": (
            np.asarray(
                center_ratio,
                dtype=float,
            )
        ),
        "center_gauge_safe": np.asarray(
            center_safe,
            dtype=bool,
        ),
    }

    if has_rho:
        payload["rho"] = np.concatenate(
            rho_values
        )

    if has_u:
        payload["u"] = np.concatenate(
            u_values
        )

    if has_v:
        payload["v"] = np.concatenate(
            v_values
        )

    np.savez_compressed(
        path,
        **payload,
    )

    return {
        "path": str(path),
        "n_modes": int(len(manifest)),
        "n_modal_rows": int(
            pointers[-1]
        ),
        "n_center_gauge_unsafe": int(
            np.sum(
                ~np.asarray(
                    center_safe,
                    dtype=bool,
                )
            )
        ),
        "center_amplitude_ratio_min": (
            float(
                np.min(center_ratio)
            )
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def write_sha256sums(
    output_dir: Path,
) -> None:
    files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file()
        and path.name != "SHA256SUMS"
    )

    lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in files
    ]

    (
        output_dir
        / "SHA256SUMS"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--spectral-reference",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--modal-reference",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--modal-index",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--anchor-quantiles",
        type=float,
        nargs=4,
        default=[
            0.10,
            0.40,
            0.70,
            0.90,
        ],
    )

    parser.add_argument(
        "--validation-mach",
        type=float,
        nargs="*",
        default=[
            1.15,
            1.45,
            1.75,
        ],
    )

    parser.add_argument(
        "--test-mach",
        type=float,
        nargs="*",
        default=[
            1.25,
            1.55,
            1.85,
        ],
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--center-safe-threshold",
        type=float,
        default=1.0e-6,
    )

    parser.add_argument(
        "--allow-neutral-proxy",
        action="store_true",
    )

    parser.add_argument(
        "--skip-modal-extraction",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    for path in [
        args.spectral_reference,
        args.modal_reference,
        args.modal_index,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} already exists. "
                "Use --overwrite."
            )

        shutil.rmtree(
            args.output_dir
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    quantiles = [
        float(value)
        for value
        in args.anchor_quantiles
    ]

    if not all(
        0.0 <= value <= 1.0
        for value in quantiles
    ):
        raise ValueError(
            "Anchor quantiles must be in [0,1]."
        )

    if quantiles != sorted(quantiles):
        raise ValueError(
            "Anchor quantiles must be ordered."
        )

    validation_mach = parse_mach_values(
        args.validation_mach
    )

    test_mach = parse_mach_values(
        args.test_mach
    )

    overlap = (
        validation_mach
        & test_mach
    )

    if overlap:
        raise ValueError(
            "Mach values cannot be both "
            f"validation and test: {overlap}"
        )

    spectral = pd.read_csv(
        args.spectral_reference
    )

    require_columns(
        spectral,
        REQUIRED_SPECTRAL_COLUMNS,
        "spectral reference",
    )

    spectral["_id_key"] = (
        canonical_id_series(
            spectral[ID_COLUMN]
        )
    )

    for column in [
        "Mach",
        "alpha",
        "cr",
        "ci",
    ]:
        spectral[column] = pd.to_numeric(
            spectral[column],
            errors="raise",
        )

    spectral = spectral.sort_values(
        ["Mach", "alpha"]
    ).reset_index(drop=True)

    duplicate_points = spectral.duplicated(
        ["Mach", "alpha"]
    )

    if duplicate_points.any():
        raise RuntimeError(
            "Duplicate (Mach, alpha) points "
            "in spectral reference."
        )

    (
        spectral,
        neutral_table,
    ) = add_neutral_coordinates(
        spectral,
        allow_proxy=(
            args.allow_neutral_proxy
        ),
    )

    available_mach = {
        round(
            float(value),
            8,
        )
        for value
        in spectral["Mach"].unique()
    }

    missing_validation = (
        validation_mach
        - available_mach
    )

    missing_test = (
        test_mach
        - available_mach
    )

    if missing_validation:
        raise RuntimeError(
            "Validation Mach absent from "
            f"reference: {missing_validation}"
        )

    if missing_test:
        raise RuntimeError(
            "Test Mach absent from "
            f"reference: {missing_test}"
        )

    spectral[
        "mach_split"
    ] = assign_mach_split(
        spectral,
        validation_mach,
        test_mach,
    )

    anchors = build_spectral_anchors(
        spectral,
        quantiles,
    )

    anchors[
        "mach_split"
    ] = assign_mach_split(
        anchors,
        validation_mach,
        test_mach,
    )

    anchor_ids = set(
        anchors["_id_key"].astype(str)
    )

    alpha_split = build_alpha_split(
        spectral,
        anchor_ids,
    )

    spectral[
        "is_spectral_anchor"
    ] = spectral["_id_key"].isin(
        anchor_ids
    )

    anchor_rank_map = anchors.set_index(
        "_id_key"
    )["anchor_rank"]

    spectral[
        "anchor_rank"
    ] = spectral["_id_key"].map(
        anchor_rank_map
    )

    spectral[
        "usable_as_training_anchor"
    ] = (
        spectral[
            "is_spectral_anchor"
        ]
        & spectral[
            "mach_split"
        ].eq("train")
    )

    modal_index = pd.read_csv(
        args.modal_index
    )

    anchors = merge_modal_index(
        anchors,
        modal_index,
    )

    modal_4_all = anchors.copy()

    modal_2_all = anchors[
        anchors[
            "anchor_rank"
        ].isin([1, 4])
    ].copy()

    modal_4_train = modal_4_all[
        modal_4_all[
            "mach_split"
        ].eq("train")
    ].copy()

    modal_2_train = modal_2_all[
        modal_2_all[
            "mach_split"
        ].eq("train")
    ].copy()

    outputs = {
        "spectral_full_audit.csv": (
            spectral
        ),
        "neutral_boundary_used.csv": (
            neutral_table
        ),
        "spectral_anchors_4_per_mach_all.csv": (
            anchors
        ),
        "spectral_anchors_4_per_mach_train_machsplit.csv": (
            anchors[
                anchors[
                    "mach_split"
                ].eq("train")
            ].copy()
        ),
        "modal_anchor_manifest_2_per_mach_all.csv": (
            modal_2_all
        ),
        "modal_anchor_manifest_4_per_mach_all.csv": (
            modal_4_all
        ),
        "modal_anchor_manifest_2_per_mach_train_machsplit.csv": (
            modal_2_train
        ),
        "modal_anchor_manifest_4_per_mach_train_machsplit.csv": (
            modal_4_train
        ),
        "split_alpha_interpolation.csv": (
            alpha_split
        ),
        "split_mach_interpolation.csv": (
            spectral
        ),
    }

    for filename, dataframe in (
        outputs.items()
    ):
        dataframe.to_csv(
            args.output_dir / filename,
            index=False,
        )

    bank_reports: dict[
        str,
        dict[str, Any],
    ] = {}

    gauge_unsafe_ids: list[str] = []

    if not args.skip_modal_extraction:
        selected_ids = set(
            modal_4_all[
                "_id_key"
            ].astype(str)
        )

        modal_rows = (
            read_selected_modal_rows(
                args.modal_reference,
                selected_ids,
                chunksize=args.chunksize,
            )
        )

        prepared_modes: dict[
            str,
            dict[
                str,
                np.ndarray
                | float
                | bool,
            ],
        ] = {}

        for reference_id, mode in (
            modal_rows.items()
        ):
            prepared = prepare_one_mode(
                mode,
                center_safe_threshold=(
                    args.center_safe_threshold
                ),
            )

            prepared_modes[
                reference_id
            ] = prepared

            if not bool(
                prepared[
                    "center_gauge_safe"
                ]
            ):
                gauge_unsafe_ids.append(
                    reference_id
                )

        banks = {
            "modal_anchors_2_per_mach_all.npz": (
                modal_2_all
            ),
            "modal_anchors_4_per_mach_all.npz": (
                modal_4_all
            ),
            "modal_anchors_2_per_mach_train_machsplit.npz": (
                modal_2_train
            ),
            "modal_anchors_4_per_mach_train_machsplit.npz": (
                modal_4_train
            ),
        }

        for filename, manifest in (
            banks.items()
        ):
            bank_reports[filename] = (
                write_modal_bank(
                    args.output_dir
                    / filename,
                    manifest,
                    prepared_modes,
                )
            )

    n_mach = int(
        spectral["Mach"].nunique()
    )

    report = {
        "spectral_reference": str(
            args.spectral_reference
        ),
        "modal_reference": str(
            args.modal_reference
        ),
        "modal_index": str(
            args.modal_index
        ),
        "n_spectral_points": int(
            len(spectral)
        ),
        "n_mach": n_mach,
        "mach_min": float(
            spectral["Mach"].min()
        ),
        "mach_max": float(
            spectral["Mach"].max()
        ),
        "anchor_quantiles": quantiles,
        "n_spectral_anchors_all": int(
            len(anchors)
        ),
        "n_spectral_anchors_train_machsplit": int(
            anchors[
                "mach_split"
            ].eq("train").sum()
        ),
        "n_modal_anchors_s4m2_all": int(
            len(modal_2_all)
        ),
        "n_modal_anchors_s4m4_all": int(
            len(modal_4_all)
        ),
        "n_modal_anchors_s4m2_train_machsplit": int(
            len(modal_2_train)
        ),
        "n_modal_anchors_s4m4_train_machsplit": int(
            len(modal_4_train)
        ),
        "validation_mach": sorted(
            validation_mach
        ),
        "test_mach": sorted(
            test_mach
        ),
        "neutral_source_counts": {
            str(key): int(value)
            for key, value
            in neutral_table[
                "alpha_neutral_source"
            ].value_counts().items()
        },
        "modal_extraction_skipped": bool(
            args.skip_modal_extraction
        ),
        "center_safe_threshold": float(
            args.center_safe_threshold
        ),
        "center_gauge_unsafe_ids": (
            sorted(gauge_unsafe_ids)
        ),
        "modal_banks": bank_reports,
    }

    expected_anchors = 4 * n_mach
    expected_modal_2 = 2 * n_mach

    report["checks"] = {
        "spectral_anchor_count_ok": bool(
            len(anchors)
            == expected_anchors
        ),
        "modal_s4m2_count_ok": bool(
            len(modal_2_all)
            == expected_modal_2
        ),
        "modal_s4m4_count_ok": bool(
            len(modal_4_all)
            == expected_anchors
        ),
        "no_neutral_proxy": bool(
            not neutral_table[
                "alpha_neutral_source"
            ]
            .eq(
                "max_available_alpha_proxy"
            )
            .any()
        ),
    }

    (
        args.output_dir
        / "dataset_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    readme = f"""# Sparse supersonic PINN dataset v1

Canonical spectral reference:
{args.spectral_reference}

Canonical modal reference:
{args.modal_reference}

Canonical modal index:
{args.modal_index}

## Sparse configurations

- S4: four spectral anchors `(cr, ci)` per Mach.
- S4M2: S4 plus modal supervision at anchor ranks 1 and 4.
- S4M4: S4 plus modal supervision at all four anchors.

## Modal variables

- `kappa = Re(p_y / p)`
- `q = Im(p_y / p)`
- `logabs_p_center_gauge = log|p| - log|p(y nearest 0)|`
- `logabs_p_peak_gauge = log|p| - max_y log|p|`

The phase is not a PINN target.

## Mach split

Validation Mach:
{sorted(validation_mach)}

Test Mach:
{sorted(test_mach)}

All other Mach are training Mach.
"""

    (
        args.output_dir
        / "README.md"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    write_sha256sums(
        args.output_dir
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
