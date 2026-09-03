#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from classic_supersonic_reference.io.blumen_reference import supersonic_digitized_x_to_mach


REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "classic_supersonic"

SPECTRAL = (
    PACKAGE
    / "data/spectral/"
    "supersonic_reference_v2_spectral.csv"
)

BLUMEN = (
    PACKAGE
    / "data/blumen/"
    "blumen_ci_digitized_points.csv"
)

BLUMEN_LEVELS = (
    PACKAGE
    / "data/blumen/"
    "blumen_ci_curve_levels.csv"
)

MODAL_RAW = (
    PACKAGE
    / "data/modal/"
    "supersonic_reference_v2_modal_raw.parquet"
)

RESULTS = (
    PACKAGE
    / "reproducibility/results"
)

WITNESS_SPECTRAL = (
    RESULTS
    / "witness_M150_a01625_spectral_reproduction.json"
)

WITNESS_MODAL = (
    RESULTS
    / "witness_M150_a01625_modal_calibration.json"
)

PUBLIC = (
    REPO
    / "assets/classic_supersonic/reference_v2"
)

INTERNAL = (
    PACKAGE
    / "assets/reference_v2"
)


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def normalized_name(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        value.lower(),
    )


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> str:
    lookup = {
        normalized_name(column): column
        for column in frame.columns
    }

    for candidate in candidates:
        match = lookup.get(
            normalized_name(candidate)
        )

        if match is not None:
            return match

    raise KeyError(
        "Expected one of these columns:\n"
        f"{list(candidates)}\n"
        "Observed columns:\n"
        f"{list(frame.columns)}"
    )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def truthy(
    frame: pd.DataFrame,
    column: str | None,
) -> pd.Series:
    if column is None:
        return pd.Series(
            False,
            index=frame.index,
        )

    values = frame[column]

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)

    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "1",
                "1.0",
                "true",
                "yes",
                "y",
            }
        )
    )


def save_figure(
    figure: plt.Figure,
    relative_stem: Path,
) -> None:
    stem = PUBLIC / relative_stem

    stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")

    figure.savefig(
        pdf,
        bbox_inches="tight",
    )

    figure.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Wrote:", pdf.relative_to(REPO))
    print("Wrote:", png.relative_to(REPO))


def write_table(
    frame: pd.DataFrame,
    relative_path: Path,
) -> None:
    path = PUBLIC / relative_path

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
    )

    print("Wrote:", path.relative_to(REPO))


def load_spectral() -> pd.DataFrame:
    if not SPECTRAL.is_file():
        raise FileNotFoundError(SPECTRAL)

    frame = pd.read_csv(SPECTRAL)

    required = [
        "Mach",
        "alpha",
        "cr",
        "ci",
        "omega_i",
        "validation_status",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"Missing spectral columns: {missing}"
        )

    for column in [
        "Mach",
        "alpha",
        "cr",
        "ci",
        "omega_i",
    ]:
        frame[column] = numeric(
            frame[column]
        )

    return frame


def article_validation_category(
    frame: pd.DataFrame,
) -> pd.Series:
    status = (
        frame["validation_status"]
        .fillna("")
        .astype(str)
    )

    boundary_column = (
        "boundary_flag"
        if "boundary_flag" in frame.columns
        else None
    )

    exported_column = (
        "has_exported_modal_fields"
        if "has_exported_modal_fields"
        in frame.columns
        else None
    )

    boundary = (
        truthy(frame, boundary_column)
        | status.str.contains(
            "boundary_flag",
            case=False,
            regex=False,
        )
    )

    tail = status.str.contains(
        "tail_sensitive",
        case=False,
        regex=False,
    )

    exported = (
        truthy(frame, exported_column)
        | status.eq(
            "modal_spectral_validated_with_exported_fields"
        )
    )

    category = pd.Series(
        "other validated",
        index=frame.index,
        dtype="object",
    )

    category.loc[exported] = (
        "modal fields exported"
    )

    category.loc[tail] = (
        "tail-sensitive"
    )

    category.loc[boundary] = (
        "boundary-sensitive"
    )

    return category


def build_validation_map(
    spectral: pd.DataFrame,
) -> None:
    frame = spectral.copy()

    frame["article_category"] = (
        article_validation_category(frame)
    )

    order = [
        "modal fields exported",
        "other validated",
        "tail-sensitive",
        "boundary-sensitive",
    ]

    markers = {
        "modal fields exported": "o",
        "other validated": "s",
        "tail-sensitive": "^",
        "boundary-sensitive": "X",
    }

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8),
        constrained_layout=True,
    )

    for category in order:
        selected = frame[
            frame["article_category"].eq(
                category
            )
        ]

        if selected.empty:
            continue

        axis.scatter(
            selected["Mach"],
            selected["alpha"],
            marker=markers[category],
            s=42,
            label=(
                f"{category} "
                f"(n={len(selected)})"
            ),
        )

    axis.set_xlabel("Mach number")
    axis.set_ylabel(
        r"Wavenumber $\alpha$"
    )

    axis.set_title(
        "Classical supersonic reference: "
        "validation coverage"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend(
        frameon=False,
        ncols=2,
    )

    save_figure(
        figure,
        Path(
            "ci/"
            "supersonic_reference_validation_status_map"
        ),
    )

    summary = (
        frame.groupby(
            [
                "article_category",
                "validation_status",
            ],
            dropna=False,
        )
        .size()
        .rename("n_points")
        .reset_index()
        .sort_values(
            [
                "article_category",
                "validation_status",
            ]
        )
    )

    write_table(
        summary,
        Path(
            "tables/"
            "supersonic_reference_v2_"
            "validation_summary.csv"
        ),
    )


def build_blumen_comparison(
    spectral: pd.DataFrame,
) -> None:
    if not BLUMEN.is_file():
        raise FileNotFoundError(BLUMEN)

    blumen = pd.read_csv(BLUMEN)

    mach_column = find_column(
        blumen,
        [
            "Mach",
            "M",
            "mach_number",
        ],
    )

    alpha_column = find_column(
        blumen,
        [
            "alpha",
            "a",
            "wavenumber",
        ],
    )

    curve_column = find_column(
        blumen,
        [
            "curve_id",
            "curve",
            "contour_id",
        ],
    )

    if not BLUMEN_LEVELS.is_file():
        raise FileNotFoundError(
            BLUMEN_LEVELS
        )

    levels = pd.read_csv(
        BLUMEN_LEVELS
    )

    required_level_columns = {
        "curve_id",
        "curve_label",
        "family",
        "ci_level",
        "include_in_quantitative_comparison",
    }

    missing_level_columns = (
        required_level_columns
        - set(levels.columns)
    )

    if missing_level_columns:
        raise KeyError(
            "Missing Blumen-level columns: "
            f"{sorted(missing_level_columns)}"
        )

    points = blumen.copy()

    points["curve_id_key"] = (
        pd.to_numeric(
            points[curve_column],
            errors="coerce",
        )
        .astype("Int64")
    )

    levels["curve_id_key"] = (
        pd.to_numeric(
            levels["curve_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    levels["ci_level"] = numeric(
        levels["ci_level"]
    )

    levels[
        "include_in_quantitative_comparison"
    ] = (
        levels[
            "include_in_quantitative_comparison"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
    )

    points = points.merge(
        levels[
            [
                "curve_id_key",
                "curve_label",
                "family",
                "ci_level",
                "include_in_quantitative_comparison",
            ]
        ],
        on="curve_id_key",
        how="left",
        validate="many_to_one",
    )

    unmapped = (
        points.loc[
            points["curve_label"].isna(),
            curve_column,
        ]
        .drop_duplicates()
        .tolist()
    )

    if unmapped:
        raise RuntimeError(
            "Unmapped Blumen curve identifiers: "
            f"{unmapped}"
        )

    print()
    print("Blumen curve mapping:")
    print(
        levels[
            [
                "curve_id",
                "curve_label",
                "family",
                "ci_level",
                "include_in_quantitative_comparison",
            ]
        ]
        .sort_values("curve_id")
        .to_string(index=False)
    )
    print()

    excluded_counts = (
        points.loc[
            ~points[
                "include_in_quantitative_comparison"
            ]
        ]
        .groupby(
            [
                "curve_id_key",
                "curve_label",
                "family",
            ],
            dropna=False,
        )
        .size()
        .rename("n_points")
        .reset_index()
    )

    print("Excluded from quantitative ci comparison:")
    print(
        excluded_counts.to_string(
            index=False
        )
    )
    print()

    selected = points.loc[
        points[
            "include_in_quantitative_comparison"
        ]
        & points["ci_level"].notna()
    ].copy()

    mach_input = numeric(
        selected[mach_column]
    ).to_numpy(
        dtype=float
    )

    alpha_input = numeric(
        selected[alpha_column]
    ).to_numpy(
        dtype=float
    )

    reference_coordinates = (
        spectral[
            [
                "Mach",
                "alpha",
            ]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna()
        .drop_duplicates()
        .to_numpy(
            dtype=float
        )
    )

    hull = Delaunay(
        reference_coordinates
    )

    mach_candidates = {
        "identity": mach_input,
        "digitized_x_plus_0p9": (
            supersonic_digitized_x_to_mach(
                mach_input
            )
        ),
    }

    candidate_inside = {}

    for calibration, candidate in (
        mach_candidates.items()
    ):
        coordinates = np.column_stack(
            [
                candidate,
                alpha_input,
            ]
        )

        finite = np.isfinite(
            coordinates
        ).all(
            axis=1
        )

        inside = np.zeros(
            len(coordinates),
            dtype=bool,
        )

        inside[finite] = (
            hull.find_simplex(
                coordinates[finite],
                tol=1.0e-12,
            )
            >= 0
        )

        candidate_inside[
            calibration
        ] = inside

    candidate_counts = {
        calibration: int(
            inside.sum()
        )
        for calibration, inside
        in candidate_inside.items()
    }

    print(
        "Blumen Mach calibration "
        "candidate coverage:"
    )

    for calibration, count in (
        candidate_counts.items()
    ):
        candidate = mach_candidates[
            calibration
        ]

        print(
            f"  {calibration}: "
            f"{count}/{len(candidate)} "
            "inside the reference hull; "
            f"Mach range "
            f"[{np.nanmin(candidate):.6g}, "
            f"{np.nanmax(candidate):.6g}]"
        )

    mach_calibration = max(
        candidate_counts,
        key=lambda calibration: (
            candidate_counts[
                calibration
            ],
            calibration == "identity",
        ),
    )

    mach_physical = mach_candidates[
        mach_calibration
    ]

    print(
        "Selected Blumen Mach calibration:",
        mach_calibration,
    )
    print()

    comparison_data = {
        "curve_id": (
            selected["curve_id_key"]
            .astype(int)
        ),
        "curve_label": (
            selected["curve_label"]
            .astype(str)
        ),
        "family": (
            selected["family"]
            .astype(str)
        ),
        "Mach_input": mach_input,
        "Mach": mach_physical,
        "mach_calibration": (
            mach_calibration
        ),
        "alpha": numeric(
            selected[alpha_column]
        ),
        "ci_blumen": numeric(
            selected["ci_level"]
        ),
    }

    if "source" in selected.columns:
        comparison_data["source"] = (
            selected["source"].astype(str)
        )

    comparison = (
        pd.DataFrame(
            comparison_data
        )
        .dropna(
            subset=[
                "Mach",
                "alpha",
                "ci_blumen",
            ]
        )
        .reset_index(drop=True)
    )

    reference = (
        spectral[
            [
                "Mach",
                "alpha",
                "ci",
            ]
        ]
        .dropna()
        .groupby(
            [
                "Mach",
                "alpha",
            ],
            as_index=False,
        )["ci"]
        .mean()
    )

    interpolator = (
        LinearNDInterpolator(
            reference[
                [
                    "Mach",
                    "alpha",
                ]
            ].to_numpy(),
            reference["ci"].to_numpy(),
            fill_value=np.nan,
        )
    )

    comparison[
        "ci_reference_interpolated"
    ] = interpolator(
        comparison["Mach"].to_numpy(),
        comparison["alpha"].to_numpy(),
    )

    comparison[
        "inside_reference_hull"
    ] = comparison[
        "ci_reference_interpolated"
    ].notna()

    comparison["delta_ci"] = (
        comparison[
            "ci_reference_interpolated"
        ]
        - comparison["ci_blumen"]
    )

    comparison["abs_delta_ci"] = (
        comparison["delta_ci"].abs()
    )

    write_table(
        comparison,
        Path(
            "tables/"
            "supersonic_blumen_"
            "interpolated_comparison.csv"
        ),
    )

    valid = comparison[
        comparison[
            "inside_reference_hull"
        ]
    ].copy()

    if len(valid) < 3:
        raise RuntimeError(
            "Fewer than three Blumen points "
            "are inside the reference convex hull."
        )

    rmse = math.sqrt(
        float(
            np.mean(
                valid["delta_ci"] ** 2
            )
        )
    )

    mae = float(
        valid["abs_delta_ci"].mean()
    )

    lower = float(
        min(
            valid["ci_blumen"].min(),
            valid[
                "ci_reference_interpolated"
            ].min(),
        )
    )

    upper = float(
        max(
            valid["ci_blumen"].max(),
            valid[
                "ci_reference_interpolated"
            ].max(),
        )
    )

    if np.isclose(lower, upper):
        lower -= 1.0e-3
        upper += 1.0e-3

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.2),
        constrained_layout=True,
    )

    scatter = axes[0].scatter(
        valid["ci_blumen"],
        valid[
            "ci_reference_interpolated"
        ],
        c=valid["alpha"],
        s=24,
    )

    axes[0].plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1,
    )

    axes[0].set_xlim(
        lower,
        upper,
    )

    axes[0].set_ylim(
        lower,
        upper,
    )

    axes[0].set_xlabel(
        r"Digitized Blumen $c_i$"
    )

    axes[0].set_ylabel(
        r"Interpolated reference $c_i$"
    )

    axes[0].set_title(
        "Pointwise comparison"
    )

    axes[0].grid(
        True,
        alpha=0.25,
    )

    figure.colorbar(
        scatter,
        ax=axes[0],
        label=r"$\alpha$",
    )

    axes[1].scatter(
        valid["Mach"],
        valid["delta_ci"],
        c=valid["alpha"],
        s=24,
    )

    axes[1].axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    axes[1].set_xlabel(
        "Mach number"
    )

    axes[1].set_ylabel(
        r"$c_i^{reference}"
        r"-c_i^{Blumen}$"
    )

    axes[1].set_title(
        f"RMSE={rmse:.2e}, "
        f"MAE={mae:.2e}"
    )

    axes[1].grid(
        True,
        alpha=0.25,
    )

    figure.suptitle(
        "Comparison at digitized Blumen locations\n"
        "Linear interpolation of the present "
        "reference inside its convex hull; "
        f"n={len(valid)}",
        fontsize=11,
    )

    save_figure(
        figure,
        Path(
            "ci/"
            "supersonic_blumen_"
            "quantitative_comparison"
        ),
    )


def load_modal_reference() -> pd.DataFrame:
    if not MODAL_RAW.is_file():
        raise FileNotFoundError(
            "Missing local raw modal Parquet:\n"
            f"{MODAL_RAW}"
        )

    columns = [
        "Mach",
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
    ]

    frame = pd.read_parquet(
        MODAL_RAW,
        columns=columns,
    )

    for column in columns:
        frame[column] = numeric(
            frame[column]
        )

    return frame.dropna(
        subset=[
            "Mach",
            "alpha",
            "y",
        ]
    )


def choose_representative_points(
    spectral: pd.DataFrame,
    modal: pd.DataFrame,
    target_machs: list[float],
) -> pd.DataFrame:
    exported_column = (
        "has_exported_modal_fields"
        if "has_exported_modal_fields"
        in spectral.columns
        else None
    )

    exported = (
        truthy(
            spectral,
            exported_column,
        )
        | spectral[
            "validation_status"
        ].eq(
            "modal_spectral_validated_with_exported_fields"
        )
    )

    candidates = spectral[
        exported
    ].copy()

    available_machs = list(
        sorted(
            candidates[
                "Mach"
            ].dropna().unique()
        )
    )

    selected_rows = []

    for target in target_machs:
        if not available_machs:
            break

        mach = min(
            available_machs,
            key=lambda value: abs(
                float(value) - target
            ),
        )

        possible = (
            candidates[
                np.isclose(
                    candidates["Mach"],
                    mach,
                    atol=1.0e-12,
                )
            ]
            .sort_values(
                [
                    "ci",
                    "alpha",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )

        chosen = None

        for _, row in possible.iterrows():
            present = modal[
                np.isclose(
                    modal["Mach"],
                    float(row["Mach"]),
                    atol=1.0e-12,
                )
                & np.isclose(
                    modal["alpha"],
                    float(row["alpha"]),
                    atol=1.0e-12,
                )
            ]

            if not present.empty:
                chosen = row
                break

        if chosen is not None:
            selected_rows.append(chosen)
            available_machs.remove(mach)

    if len(selected_rows) < 3:
        raise RuntimeError(
            "Fewer than three representative "
            "modal points could be matched."
        )

    return pd.DataFrame(
        selected_rows
    ).reset_index(
        drop=True
    )


def build_representative_modes(
    spectral: pd.DataFrame,
    target_machs: list[float],
) -> None:
    modal = load_modal_reference()

    selected = choose_representative_points(
        spectral,
        modal,
        target_machs,
    )

    table_columns = [
        column
        for column in [
            "point_id",
            "Mach",
            "alpha",
            "cr",
            "ci",
            "omega_i",
            "validation_status",
            "boundary_flag",
        ]
        if column in selected.columns
    ]

    write_table(
        selected[table_columns],
        Path(
            "tables/"
            "supersonic_representative_"
            "mode_points.csv"
        ),
    )

    fields = [
        "p",
        "rho",
        "u",
        "v",
    ]

    n_rows = len(selected)

    figure, axes = plt.subplots(
        n_rows,
        len(fields),
        figsize=(
            12.0,
            2.25 * n_rows,
        ),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, point in (
        selected.iterrows()
    ):
        point_modal = modal[
            np.isclose(
                modal["Mach"],
                float(point["Mach"]),
                atol=1.0e-12,
            )
            & np.isclose(
                modal["alpha"],
                float(point["alpha"]),
                atol=1.0e-12,
            )
        ].sort_values("y")

        envelope = np.zeros(
            len(point_modal),
            dtype=float,
        )

        values_by_field = {}

        for field in fields:
            values = (
                point_modal[
                    f"{field}_real"
                ].to_numpy()
                + 1j
                * point_modal[
                    f"{field}_imag"
                ].to_numpy()
            )

            values_by_field[field] = values

            peak = float(
                np.nanmax(
                    np.abs(values)
                )
            )

            if peak > 0.0:
                envelope = np.maximum(
                    envelope,
                    np.abs(values) / peak,
                )

        core = envelope >= 1.0e-2

        if np.any(core):
            core_y = point_modal.loc[
                core,
                "y",
            ]

            y_limit = min(
                200.0,
                max(
                    10.0,
                    1.10
                    * float(
                        np.nanmax(
                            np.abs(core_y)
                        )
                    ),
                ),
            )
        else:
            y_limit = 80.0

        y = point_modal[
            "y"
        ].to_numpy()

        window = (
            np.abs(y) <= y_limit
        )

        for column_index, field in enumerate(
            fields
        ):
            axis = axes[
                row_index,
                column_index,
            ]

            values = values_by_field[
                field
            ]

            peak = float(
                np.nanmax(
                    np.abs(values)
                )
            )

            if (
                not np.isfinite(peak)
                or peak <= 0.0
            ):
                peak = 1.0

            axis.plot(
                y[window],
                values.real[window] / peak,
                label="Re",
            )

            axis.plot(
                y[window],
                np.abs(
                    values[window]
                ) / peak,
                linestyle="--",
                label="|q|",
            )

            axis.axhline(
                0.0,
                linewidth=0.7,
            )

            axis.set_ylim(
                -1.08,
                1.08,
            )

            axis.grid(
                True,
                alpha=0.2,
            )

            if row_index == 0:
                axis.set_title(field)

            if row_index == n_rows - 1:
                axis.set_xlabel("y")

            if column_index == 0:
                axis.set_ylabel(
                    "normalized\n"
                    f"M={point['Mach']:.2f}\n"
                    f"α={point['alpha']:.5g}\n"
                    f"ci={point['ci']:.4g}"
                )

            if (
                row_index == 0
                and column_index == 0
            ):
                axis.legend(
                    frameon=False,
                )

    figure.suptitle(
        "Representative raw-confirmed modal fields\n"
        "Highest-growth exported point at each "
        "selected Mach; each field normalized "
        "by its own peak",
        fontsize=12,
    )

    save_figure(
        figure,
        Path(
            "modes/"
            "supersonic_representative_"
            "modes_core"
        ),
    )


def build_witness_metrics() -> None:
    if not WITNESS_SPECTRAL.is_file():
        raise FileNotFoundError(
            "Run witness-solver-smoke first:\n"
            f"{WITNESS_SPECTRAL}"
        )

    if not WITNESS_MODAL.is_file():
        raise FileNotFoundError(
            "Run witness-modal-smoke first:\n"
            f"{WITNESS_MODAL}"
        )

    spectral = json.loads(
        WITNESS_SPECTRAL.read_text()
    )

    modal = json.loads(
        WITNESS_MODAL.read_text()
    )

    ratio_definitions = [
        (
            "abs_cr_error",
            "abs_cr_error_max",
            r"$|\Delta c_r|$",
        ),
        (
            "abs_ci_error",
            "abs_ci_error_max",
            r"$|\Delta c_i|$",
        ),
        (
            "abs_omega_i_error",
            "abs_omega_i_error_max",
            r"$|\Delta \omega_i|$",
        ),
        (
            "stage1_mismatch_solved",
            "stage1_mismatch_max",
            "stage-1 mismatch",
        ),
    ]

    ratios = []

    labels = []

    for metric, limit, label in (
        ratio_definitions
    ):
        ratios.append(
            float(
                spectral["metrics"][
                    metric
                ]
            )
            / float(
                spectral["acceptance"][
                    limit
                ]
            )
        )

        labels.append(label)

    variants = {
        item["variant"]: item
        for item in modal[
            "comparison"
        ]["variants"]
    }

    variant_order = [
        name
        for name in [
            "solved_spectral_root",
            "frozen_spectral_values",
        ]
        if name in variants
    ]

    fields = [
        "rho",
        "u",
        "v",
        "p",
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 4.2),
        constrained_layout=True,
    )

    positions = np.arange(
        len(labels)
    )

    axes[0].bar(
        positions,
        ratios,
    )

    axes[0].axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        label="acceptance limit",
    )

    axes[0].set_yscale("log")

    axes[0].set_xticks(
        positions,
        labels,
        rotation=25,
        ha="right",
    )

    axes[0].set_ylabel(
        "metric / acceptance limit"
    )

    axes[0].set_title(
        "Spectral reproduction"
    )

    axes[0].grid(
        True,
        axis="y",
        alpha=0.25,
    )

    axes[0].legend(
        frameon=False,
    )

    x = np.arange(
        len(fields)
    )

    width = 0.36

    metric_rows = []

    for variant_index, variant_name in (
        enumerate(variant_order)
    ):
        variant = variants[
            variant_name
        ]

        core_errors = []

        correlation_defects = []

        for field in fields:
            metrics = variant[
                "fields"
            ][field]

            core_error = float(
                metrics[
                    "relative_l2_core_amp_ge_5pct"
                ]
            )

            correlation = float(
                metrics[
                    "complex_correlation_core"
                ]
            )

            defect = max(
                1.0e-16,
                1.0 - correlation,
            )

            core_errors.append(
                core_error
            )

            correlation_defects.append(
                defect
            )

            metric_rows.append(
                {
                    "variant": variant_name,
                    "field": field,
                    "relative_l2_core_amp_ge_5pct": (
                        core_error
                    ),
                    "complex_correlation_core": (
                        correlation
                    ),
                    "one_minus_complex_correlation_core": (
                        defect
                    ),
                    "alignment_absolute": float(
                        variant[
                            "alignment"
                        ]["absolute"]
                    ),
                }
            )

        offset = (
            variant_index
            - (
                len(variant_order) - 1
            ) / 2
        ) * width

        label = variant_name.replace(
            "_",
            " ",
        )

        axes[1].bar(
            x + offset,
            core_errors,
            width=width,
            label=label,
        )

        axes[2].bar(
            x + offset,
            correlation_defects,
            width=width,
            label=label,
        )

    axes[1].set_yscale("log")
    axes[1].set_xticks(x, fields)

    axes[1].set_ylabel(
        "relative core L2 error"
    )

    axes[1].set_title(
        "Modal reconstruction error"
    )

    axes[1].grid(
        True,
        axis="y",
        alpha=0.25,
    )

    axes[1].legend(
        frameon=False,
    )

    axes[2].set_yscale("log")
    axes[2].set_xticks(x, fields)

    axes[2].set_ylabel(
        "1 − complex correlation"
    )

    axes[2].set_title(
        "Modal correlation defect"
    )

    axes[2].grid(
        True,
        axis="y",
        alpha=0.25,
    )

    axes[2].legend(
        frameon=False,
    )

    case = spectral["case"]

    figure.suptitle(
        "Witness reproduction: "
        f"M={case['Mach']:.2f}, "
        f"α={case['alpha']:.5g}\n"
        "Raw-confirmed modal reference; "
        "comparison restricted to the "
        "reconstructed overlap",
        fontsize=12,
    )

    save_figure(
        figure,
        Path(
            "reproducibility/"
            "supersonic_witness_"
            "reproduction_M150_a01625"
        ),
    )

    write_table(
        pd.DataFrame(
            metric_rows
        ),
        Path(
            "tables/"
            "supersonic_witness_"
            "reproduction_metrics.csv"
        ),
    )


def build_workflow() -> None:
    labels = [
        "Reference inputs\n"
        "Blumen + campaign data",
        "Candidate spectrum\n"
        "GEP / continuation",
        "Spectral shooting\n"
        "root refinement",
        "Modal reconstruction\n"
        "p, ρ, u, v",
        "Validation\n"
        "core, tails, boundaries",
        "Frozen reference\n"
        "assets, tables, tests",
    ]

    figure, axis = plt.subplots(
        figsize=(12.0, 2.8),
        constrained_layout=True,
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.axis("off")

    width = 0.135
    height = 0.38
    y = 0.31

    x_positions = np.linspace(
        0.015,
        0.85,
        len(labels),
    )

    for index, (x, label) in enumerate(
        zip(
            x_positions,
            labels,
        )
    ):
        box = FancyBboxPatch(
            (
                x,
                y,
            ),
            width,
            height,
            boxstyle="round,pad=0.015",
            linewidth=1.0,
            fill=False,
        )

        axis.add_patch(box)

        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
        )

        if index < len(labels) - 1:
            arrow = FancyArrowPatch(
                (
                    x + width,
                    y + height / 2,
                ),
                (
                    x_positions[
                        index + 1
                    ],
                    y + height / 2,
                ),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.0,
            )

            axis.add_patch(arrow)

    axis.set_title(
        "Classical supersonic reference "
        "generation and validation workflow",
        fontsize=12,
    )

    save_figure(
        figure,
        Path(
            "method/"
            "supersonic_reference_"
            "generation_workflow"
        ),
    )


def sync_public_to_internal() -> None:
    for path in PUBLIC.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(
            PUBLIC
        )

        destination = (
            INTERNAL
            / relative
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            path,
            destination,
        )

    print(
        "Synchronized:",
        PUBLIC.relative_to(REPO),
        "->",
        INTERNAL.relative_to(REPO),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--representative-machs",
        nargs="+",
        type=float,
        default=[
            1.10,
            1.30,
            1.50,
            1.80,
            1.90,
        ],
    )

    parser.add_argument(
        "--skip-modes",
        action="store_true",
    )

    arguments = parser.parse_args()

    spectral = load_spectral()

    build_validation_map(
        spectral
    )

    build_blumen_comparison(
        spectral
    )

    if not arguments.skip_modes:
        build_representative_modes(
            spectral,
            arguments.representative_machs,
        )

    build_witness_metrics()
    build_workflow()
    sync_public_to_internal()


if __name__ == "__main__":
    main()
