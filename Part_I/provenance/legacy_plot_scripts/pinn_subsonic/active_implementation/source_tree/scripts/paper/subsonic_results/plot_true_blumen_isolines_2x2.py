#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator


ROOT = Path(__file__).resolve().parents[9]

BLUMEN_CSV = (
    ROOT
    / "assets/pinn_subsonic/article/results_pinn/release_final"
    / "data/blumen_ci_datasets.csv"
)

VALIDATION_CSV = (
    ROOT
    / "assets/pinn_subsonic/article/results_pinn/release_final"
    / "data/validation_pointwise_canonical.csv"
)

OUTPUT = (
    ROOT
    / "assets/pinn_subsonic/article/results_pinn/release_final"
    / "figures/Fig_Blumen_classical_PINN_GEP_2x2.png"
)


def first_existing(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    lower = {
        str(column).lower(): str(column)
        for column in frame.columns
    }

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

        if candidate.lower() in lower:
            return lower[candidate.lower()]

    if required:
        raise KeyError(
            f"Aucune colonne trouvée parmi {list(candidates)}.\n"
            f"Colonnes disponibles : {list(frame.columns)}"
        )

    return None


def load_blumen() -> pd.DataFrame:
    if not BLUMEN_CSV.is_file():
        raise FileNotFoundError(
            f"Fichier Blumen introuvable : {BLUMEN_CSV}"
        )

    frame = pd.read_csv(BLUMEN_CSV)

    required = {
        "ci",
        "alpha",
        "Mach",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise KeyError(
            f"Colonnes Blumen absentes : {sorted(missing)}\n"
            f"Colonnes disponibles : {list(frame.columns)}"
        )

    for column in ["ci", "alpha", "Mach"]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = (
        frame
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["ci", "alpha", "Mach"])
        .copy()
    )

    if "source_file" not in frame.columns:
        frame["source_file"] = ""

    if "point_index" not in frame.columns:
        frame["point_index"] = np.arange(len(frame))

    return frame


def load_validation(
    blumen: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if not VALIDATION_CSV.is_file():
        raise FileNotFoundError(
            f"Validation canonique introuvable : {VALIDATION_CSV}"
        )

    raw = pd.read_csv(VALIDATION_CSV)

    mach_column = first_existing(
        raw,
        ["Mach", "M", "mach"],
    )

    alpha_column = first_existing(
        raw,
        ["alpha", "Alpha"],
    )

    seed_column = first_existing(
        raw,
        [
            "ci_seed",
            "ci_seed_target",
            "pinn_ci",
            "ci_pinn",
        ],
    )

    final_column = first_existing(
        raw,
        [
            "ci_final",
            "gep_ci",
            "ci_forward",
            "ci_gep",
        ],
    )

    reference_column = first_existing(
        raw,
        [
            "ci_classic",
            "ci_classical",
            "ci_reference",
            "ci_ref",
            "ci_shooting",
            "ci_true",
        ],
        required=False,
    )

    data = pd.DataFrame(
        {
            "Mach": pd.to_numeric(
                raw[mach_column],
                errors="coerce",
            ),
            "alpha": pd.to_numeric(
                raw[alpha_column],
                errors="coerce",
            ),
            "ci_direct": pd.to_numeric(
                raw[seed_column],
                errors="coerce",
            ),
            "ci_gep": pd.to_numeric(
                raw[final_column],
                errors="coerce",
            ),
        }
    )

    if reference_column is not None:
        data["ci_reference"] = pd.to_numeric(
            raw[reference_column],
            errors="coerce",
        )

    data = (
        data
        .replace([np.inf, -np.inf], np.nan)
        .dropna(
            subset=[
                "Mach",
                "alpha",
                "ci_direct",
                "ci_gep",
            ]
        )
        .groupby(
            ["Mach", "alpha"],
            as_index=False,
        )
        .mean(numeric_only=True)
    )

    # Interpolation des vraies courbes Blumen uniquement pour
    # identifier leur domaine de support.
    interpolation_data = (
        blumen[
            ["Mach", "alpha", "ci"]
        ]
        .groupby(
            ["Mach", "alpha"],
            as_index=False,
        )
        .mean(numeric_only=True)
    )

    blumen_interpolator = LinearNDInterpolator(
        interpolation_data[
            ["Mach", "alpha"]
        ].to_numpy(dtype=float),
        interpolation_data["ci"].to_numpy(dtype=float),
        fill_value=np.nan,
    )

    query = data[
        ["Mach", "alpha"]
    ].to_numpy(dtype=float)

    data["ci_blumen_interpolated"] = np.asarray(
        blumen_interpolator(query),
        dtype=float,
    )

    # La figure reste limitée au support couvert par Blumen.
    data = data.loc[
        np.isfinite(
            data["ci_blumen_interpolated"]
        )
    ].copy()

    if reference_column is None:
        data["ci_reference"] = (
            data["ci_blumen_interpolated"]
        )
        reference_label = "Blumen interpolation"
    else:
        data = data.dropna(
            subset=["ci_reference"]
        )
        reference_label = "Classical reference"

    if len(data) < 10:
        raise RuntimeError(
            "Trop peu de points valides dans le support Blumen : "
            f"{len(data)}"
        )

    return data, reference_label


def ordered_blumen_curves(
    blumen: pd.DataFrame,
):
    for ci_level, group in blumen.groupby(
        "ci",
        sort=True,
    ):
        yield ci_level, group.sort_values(
            ["source_file", "point_index"]
        )


def contour_levels_in_range(
    levels: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))

    selected = levels[
        (levels >= minimum)
        & (levels <= maximum)
    ]

    if selected.size == 0:
        raise RuntimeError(
            f"Aucun niveau Blumen dans [{minimum}, {maximum}]."
        )

    return selected


def configure_axis(
    axis: plt.Axes,
    *,
    alpha_max: float,
    mach_min: float,
    mach_max: float,
) -> None:
    axis.set_xlabel(r"Physical wavenumber $\alpha$")
    axis.set_ylabel(r"Mach number $M$")
    axis.set_xlim(0.0, 1.02 * alpha_max)
    axis.set_ylim(
        max(0.0, mach_min - 0.02),
        min(1.02, mach_max + 0.02),
    )
    axis.grid(alpha=0.15)


def add_raw_blumen(
    axis: plt.Axes,
    blumen: pd.DataFrame,
    *,
    color: str,
    linewidth: float,
    linestyle: str,
    alpha_value: float,
) -> None:
    for _, group in ordered_blumen_curves(
        blumen
    ):
        axis.plot(
            group["alpha"],
            group["Mach"],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha_value,
            zorder=3,
        )


def plot_comparison(
    axis: plt.Axes,
    *,
    triangulation: mtri.Triangulation,
    data: pd.DataFrame,
    blumen: pd.DataFrame,
    levels: np.ndarray,
    model_column: str,
    model_label: str,
    model_color: str,
    reference_label: str,
    title: str,
    alpha_max: float,
    mach_min: float,
    mach_max: float,
) -> None:
    reference_values = data[
        "ci_reference"
    ].to_numpy(dtype=float)

    model_values = data[
        model_column
    ].to_numpy(dtype=float)

    reference_levels = contour_levels_in_range(
        levels,
        reference_values,
    )

    model_levels = contour_levels_in_range(
        levels,
        model_values,
    )

    # Vraies courbes numérisées de Blumen.
    add_raw_blumen(
        axis,
        blumen,
        color="0.45",
        linewidth=1.7,
        linestyle=":",
        alpha_value=0.95,
    )

    reference_contours = axis.tricontour(
        triangulation,
        reference_values,
        levels=reference_levels,
        colors="tab:blue",
        linewidths=2.0,
        linestyles="-",
        zorder=4,
    )

    axis.clabel(
        reference_contours,
        inline=True,
        fontsize=7,
        fmt=lambda value: f"{value:g}",
    )

    axis.tricontour(
        triangulation,
        model_values,
        levels=model_levels,
        colors=model_color,
        linewidths=1.9,
        linestyles="--",
        zorder=5,
    )

    legend_handles = [
        Line2D(
            [],
            [],
            color="0.45",
            linewidth=1.7,
            linestyle=":",
            label="Blumen digitized",
        ),
        Line2D(
            [],
            [],
            color="tab:blue",
            linewidth=2.0,
            linestyle="-",
            label=reference_label,
        ),
        Line2D(
            [],
            [],
            color=model_color,
            linewidth=1.9,
            linestyle="--",
            label=model_label,
        ),
    ]

    axis.legend(
        handles=legend_handles,
        loc="best",
        fontsize=8,
        framealpha=0.95,
    )

    axis.set_title(title)

    configure_axis(
        axis,
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
    )


def plot_error(
    axis: plt.Axes,
    *,
    triangulation: mtri.Triangulation,
    data: pd.DataFrame,
    blumen: pd.DataFrame,
    error_column: str,
    error_levels: np.ndarray,
    error_norm: LogNorm,
    contour_levels: np.ndarray,
    title: str,
    alpha_max: float,
    mach_min: float,
    mach_max: float,
):
    values = np.maximum(
        data[error_column].to_numpy(dtype=float),
        error_norm.vmin,
    )

    error_map = axis.tricontourf(
        triangulation,
        values,
        levels=error_levels,
        norm=error_norm,
        cmap="magma",
        extend="max",
    )

    reference_values = data[
        "ci_reference"
    ].to_numpy(dtype=float)

    valid_reference_levels = (
        contour_levels_in_range(
            contour_levels,
            reference_values,
        )
    )

    axis.tricontour(
        triangulation,
        reference_values,
        levels=valid_reference_levels,
        colors="white",
        linewidths=0.65,
        alpha=0.65,
    )

    add_raw_blumen(
        axis,
        blumen,
        color="white",
        linewidth=0.8,
        linestyle=":",
        alpha_value=0.55,
    )

    axis.set_title(title)

    configure_axis(
        axis,
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
    )

    return error_map


def main() -> None:
    blumen = load_blumen()

    data, reference_label = load_validation(
        blumen
    )

    data["direct_abs_error"] = np.abs(
        data["ci_direct"]
        - data["ci_reference"]
    )

    data["gep_abs_error"] = np.abs(
        data["ci_gep"]
        - data["ci_reference"]
    )

    levels = np.asarray(
        sorted(
            blumen["ci"].unique()
        ),
        dtype=float,
    )

    triangulation = mtri.Triangulation(
        data["alpha"].to_numpy(dtype=float),
        data["Mach"].to_numpy(dtype=float),
    )

    flat_mask = mtri.TriAnalyzer(
        triangulation
    ).get_flat_tri_mask(
        min_circle_ratio=0.01
    )

    triangulation.set_mask(flat_mask)

    positive_errors = np.concatenate(
        [
            data.loc[
                data["direct_abs_error"] > 0.0,
                "direct_abs_error",
            ].to_numpy(dtype=float),
            data.loc[
                data["gep_abs_error"] > 0.0,
                "gep_abs_error",
            ].to_numpy(dtype=float),
        ]
    )

    if positive_errors.size == 0:
        error_min = 1.0e-8
        error_max = 1.0e-6
    else:
        error_min = max(
            float(
                np.nanpercentile(
                    positive_errors,
                    1.0,
                )
            ),
            1.0e-8,
        )

        error_max = max(
            float(np.nanmax(positive_errors)),
            10.0 * error_min,
        )

    error_norm = LogNorm(
        vmin=error_min,
        vmax=error_max,
    )

    error_levels = np.geomspace(
        error_min,
        error_max,
        64,
    )

    alpha_max = max(
        float(blumen["alpha"].max()),
        float(data["alpha"].max()),
    )

    mach_min = min(
        float(blumen["Mach"].min()),
        float(data["Mach"].min()),
    )

    mach_max = max(
        float(blumen["Mach"].max()),
        float(data["Mach"].max()),
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 10.0),
        constrained_layout=True,
    )

    plot_comparison(
        axes[0, 0],
        triangulation=triangulation,
        data=data,
        blumen=blumen,
        levels=levels,
        model_column="ci_direct",
        model_label="Direct PINN",
        model_color="tab:orange",
        reference_label=reference_label,
        title="(a) Blumen isolines: classical vs direct PINN",
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
    )

    direct_map = plot_error(
        axes[0, 1],
        triangulation=triangulation,
        data=data,
        blumen=blumen,
        error_column="direct_abs_error",
        error_levels=error_levels,
        error_norm=error_norm,
        contour_levels=levels,
        title=(
            r"(b) Direct PINN absolute error "
            r"$|c_i^{\mathrm{PINN}}-c_i^{\mathrm{ref}}|$"
        ),
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
    )

    plot_comparison(
        axes[1, 0],
        triangulation=triangulation,
        data=data,
        blumen=blumen,
        levels=levels,
        model_column="ci_gep",
        model_label="PINN-seeded GEP",
        model_color="black",
        reference_label=reference_label,
        title="(c) Blumen isolines: classical vs PINN-seeded GEP",
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
    )

    plot_error(
        axes[1, 1],
        triangulation=triangulation,
        data=data,
        blumen=blumen,
        error_column="gep_abs_error",
        error_levels=error_levels,
        error_norm=error_norm,
        contour_levels=levels,
        title=(
            r"(d) PINN-seeded GEP absolute error "
            r"$|c_i^{\mathrm{GEP}}-c_i^{\mathrm{ref}}|$"
        ),
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
    )

    colorbar = figure.colorbar(
        direct_map,
        ax=axes[:, 1],
        fraction=0.045,
        pad=0.025,
    )

    colorbar.set_label(
        r"Absolute error in $c_i$"
    )

    figure.suptitle(
        "Reconstruction of the digitized Blumen subsonic isolines",
        fontsize=17,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"PNG créé : {OUTPUT}")
    print(f"Référence utilisée : {reference_label}")
    print(f"Points dans le support Blumen : {len(data)}")
    print(
        "Direct PINN MAE : "
        f"{data['direct_abs_error'].mean():.6e}"
    )
    print(
        "PINN-seeded GEP MAE : "
        f"{data['gep_abs_error'].mean():.6e}"
    )


if __name__ == "__main__":
    main()
