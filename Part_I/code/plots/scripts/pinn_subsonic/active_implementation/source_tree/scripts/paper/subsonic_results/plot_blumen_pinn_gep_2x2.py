#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    / "figures/Fig_Blumen_PINN_direct_GEP_2x2.png"
)


def load_blumen() -> pd.DataFrame:
    if not BLUMEN_CSV.is_file():
        raise FileNotFoundError(BLUMEN_CSV)

    frame = pd.read_csv(BLUMEN_CSV)

    required = {"Mach", "alpha", "ci"}
    missing = required.difference(frame.columns)

    if missing:
        raise KeyError(
            f"Colonnes Blumen absentes : {sorted(missing)}\n"
            f"Colonnes présentes : {list(frame.columns)}"
        )

    frame = frame.copy()

    for column in ["Mach", "alpha", "ci"]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = (
        frame
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["Mach", "alpha", "ci"])
    )

    # Dans cet asset historique, la colonne `ci` contient le niveau
    # d'isoligne de Blumen, c'est-à-dire omega_i.
    frame = frame.rename(
        columns={"ci": "omega_blumen"}
    )

    # Dans le CSV numérisé, les coordonnées géométriques sont stockées
    # dans le sens inverse de la convention utilisée dans l'article.
    # On impose ici :
    #     abscisse = Mach
    #     ordonnée = alpha
    mach_physical = frame["alpha"].copy()
    alpha_physical = frame["Mach"].copy()

    frame["Mach"] = mach_physical
    frame["alpha"] = alpha_physical

    return frame


def load_validation() -> pd.DataFrame:
    if not VALIDATION_CSV.is_file():
        raise FileNotFoundError(VALIDATION_CSV)

    frame = pd.read_csv(VALIDATION_CSV)

    required = {
        "Mach",
        "alpha",
        "ci_seed",
        "ci_final",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise KeyError(
            f"Colonnes validation absentes : {sorted(missing)}\n"
            f"Colonnes présentes : {list(frame.columns)}"
        )

    frame = frame[
        ["Mach", "alpha", "ci_seed", "ci_final"]
    ].copy()

    for column in frame.columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = (
        frame
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .groupby(
            ["Mach", "alpha"],
            as_index=False,
        )
        .mean(numeric_only=True)
    )

    frame["omega_direct"] = (
        frame["alpha"] * frame["ci_seed"]
    )

    frame["omega_gep"] = (
        frame["alpha"] * frame["ci_final"]
    )

    return frame


def build_interpolator(
    validation: pd.DataFrame,
    value_column: str,
) -> LinearNDInterpolator:
    points = validation[
        ["Mach", "alpha"]
    ].to_numpy(dtype=float)

    values = validation[
        value_column
    ].to_numpy(dtype=float)

    return LinearNDInterpolator(
        points,
        values,
        fill_value=np.nan,
    )


def evaluate_interpolator(
    interpolator: LinearNDInterpolator,
    mach: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    points = np.column_stack(
        [mach, alpha]
    )

    return np.asarray(
        interpolator(points),
        dtype=float,
    ).reshape(-1)


def make_grid(
    interpolator: LinearNDInterpolator,
    n: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mach_axis = np.linspace(0.0, 0.999, n)
    alpha_axis = np.linspace(0.0, 0.999, n)

    mach_grid, alpha_grid = np.meshgrid(
        mach_axis,
        alpha_axis,
    )

    query = np.column_stack(
        [
            mach_grid.ravel(),
            alpha_grid.ravel(),
        ]
    )

    values = np.asarray(
        interpolator(query),
        dtype=float,
    ).reshape(mach_grid.shape)

    unstable_domain = (
        alpha_grid**2 + mach_grid**2 <= 1.0
    )

    values[~unstable_domain] = np.nan

    return mach_grid, alpha_grid, values


def neutral_boundary(
    axis: plt.Axes,
) -> None:
    mach = np.linspace(0.0, 0.999, 800)
    alpha = np.sqrt(
        np.maximum(0.0, 1.0 - mach**2)
    )

    axis.plot(
        mach,
        alpha,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=r"Neutral boundary $\alpha^2+M^2=1$",
        zorder=8,
    )


def configure_axis(
    axis: plt.Axes,
) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(r"Mach number $M$")
    axis.set_ylabel(r"Wavenumber $\alpha$")
    axis.grid(alpha=0.22)


def level_colors(
    levels: np.ndarray,
) -> dict[float, tuple]:
    cmap = plt.get_cmap("tab10")

    return {
        float(level): cmap(index % 10)
        for index, level in enumerate(levels)
    }


def plot_overlay(
    axis: plt.Axes,
    *,
    blumen: pd.DataFrame,
    mach_grid: np.ndarray,
    alpha_grid: np.ndarray,
    omega_grid: np.ndarray,
    levels: np.ndarray,
    colors: dict[float, tuple],
    model_label: str,
    model_linestyle: str,
    title: str,
) -> None:
    for level in levels:
        subset = blumen.loc[
            np.isclose(
                blumen["omega_blumen"],
                level,
                rtol=0.0,
                atol=1.0e-10,
            )
        ]

        axis.scatter(
            subset["Mach"],
            subset["alpha"],
            s=22,
            color=colors[float(level)],
            alpha=0.72,
            edgecolors="none",
            zorder=5,
        )

    positive_levels = levels[
        levels > 0.0
    ]

    if positive_levels.size:
        axis.contour(
            mach_grid,
            alpha_grid,
            omega_grid,
            levels=positive_levels,
            colors=[
                colors[float(level)]
                for level in positive_levels
            ],
            linewidths=2.1,
            linestyles=model_linestyle,
            zorder=6,
        )

    neutral_boundary(axis)
    configure_axis(axis)
    axis.set_title(title)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=colors[float(level)],
            markeredgecolor="none",
            label=rf"Blumen $\omega_i={level:g}$",
        )
        for level in levels
    ]

    handles.extend(
        [
            Line2D(
                [],
                [],
                color="0.25",
                linewidth=2.1,
                linestyle=model_linestyle,
                label=model_label,
            ),
            Line2D(
                [],
                [],
                color="black",
                linewidth=1.8,
                linestyle="--",
                label=r"Neutral boundary",
            ),
        ]
    )

    axis.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.95,
        ncol=2,
    )


def plot_error_points(
    axis: plt.Axes,
    *,
    blumen: pd.DataFrame,
    error_column: str,
    norm: LogNorm,
    title: str,
):
    scatter = axis.scatter(
        blumen["Mach"],
        blumen["alpha"],
        c=np.maximum(
            blumen[error_column],
            norm.vmin,
        ),
        cmap="inferno",
        norm=norm,
        s=28,
        edgecolors="none",
        zorder=4,
    )

    threshold = float(
        blumen[error_column].quantile(0.90)
    )

    worst = blumen.loc[
        blumen[error_column] >= threshold
    ]

    axis.scatter(
        worst["Mach"],
        worst["alpha"],
        facecolors="none",
        edgecolors="cyan",
        linewidths=1.2,
        s=48,
        label="Top 10% errors",
        zorder=7,
    )

    neutral_boundary(axis)
    configure_axis(axis)
    axis.set_title(title)

    axis.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.95,
    )

    return scatter


def main() -> None:
    blumen = load_blumen()
    validation = load_validation()

    direct_interpolator = build_interpolator(
        validation,
        "omega_direct",
    )

    gep_interpolator = build_interpolator(
        validation,
        "omega_gep",
    )

    mach_blumen = blumen[
        "Mach"
    ].to_numpy(dtype=float)

    alpha_blumen = blumen[
        "alpha"
    ].to_numpy(dtype=float)

    blumen["omega_direct"] = evaluate_interpolator(
        direct_interpolator,
        mach_blumen,
        alpha_blumen,
    )

    blumen["omega_gep"] = evaluate_interpolator(
        gep_interpolator,
        mach_blumen,
        alpha_blumen,
    )

    blumen = blumen.dropna(
        subset=[
            "omega_direct",
            "omega_gep",
        ]
    ).copy()

    if blumen.empty:
        raise RuntimeError(
            "Aucun point Blumen n'est situé dans le support "
            "d'interpolation de l'atlas."
        )

    blumen["direct_abs_error"] = np.abs(
        blumen["omega_direct"]
        - blumen["omega_blumen"]
    )

    blumen["gep_abs_error"] = np.abs(
        blumen["omega_gep"]
        - blumen["omega_blumen"]
    )

    levels = np.asarray(
        sorted(
            blumen["omega_blumen"].unique()
        ),
        dtype=float,
    )

    colors = level_colors(levels)

    mach_grid_direct, alpha_grid_direct, omega_grid_direct = (
        make_grid(direct_interpolator)
    )

    mach_grid_gep, alpha_grid_gep, omega_grid_gep = (
        make_grid(gep_interpolator)
    )

    positive_errors = np.concatenate(
        [
            blumen.loc[
                blumen["direct_abs_error"] > 0.0,
                "direct_abs_error",
            ].to_numpy(dtype=float),
            blumen.loc[
                blumen["gep_abs_error"] > 0.0,
                "gep_abs_error",
            ].to_numpy(dtype=float),
        ]
    )

    if positive_errors.size:
        error_min = max(
            float(
                np.percentile(
                    positive_errors,
                    1.0,
                )
            ),
            1.0e-7,
        )

        error_max = max(
            float(np.max(positive_errors)),
            10.0 * error_min,
        )
    else:
        error_min = 1.0e-7
        error_max = 1.0e-5

    norm = LogNorm(
        vmin=error_min,
        vmax=error_max,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 12.0),
        constrained_layout=True,
    )

    # Haut gauche : PINN + GEP
    plot_overlay(
        axes[0, 0],
        blumen=blumen,
        mach_grid=mach_grid_gep,
        alpha_grid=alpha_grid_gep,
        omega_grid=omega_grid_gep,
        levels=levels,
        colors=colors,
        model_label="PINN-seeded GEP isolines",
        model_linestyle="-",
        title=(
            "(a) Digitized Blumen points vs "
            "PINN-seeded GEP"
        ),
    )

    # Haut droite : PINN direct
    plot_overlay(
        axes[0, 1],
        blumen=blumen,
        mach_grid=mach_grid_direct,
        alpha_grid=alpha_grid_direct,
        omega_grid=omega_grid_direct,
        levels=levels,
        colors=colors,
        model_label="Direct PINN isolines",
        model_linestyle="-.",
        title=(
            "(b) Digitized Blumen points vs "
            "direct PINN"
        ),
    )

    # Bas gauche : erreur GEP
    gep_scatter = plot_error_points(
        axes[1, 0],
        blumen=blumen,
        error_column="gep_abs_error",
        norm=norm,
        title=(
            r"(c) PINN-seeded GEP error at Blumen points "
            r"$|\omega_i^{\mathrm{GEP}}-\omega_i^{\mathrm{Blumen}}|$"
        ),
    )

    # Bas droite : erreur PINN direct
    plot_error_points(
        axes[1, 1],
        blumen=blumen,
        error_column="direct_abs_error",
        norm=norm,
        title=(
            r"(d) Direct PINN error at Blumen points "
            r"$|\omega_i^{\mathrm{PINN}}-\omega_i^{\mathrm{Blumen}}|$"
        ),
    )

    colorbar = figure.colorbar(
        gep_scatter,
        ax=axes[1, :],
        orientation="vertical",
        fraction=0.025,
        pad=0.025,
    )

    colorbar.set_label(
        r"Absolute error in $\omega_i$"
    )

    figure.suptitle(
        "Reconstruction of the digitized Blumen subsonic isolines",
        fontsize=18,
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

    diagnostics = blumen[
        [
            "Mach",
            "alpha",
            "omega_blumen",
            "omega_direct",
            "omega_gep",
            "direct_abs_error",
            "gep_abs_error",
        ]
    ]

    diagnostics_path = OUTPUT.with_suffix(
        ".csv"
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    print(f"PNG créé : {OUTPUT}")
    print(f"CSV diagnostic : {diagnostics_path}")
    print(f"Points Blumen utilisés : {len(blumen)}")
    print(
        "PINN direct MAE sur omega_i : "
        f"{blumen['direct_abs_error'].mean():.6e}"
    )
    print(
        "PINN-seeded GEP MAE sur omega_i : "
        f"{blumen['gep_abs_error'].mean():.6e}"
    )


if __name__ == "__main__":
    main()
