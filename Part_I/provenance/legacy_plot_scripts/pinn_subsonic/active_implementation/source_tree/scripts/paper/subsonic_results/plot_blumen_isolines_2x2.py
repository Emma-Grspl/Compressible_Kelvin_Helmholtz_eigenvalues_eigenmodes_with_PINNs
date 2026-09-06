#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


INPUT = Path(
    "assets/pinn_subsonic/csv/article/results_pinn/"
    "release_final/data/Table_Blumen_interpolated_comparison.csv"
)

OUTPUT = Path(
    "assets/pinn_subsonic/article/results_pinn/"
    "release_final/figures/Fig_blumen_isolines_2x2.png"
)


def add_neutral_boundary(axis: plt.Axes) -> None:
    mach = np.linspace(0.0, 0.999, 800)
    alpha = np.sqrt(1.0 - mach**2)

    axis.plot(
        mach,
        alpha,
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        label="Neutral boundary",
    )


def configure_axis(axis: plt.Axes) -> None:
    axis.set_xlabel(r"$M$")
    axis.set_ylabel(r"$\alpha$")
    axis.grid(alpha=0.20)


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {INPUT}")

    data = pd.read_csv(INPUT)

    required = {
        "Mach",
        "alpha",
        "ci_seed",
        "ci_final",
        "ci_blumen",
    }

    missing = required.difference(data.columns)

    if missing:
        raise KeyError(
            f"Colonnes absentes : {sorted(missing)}\n"
            f"Colonnes présentes : {list(data.columns)}"
        )

    if "inside_blumen_support" in data.columns:
        support = data["inside_blumen_support"]

        if support.dtype == bool:
            data = data.loc[support]
        else:
            support_text = (
                support.astype(str)
                .str.strip()
                .str.lower()
            )

            data = data.loc[
                support_text.isin(
                    {"true", "1", "yes", "y"}
                )
            ]

    columns = [
        "Mach",
        "alpha",
        "ci_seed",
        "ci_final",
        "ci_blumen",
    ]

    data = data[columns].copy()

    for column in columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .groupby(
            ["Mach", "alpha"],
            as_index=False,
        )
        .mean(numeric_only=True)
    )

    if len(data) < 3:
        raise RuntimeError(
            f"Seulement {len(data)} points Blumen valides."
        )

    mach = data["Mach"].to_numpy(float)
    alpha = data["alpha"].to_numpy(float)

    ci_blumen = data["ci_blumen"].to_numpy(float)
    ci_direct = data["ci_seed"].to_numpy(float)
    ci_gep = data["ci_final"].to_numpy(float)

    error_direct = np.abs(ci_direct - ci_blumen)
    error_gep = np.abs(ci_gep - ci_blumen)

    triangulation = mtri.Triangulation(
        mach,
        alpha,
    )

    ci_min = float(
        np.nanmin(
            np.concatenate(
                [ci_blumen, ci_direct, ci_gep]
            )
        )
    )

    ci_max = float(
        np.nanmax(
            np.concatenate(
                [ci_blumen, ci_direct, ci_gep]
            )
        )
    )

    if np.isclose(ci_min, ci_max):
        raise RuntimeError(
            "Les valeurs de ci ne couvrent pas un intervalle."
        )

    contour_levels = np.linspace(
        ci_min,
        ci_max,
        10,
    )

    positive_errors = np.concatenate(
        [
            error_direct[error_direct > 0.0],
            error_gep[error_gep > 0.0],
        ]
    )

    if positive_errors.size == 0:
        error_min = 1.0e-8
        error_max = 1.0e-2
    else:
        error_min = max(
            float(np.nanpercentile(positive_errors, 1)),
            1.0e-8,
        )
        error_max = max(
            float(np.nanmax(positive_errors)),
            10.0 * error_min,
        )

    error_levels = np.geomspace(
        error_min,
        error_max,
        80,
    )

    error_norm = LogNorm(
        vmin=error_min,
        vmax=error_max,
    )

    direct_for_plot = np.maximum(
        error_direct,
        error_min,
    )

    gep_for_plot = np.maximum(
        error_gep,
        error_min,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 10.5),
        constrained_layout=True,
    )

    # Blumen vs direct PINN
    axis = axes[0, 0]

    axis.tricontour(
        triangulation,
        ci_blumen,
        levels=contour_levels,
        colors="tab:blue",
        linewidths=2.0,
    )

    axis.tricontour(
        triangulation,
        ci_direct,
        levels=contour_levels,
        colors="tab:orange",
        linewidths=1.7,
        linestyles="--",
    )

    axis.plot(
        [],
        [],
        color="tab:blue",
        linewidth=2.0,
        label="Blumen reference",
    )

    axis.plot(
        [],
        [],
        color="tab:orange",
        linewidth=1.7,
        linestyle="--",
        label="Direct PINN",
    )

    add_neutral_boundary(axis)
    configure_axis(axis)

    axis.set_title(
        "Blumen reference vs direct PINN"
    )

    axis.legend(
        loc="best",
        framealpha=0.95,
    )

    # Direct PINN error
    axis = axes[0, 1]

    direct_map = axis.tricontourf(
        triangulation,
        direct_for_plot,
        levels=error_levels,
        norm=error_norm,
        cmap="magma",
        extend="max",
    )

    axis.tricontour(
        triangulation,
        ci_blumen,
        levels=contour_levels,
        colors="white",
        linewidths=0.6,
        alpha=0.65,
    )

    add_neutral_boundary(axis)
    configure_axis(axis)

    axis.set_title(
        r"Direct PINN error "
        r"$|c_i^{\mathrm{PINN}}-c_i^{\mathrm{Blumen}}|$"
    )

    colorbar = figure.colorbar(
        direct_map,
        ax=axis,
    )

    colorbar.set_label(
        r"Absolute error in $c_i$"
    )

    # Blumen vs PINN-seeded GEP
    axis = axes[1, 0]

    axis.tricontour(
        triangulation,
        ci_blumen,
        levels=contour_levels,
        colors="tab:blue",
        linewidths=2.0,
    )

    axis.tricontour(
        triangulation,
        ci_gep,
        levels=contour_levels,
        colors="black",
        linewidths=1.7,
        linestyles="--",
    )

    axis.plot(
        [],
        [],
        color="tab:blue",
        linewidth=2.0,
        label="Blumen reference",
    )

    axis.plot(
        [],
        [],
        color="black",
        linewidth=1.7,
        linestyle="--",
        label="PINN-seeded GEP",
    )

    add_neutral_boundary(axis)
    configure_axis(axis)

    axis.set_title(
        "Blumen reference vs PINN-seeded GEP"
    )

    axis.legend(
        loc="best",
        framealpha=0.95,
    )

    # PINN-seeded GEP error
    axis = axes[1, 1]

    gep_map = axis.tricontourf(
        triangulation,
        gep_for_plot,
        levels=error_levels,
        norm=error_norm,
        cmap="magma",
        extend="max",
    )

    axis.tricontour(
        triangulation,
        ci_blumen,
        levels=contour_levels,
        colors="white",
        linewidths=0.6,
        alpha=0.65,
    )

    add_neutral_boundary(axis)
    configure_axis(axis)

    axis.set_title(
        r"PINN-seeded GEP error "
        r"$|c_i^{\mathrm{GEP}}-c_i^{\mathrm{Blumen}}|$"
    )

    colorbar = figure.colorbar(
        gep_map,
        ax=axis,
    )

    colorbar.set_label(
        r"Absolute error in $c_i$"
    )

    figure.suptitle(
        "Reconstruction of the Blumen subsonic isolines",
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
    print(f"Nombre de points : {len(data)}")
    print(
        "Direct PINN MAE vs Blumen : "
        f"{np.mean(error_direct):.6e}"
    )
    print(
        "PINN-seeded GEP MAE vs Blumen : "
        f"{np.mean(error_gep):.6e}"
    )


if __name__ == "__main__":
    main()
