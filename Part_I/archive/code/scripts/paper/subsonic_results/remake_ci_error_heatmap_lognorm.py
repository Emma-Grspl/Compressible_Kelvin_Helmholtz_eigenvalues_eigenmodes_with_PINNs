#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT
    / "assets/pinn_subsonic/article/results_pinn/release_final"
    / "data/validation_pointwise_canonical.csv"
)

OUTPUT = (
    ROOT
    / "assets/pinn_subsonic/article/results_pinn/release_final"
    / "figures/Fig_ci_error_heatmap_PINN_GEP_vs_classical.png"
)


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    list[str],
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
            f"Aucune colonne trouvée parmi {candidates}\n"
            f"Colonnes disponibles : {list(frame.columns)}"
        )

    return None


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(
            f"CSV introuvable : {INPUT}"
        )

    raw = pd.read_csv(INPUT)

    mach_column = find_column(
        raw,
        ["Mach", "mach", "M"],
    )

    alpha_column = find_column(
        raw,
        ["alpha", "Alpha"],
    )

    error_column = find_column(
        raw,
        [
            "ci_final_abs_err",
            "ci_final_abs_error",
            "ci_gep_abs_err",
            "gep_ci_abs_err_classic",
            "ci_final_error_abs",
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
        }
    )

    if error_column is not None:
        data["absolute_error"] = pd.to_numeric(
            raw[error_column],
            errors="coerce",
        ).abs()

    else:
        final_column = find_column(
            raw,
            [
                "ci_final",
                "ci_gep",
                "gep_ci",
                "pinn_matched_ci",
            ],
        )

        reference_column = find_column(
            raw,
            [
                "ci_classic",
                "ci_classical",
                "ci_reference",
                "ci_ref",
                "ci_shooting",
            ],
        )

        data["absolute_error"] = np.abs(
            pd.to_numeric(
                raw[final_column],
                errors="coerce",
            )
            -
            pd.to_numeric(
                raw[reference_column],
                errors="coerce",
            )
        )

    data = (
        data
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .groupby(
            ["Mach", "alpha"],
            as_index=False,
        )
        .mean(numeric_only=True)
    )

    if len(data) < 3:
        raise RuntimeError(
            f"Seulement {len(data)} points valides."
        )

    mach = data["Mach"].to_numpy(dtype=float)
    alpha = data["alpha"].to_numpy(dtype=float)
    error = data["absolute_error"].to_numpy(dtype=float)

    positive = error[
        np.isfinite(error) & (error > 0.0)
    ]

    if positive.size == 0:
        raise RuntimeError(
            "Toutes les erreurs sont nulles ou non finies."
        )

    # Échelle commune lisible pour l'article.
    vmin = 1.0e-6
    vmax = max(
        float(np.nanmax(positive)),
        1.0e-5,
    )

    plotted_error = np.maximum(
        error,
        vmin,
    )

    triangulation = mtri.Triangulation(
        mach,
        alpha,
    )

    triangles = triangulation.triangles

    triangle_mach = mach[triangles].mean(axis=1)
    triangle_alpha = alpha[triangles].mean(axis=1)

    outside_domain = (
        triangle_mach**2
        + triangle_alpha**2
        > 1.0005
    )

    flat_triangles = (
        mtri.TriAnalyzer(triangulation)
        .get_flat_tri_mask(
            min_circle_ratio=0.005
        )
    )

    triangulation.set_mask(
        outside_domain | flat_triangles
    )

    levels = np.geomspace(
        vmin,
        vmax,
        160,
    )

    figure, axis = plt.subplots(
        figsize=(9.2, 7.6),
    )

    heatmap = axis.tricontourf(
        triangulation,
        plotted_error,
        levels=levels,
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
        cmap="viridis",
        extend="max",
    )

    mach_boundary = np.linspace(
        0.0,
        0.9999,
        1000,
    )

    alpha_boundary = np.sqrt(
        np.maximum(
            0.0,
            1.0 - mach_boundary**2,
        )
    )

    axis.plot(
        mach_boundary,
        alpha_boundary,
        linestyle="--",
        linewidth=1.3,
        color="tab:blue",
        label=r"$\alpha^2+M^2=1$",
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)

    axis.set_xlabel(r"Mach number $M$")
    axis.set_ylabel(r"Wavenumber $\alpha$")

    axis.set_title(
        "Absolute spectral error: "
        "PINN-seeded GEP versus classical shooting"
    )

    axis.legend(
        loc="upper right",
        framealpha=0.95,
    )

    colorbar = figure.colorbar(
        heatmap,
        ax=axis,
        pad=0.055,
    )

    colorbar.set_label(
        r"$\left|c_i^{\mathrm{PINN+GEP}}"
        r"-c_i^{\mathrm{shoot}}\right|$"
    )

    colorbar.locator = LogLocator(
        base=10.0,
    )

    colorbar.formatter = LogFormatterMathtext(
        base=10.0,
    )

    colorbar.update_ticks()

    figure.tight_layout()

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
    print(f"Erreur minimale positive : {positive.min():.6e}")
    print(f"Erreur médiane : {np.median(error):.6e}")
    print(f"Erreur maximale : {positive.max():.6e}")


if __name__ == "__main__":
    main()
