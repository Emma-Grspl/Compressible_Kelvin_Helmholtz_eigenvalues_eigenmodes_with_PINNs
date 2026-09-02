#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, PowerNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator


CI_LEVELS = np.array([0.01, 0.03, 0.05, 0.07, 0.10], dtype=float)

ACCEPTED_STATUSES = {
    "accepted_root",
    "accepted",
    "converged",
    "success",
}


def numeric(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    out = frame.copy()

    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(
                out[column],
                errors="coerce",
            )

    return out


def load_reference(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    required = {"Mach", "alpha", "ci"}
    missing = sorted(required.difference(frame.columns))

    if missing:
        raise KeyError(
            "Colonnes absentes de la référence : "
            + ", ".join(missing)
        )

    frame = numeric(
        frame,
        (
            "Mach",
            "alpha",
            "ci",
            "cr",
            "residual_norm",
        ),
    )

    frame = frame.dropna(
        subset=["Mach", "alpha", "ci"]
    ).copy()

    if "accepted" in frame.columns:
        accepted = (
            frame["accepted"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )

        frame = frame.loc[accepted].copy()

    return frame


def load_blumen(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    required = {
        "Mach",
        "alpha",
        "blumen_ci",
        "classical_ci",
        "status",
    }

    missing = sorted(required.difference(frame.columns))

    if missing:
        raise KeyError(
            "Colonnes absentes du CSV Blumen : "
            + ", ".join(missing)
        )

    frame = numeric(
        frame,
        (
            "Mach",
            "alpha",
            "blumen_ci",
            "classical_ci",
            "classical_cr",
            "residual_norm",
        ),
    )

    # Isolignes strictement positives.
    frame = frame.loc[
        np.isfinite(frame["blumen_ci"])
        & (frame["blumen_ci"] > 0.0)
        & np.isfinite(frame["Mach"])
        & np.isfinite(frame["alpha"])
    ].copy()

    frame["has_root"] = (
        frame["status"]
        .astype(str)
        .str.strip()
        .isin(ACCEPTED_STATUSES)
        & np.isfinite(frame["classical_ci"])
    )

    frame["delta_ci"] = (
        frame["classical_ci"]
        - frame["blumen_ci"]
    )

    frame["abs_delta_ci"] = frame["delta_ci"].abs()

    return frame


def build_classical_samples(
    reference: pd.DataFrame,
    blumen: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct a scattered sampling of the same classical function ci(M,alpha).

    The canonical reference is preferred whenever a coordinate appears in
    both datasets. Pointwise Blumen evaluations extend the sampling outside
    the canonical M/alpha envelope.
    """
    canonical = reference[
        ["Mach", "alpha", "ci"]
    ].copy()

    canonical = canonical.rename(
        columns={"ci": "classical_ci"}
    )

    canonical["source"] = "canonical_reference"
    canonical["priority"] = 2

    pointwise = blumen.loc[
        blumen["has_root"],
        ["Mach", "alpha", "classical_ci"],
    ].copy()

    pointwise["source"] = "blumen_pointwise"
    pointwise["priority"] = 1

    samples = pd.concat(
        [pointwise, canonical],
        ignore_index=True,
        sort=False,
    )

    samples = samples.dropna(
        subset=["Mach", "alpha", "classical_ci"]
    ).copy()

    # Déduplication numérique des coordonnées :
    # la référence canonique est prioritaire.
    samples["Mach_key"] = samples["Mach"].round(10)
    samples["alpha_key"] = samples["alpha"].round(10)

    samples = (
        samples.sort_values("priority")
        .drop_duplicates(
            ["Mach_key", "alpha_key"],
            keep="last",
        )
        .drop(columns=["Mach_key", "alpha_key"])
        .reset_index(drop=True)
    )

    return samples


def build_interpolated_field(
    samples: pd.DataFrame,
    blumen: pd.DataFrame,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    # Le domaine visible est déterminé par les points Blumen positifs.
    x_min = float(blumen["Mach"].min())
    x_max = float(blumen["Mach"].max())
    y_min = max(0.0, float(blumen["alpha"].min()))
    y_max = float(blumen["alpha"].max())

    x_margin = 0.015 * (x_max - x_min)
    y_margin = 0.025 * max(y_max - y_min, 1.0e-3)

    x = np.linspace(
        x_min - x_margin,
        x_max + x_margin,
        nx,
    )

    y = np.linspace(
        max(0.0, y_min - y_margin),
        y_max + y_margin,
        ny,
    )

    grid_M, grid_alpha = np.meshgrid(x, y)

    interpolator = LinearNDInterpolator(
        list(
            zip(
                samples["Mach"].to_numpy(float),
                samples["alpha"].to_numpy(float),
                strict=True,
            )
        ),
        samples["classical_ci"].to_numpy(float),
        fill_value=np.nan,
        rescale=True,
    )

    grid_ci = interpolator(grid_M, grid_alpha)

    # Les NaN sont hors de l’enveloppe convexe des calculs disponibles.
    grid_ci = np.ma.masked_invalid(grid_ci)

    return grid_M, grid_alpha, grid_ci


def contour_segment_counts(contours) -> dict[float, int]:
    return {
        float(level): sum(
            len(segment) >= 2
            for segment in level_segments
        )
        for level, level_segments in zip(
            contours.levels,
            contours.allsegs,
            strict=True,
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--blumen",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=720,
    )

    parser.add_argument(
        "--ny",
        type=int,
        default=620,
    )

    args = parser.parse_args()

    reference_path = args.reference.expanduser().resolve()
    blumen_path = args.blumen.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_reference(reference_path)
    blumen = load_blumen(blumen_path)

    accepted = blumen.loc[blumen["has_root"]].copy()
    failed = blumen.loc[~blumen["has_root"]].copy()

    samples = build_classical_samples(reference, blumen)

    grid_M, grid_alpha, grid_ci = build_interpolated_field(
        samples,
        blumen,
        nx=args.nx,
        ny=args.ny,
    )

    # Cinq couleurs discrètes cohérentes entre points et isolignes.
    base_cmap = plt.get_cmap("viridis")

    level_colors = [
        base_cmap(value)
        for value in np.linspace(0.08, 0.92, len(CI_LEVELS))
    ]

    discrete_cmap = ListedColormap(level_colors)

    boundaries = np.array(
        [0.0, 0.02, 0.04, 0.06, 0.085, 0.115],
        dtype=float,
    )

    discrete_norm = BoundaryNorm(
        boundaries,
        discrete_cmap.N,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16.2, 6.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    ax_overlay, ax_error = axes

    # ================================================================
    # (a) Isolignes classiques + points Blumen
    # ================================================================

    contours = ax_overlay.contour(
        grid_M,
        grid_alpha,
        grid_ci,
        levels=CI_LEVELS,
        colors=level_colors,
        linewidths=2.2,
        zorder=2,
    )

    contour_labels = {
        level: rf"$c_i={level:g}$"
        for level in CI_LEVELS
    }

    ax_overlay.clabel(
        contours,
        contours.levels,
        fmt=contour_labels,
        inline=True,
        fontsize=8,
    )

    # Points digitalisés Blumen :
    # même couleur que leur niveau nominal, marqueur ouvert.
    for index, level in enumerate(CI_LEVELS):
        subset = blumen.loc[
            np.isclose(
                blumen["blumen_ci"],
                level,
                atol=1.0e-12,
            )
        ]

        ax_overlay.scatter(
            subset["Mach"],
            subset["alpha"],
            s=31,
            marker="o",
            facecolors="white",
            edgecolors=[level_colors[index]],
            linewidths=1.25,
            zorder=3,
        )

    if not failed.empty:
        ax_overlay.scatter(
            failed["Mach"],
            failed["alpha"],
            marker="x",
            s=58,
            color="black",
            linewidths=1.5,
            zorder=4,
        )

    style_legend = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.2,
            label="Classical isolines",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.3,
            markersize=6,
            label="Digitized Blumen points",
        ),
    ]

    if not failed.empty:
        style_legend.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="x",
                color="black",
                markeredgewidth=1.5,
                markersize=7,
                label="No accepted classical root",
            )
        )

    ax_overlay.legend(
        handles=style_legend,
        loc="best",
        frameon=True,
        fontsize=9,
    )

    ax_overlay.set_xlabel(r"Mach number $M$")
    ax_overlay.set_ylabel(r"Wavenumber $\alpha$")
    ax_overlay.set_title(
        "(a) Blumen digitization and reconstructed classical isolines"
    )
    ax_overlay.grid(True, alpha=0.20)

    # ================================================================
    # (b) Heatmap pointwise de l'erreur
    # ================================================================

    max_error = float(accepted["abs_delta_ci"].max())

    error_norm = PowerNorm(
        gamma=0.55,
        vmin=0.0,
        vmax=max_error,
    )

    error_scatter = ax_error.scatter(
        accepted["Mach"],
        accepted["alpha"],
        c=accepted["abs_delta_ci"],
        cmap="inferno",
        norm=error_norm,
        s=35,
        marker="o",
        edgecolors="none",
        zorder=2,
    )

    error_q90 = float(
        accepted["abs_delta_ci"].quantile(0.90)
    )

    largest_errors = accepted.loc[
        accepted["abs_delta_ci"] >= error_q90
    ]

    ax_error.scatter(
        largest_errors["Mach"],
        largest_errors["alpha"],
        s=68,
        marker="o",
        facecolors="none",
        edgecolors="cyan",
        linewidths=1.25,
        label="Top 10% errors",
        zorder=3,
    )

    if not failed.empty:
        ax_error.scatter(
            failed["Mach"],
            failed["alpha"],
            marker="x",
            s=58,
            color="black",
            linewidths=1.5,
            label="No accepted classical root",
            zorder=4,
        )

    colorbar = figure.colorbar(
        error_scatter,
        ax=ax_error,
        pad=0.018,
    )

    colorbar.set_label(
        r"$|c_i^{\mathrm{classical}}(M_B,\alpha_B)-c_i^B|$"
    )

    ax_error.set_xlabel(r"Mach number $M$")
    ax_error.set_title(
        "(b) Pointwise error at the Blumen coordinates"
    )
    ax_error.grid(True, alpha=0.20)
    ax_error.legend(
        loc="best",
        frameon=True,
        fontsize=9,
    )

    x_min = float(blumen["Mach"].min())
    x_max = float(blumen["Mach"].max())
    y_min = max(0.0, float(blumen["alpha"].min()))
    y_max = float(blumen["alpha"].max())

    x_margin = 0.025 * (x_max - x_min)
    y_margin = 0.035 * max(y_max - y_min, 1.0e-3)

    for axis in axes:
        axis.set_xlim(
            x_min - x_margin,
            x_max + x_margin,
        )
        axis.set_ylim(
            max(0.0, y_min - y_margin),
            y_max + y_margin,
        )

    output_base = (
        output_dir
        / "Fig_supersonic_blumen_classical_isolines_and_error"
    )

    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        png_path,
        dpi=350,
        bbox_inches="tight",
    )

    plt.close(figure)

    # Données utilisées pour reconstruire le champ classique.
    sample_path = (
        output_dir.parent
        / "source_data"
        / "supersonic_classical_contour_interpolation_samples.csv"
    )

    sample_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples.to_csv(sample_path, index=False)

    segment_counts = contour_segment_counts(contours)

    print("=== SUPERSONIC BLUMEN / CLASSICAL 1x2 ===")
    print("Canonical samples :", len(reference))
    print("Pointwise roots   :", len(accepted))
    print("Combined samples  :", len(samples))
    print("Missing roots     :", len(failed))
    print()
    print("Contour segments:")
    for level in CI_LEVELS:
        print(
            f"  ci={level:g}: "
            f"{segment_counts.get(float(level), 0)}"
        )
    print()
    print("Mean |delta ci|   :", accepted["abs_delta_ci"].mean())
    print("Max |delta ci|    :", accepted["abs_delta_ci"].max())
    print("Top-10% threshold :", error_q90)
    print()
    print("PDF               :", pdf_path)
    print("PNG               :", png_path)
    print("Interpolation data:", sample_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
