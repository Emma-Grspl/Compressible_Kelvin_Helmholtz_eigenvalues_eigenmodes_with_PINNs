#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"CSV source introuvable : {input_path}"
        )

    data = pd.read_csv(input_path)

    required = {
        "Mach",
        "alpha",
        "blumen_ci",
        "classical_cr",
        "classical_ci",
        "status",
    }

    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(
            "Colonnes manquantes : " + ", ".join(missing)
        )

    for column in (
        "Mach",
        "alpha",
        "blumen_ci",
        "classical_cr",
        "classical_ci",
        "residual_norm",
    ):
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    # Isolignes strictement positives uniquement.
    positive = data.loc[data["blumen_ci"] > 0.0].copy()

    positive["has_root"] = (
        positive["status"].astype(str).eq("accepted_root")
        & np.isfinite(positive["classical_cr"])
        & np.isfinite(positive["classical_ci"])
    )

    positive["delta_ci"] = (
        positive["classical_ci"]
        - positive["blumen_ci"]
    )

    levels = sorted(
        float(value)
        for value in positive["blumen_ci"].dropna().unique()
    )

    cmap = plt.get_cmap("viridis")
    colors = {
        level: cmap(
            0.08
            + 0.84 * index / max(1, len(levels) - 1)
        )
        for index, level in enumerate(levels)
    }

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17.5, 5.2),
        constrained_layout=True,
    )

    ax_geometry, ax_values, ax_error = axes

    # ================================================================
    # (a) Coordonnées digitalisées de Blumen
    # ================================================================

    for level in levels:
        group = positive.loc[
            np.isclose(
                positive["blumen_ci"],
                level,
                atol=1.0e-12,
            )
        ]

        accepted = group.loc[group["has_root"]]
        failed = group.loc[~group["has_root"]]

        # Aucun plot : uniquement scatter.
        ax_geometry.scatter(
            accepted["Mach"],
            accepted["alpha"],
            s=42,
            marker="o",
            color=colors[level],
            edgecolors="black",
            linewidths=0.35,
            alpha=0.90,
            label=rf"$c_i^B={level:g}$",
        )

        if not failed.empty:
            ax_geometry.scatter(
                failed["Mach"],
                failed["alpha"],
                s=75,
                marker="x",
                color="black",
                linewidths=1.7,
                zorder=5,
            )

    ax_geometry.set_xlabel(r"Mach number $M$")
    ax_geometry.set_ylabel(r"Wavenumber $\alpha$")
    ax_geometry.set_title(
        "(a) Digitized positive Blumen isolines\n"
        "markers only"
    )
    ax_geometry.grid(True, alpha=0.20)
    ax_geometry.legend(
        frameon=False,
        fontsize=9,
        ncols=2,
    )

    # ================================================================
    # (b) ci classique aux coordonnées Blumen
    # ================================================================

    for level in levels:
        group = positive.loc[
            np.isclose(
                positive["blumen_ci"],
                level,
                atol=1.0e-12,
            )
            & positive["has_root"]
        ]

        ax_values.scatter(
            group["Mach"],
            group["classical_ci"],
            s=42,
            marker="o",
            color=colors[level],
            edgecolors="black",
            linewidths=0.35,
            alpha=0.90,
            label=rf"$c_i^B={level:g}$",
        )

        # Niveau nominal de Blumen : ligne de référence seulement.
        if not group.empty:
            ax_values.hlines(
                y=level,
                xmin=float(group["Mach"].min()),
                xmax=float(group["Mach"].max()),
                color=colors[level],
                linestyle="--",
                linewidth=1.1,
                alpha=0.55,
            )

    ax_values.set_xlabel(r"Mach number $M$")
    ax_values.set_ylabel(
        r"Classical phase-velocity growth rate $c_i$"
    )
    ax_values.set_title(
        "(b) Classical values at Blumen coordinates\n"
        "dashed lines: nominal Blumen levels"
    )
    ax_values.grid(True, alpha=0.20)
    ax_values.legend(
        frameon=False,
        fontsize=9,
        ncols=2,
    )

    # ================================================================
    # (c) Erreur pointwise
    # ================================================================

    for level in levels:
        group = positive.loc[
            np.isclose(
                positive["blumen_ci"],
                level,
                atol=1.0e-12,
            )
            & positive["has_root"]
        ]

        ax_error.scatter(
            group["Mach"],
            group["delta_ci"],
            s=42,
            marker="o",
            color=colors[level],
            edgecolors="black",
            linewidths=0.35,
            alpha=0.90,
            label=rf"$c_i^B={level:g}$",
        )

    ax_error.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.1,
    )

    ax_error.set_xlabel(r"Mach number $M$")
    ax_error.set_ylabel(
        r"$\Delta c_i="
        r"c_i^{\mathrm{classical}}-c_i^B$"
    )
    ax_error.set_title(
        "(c) Pointwise spectral error\n"
        "markers only"
    )
    ax_error.grid(True, alpha=0.20)
    ax_error.legend(
        frameon=False,
        fontsize=9,
        ncols=2,
    )

    output_base = (
        output_dir
        / "Fig_supersonic_blumen_positive_isolines_scatter"
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

    print("=== BLUMEN POSITIVE ISOLINES — SCATTER ONLY ===")
    print("Positive points :", len(positive))
    print(
        "Accepted roots :",
        int(positive["has_root"].sum()),
    )
    print(
        "Missing roots  :",
        int((~positive["has_root"]).sum()),
    )
    print("PDF :", pdf_path)
    print("PNG :", png_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
