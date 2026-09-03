#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"CSV introuvable : {input_path}"
        )

    data = pd.read_csv(input_path)

    required = {
        "Mach",
        "alpha",
        "blumen_ci",
        "classical_ci",
        "status",
    }

    missing_columns = sorted(required.difference(data.columns))

    if missing_columns:
        raise KeyError(
            "Colonnes manquantes : "
            + ", ".join(missing_columns)
        )

    for column in (
        "Mach",
        "alpha",
        "blumen_ci",
        "classical_ci",
        "classical_cr",
        "residual_norm",
    ):
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    # Isolignes positives uniquement.
    positive = data.loc[
        data["blumen_ci"] > 0.0
    ].copy()

    positive["has_root"] = (
        positive["status"].astype(str).eq("accepted_root")
        & np.isfinite(positive["classical_ci"])
    )

    accepted = positive.loc[
        positive["has_root"]
    ].copy()

    failed = positive.loc[
        ~positive["has_root"]
    ].copy()

    accepted["delta_ci"] = (
        accepted["classical_ci"]
        - accepted["blumen_ci"]
    )

    # Échelle de couleurs commune à Blumen et au classique.
    all_ci = np.concatenate(
        [
            accepted["blumen_ci"].to_numpy(float),
            accepted["classical_ci"].to_numpy(float),
        ]
    )

    norm = Normalize(
        vmin=float(np.nanmin(all_ci)),
        vmax=float(np.nanmax(all_ci)),
    )

    cmap = plt.get_cmap("viridis")

    figure, axis = plt.subplots(
        figsize=(9.2, 7.0),
        constrained_layout=True,
    )

    # -----------------------------------------------------------------
    # Blumen : grand anneau creux
    # -----------------------------------------------------------------

    blumen_edge_colors = cmap(
        norm(accepted["blumen_ci"].to_numpy(float))
    )

    axis.scatter(
        accepted["Mach"],
        accepted["alpha"],
        s=105,
        marker="o",
        facecolors="none",
        edgecolors=blumen_edge_colors,
        linewidths=2.4,
        zorder=2,
    )

    # -----------------------------------------------------------------
    # Classique : petit point plein exactement au même endroit
    # -----------------------------------------------------------------

    classical_scatter = axis.scatter(
        accepted["Mach"],
        accepted["alpha"],
        c=accepted["classical_ci"],
        cmap=cmap,
        norm=norm,
        s=34,
        marker="o",
        edgecolors="black",
        linewidths=0.30,
        zorder=3,
    )

    # Points sans racine classique acceptée.
    if not failed.empty:
        axis.scatter(
            failed["Mach"],
            failed["alpha"],
            s=90,
            marker="x",
            color="black",
            linewidths=1.8,
            zorder=4,
        )

    colorbar = figure.colorbar(
        classical_scatter,
        ax=axis,
        pad=0.025,
    )

    colorbar.set_label(
        r"Growth component $c_i$"
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=2.2,
            markersize=10,
            label=r"Blumen $c_i^B$ — outer ring",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="gray",
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=6,
            label=(
                r"Classical $c_i(M_B,\alpha_B)$"
                " — inner point"
            ),
        ),
    ]

    if not failed.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="none",
                color="black",
                markeredgewidth=1.8,
                markersize=8,
                label="No accepted classical root",
            )
        )

    axis.legend(
        handles=legend_handles,
        frameon=False,
        loc="best",
        fontsize=9,
    )

    axis.set_xlabel(r"Mach number $M$")
    axis.set_ylabel(r"Wavenumber $\alpha$")
    axis.set_title(
        "Blumen and classical growth rates at identical coordinates\n"
        "outer ring: Blumen; inner point: classical"
    )
    axis.grid(True, alpha=0.20)

    output_base = (
        output_dir
        / "Fig_supersonic_blumen_classical_ci_overlay"
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

    print("=== BLUMEN / CLASSICAL CI OVERLAY ===")
    print("Positive points :", len(positive))
    print("Accepted roots  :", len(accepted))
    print("Missing roots   :", len(failed))
    print()
    print("Mean delta ci   :", accepted["delta_ci"].mean())
    print("Mean |delta ci| :", accepted["delta_ci"].abs().mean())
    print("Max |delta ci|  :", accepted["delta_ci"].abs().max())
    print()
    print("PDF :", pdf_path)
    print("PNG :", png_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
