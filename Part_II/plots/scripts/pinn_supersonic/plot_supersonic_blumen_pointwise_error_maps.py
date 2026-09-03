#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_missing_roots(
    axis: plt.Axes,
    missing: pd.DataFrame,
) -> None:
    """Display digitized coordinates without an accepted classical root."""
    if missing.empty:
        return

    axis.scatter(
        missing["Mach"],
        missing["alpha"],
        marker="x",
        s=85,
        linewidths=1.8,
        color="black",
        label="No accepted classical root",
        zorder=5,
    )


def add_top_error_outlines(
    axis: plt.Axes,
    frame: pd.DataFrame,
    error_column: str,
) -> float:
    """Outline the largest 10% pointwise errors."""
    threshold = float(frame[error_column].quantile(0.90))

    largest = frame.loc[
        frame[error_column] >= threshold
    ]

    axis.scatter(
        largest["Mach"],
        largest["alpha"],
        s=73,
        marker="o",
        facecolors="none",
        edgecolors="cyan",
        linewidths=1.7,
        label="Top 10% errors",
        zorder=4,
    )

    return threshold


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

    data = pd.read_csv(input_path)

    required = {
        "Mach",
        "alpha",
        "blumen_ci",
        "classical_cr",
        "classical_ci",
        "status",
    }

    missing_columns = sorted(required.difference(data.columns))

    if missing_columns:
        raise KeyError(
            "Colonnes absentes : "
            + ", ".join(missing_columns)
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

    # Exclusion complète de la courbe neutre.
    positive = data.loc[
        data["blumen_ci"] > 0.0
    ].copy()

    positive["has_root"] = (
        positive["status"].astype(str).eq("accepted_root")
        & np.isfinite(positive["classical_cr"])
        & np.isfinite(positive["classical_ci"])
    )

    accepted = positive.loc[
        positive["has_root"]
    ].copy()

    missing = positive.loc[
        ~positive["has_root"]
    ].copy()

    # Valeurs pointwise aux coordonnées exactes de Blumen.
    accepted["delta_ci_signed"] = (
        accepted["classical_ci"]
        - accepted["blumen_ci"]
    )

    accepted["abs_delta_ci"] = (
        accepted["delta_ci_signed"].abs()
    )

    accepted["blumen_omega_i"] = (
        accepted["alpha"]
        * accepted["blumen_ci"]
    )

    accepted["classical_omega_i"] = (
        accepted["alpha"]
        * accepted["classical_ci"]
    )

    accepted["delta_omega_i_signed"] = (
        accepted["classical_omega_i"]
        - accepted["blumen_omega_i"]
    )

    accepted["abs_delta_omega_i"] = (
        accepted["delta_omega_i_signed"].abs()
    )

    # ================================================================
    # Figure 1 : ci classique réellement recalculé aux points Blumen
    # ================================================================

    figure_ci, axis_ci = plt.subplots(
        figsize=(8.2, 6.6),
        constrained_layout=True,
    )

    ci_scatter = axis_ci.scatter(
        accepted["Mach"],
        accepted["alpha"],
        c=accepted["classical_ci"],
        cmap="viridis",
        s=47,
        marker="o",
        edgecolors="black",
        linewidths=0.30,
    )

    add_missing_roots(axis_ci, missing)

    colorbar_ci = figure_ci.colorbar(
        ci_scatter,
        ax=axis_ci,
        pad=0.025,
    )

    colorbar_ci.set_label(
        r"$c_i^{\mathrm{classical}}(M_B,\alpha_B)$"
    )

    axis_ci.set_xlabel(r"Mach number $M$")
    axis_ci.set_ylabel(r"Wavenumber $\alpha$")
    axis_ci.set_title(
        "Classical growth component evaluated\n"
        "at the digitized Blumen coordinates"
    )
    axis_ci.grid(True, alpha=0.20)

    if not missing.empty:
        axis_ci.legend(
            frameon=False,
            loc="best",
        )

    ci_pdf = (
        output_dir
        / "Fig_supersonic_blumen_classical_ci_at_digitized_points.pdf"
    )
    ci_png = (
        output_dir
        / "Fig_supersonic_blumen_classical_ci_at_digitized_points.png"
    )

    figure_ci.savefig(ci_pdf, bbox_inches="tight")
    figure_ci.savefig(ci_png, dpi=350, bbox_inches="tight")
    plt.close(figure_ci)

    # ================================================================
    # Figure 2 : erreurs pointwise, analogue à la figure subsonique
    # ================================================================

    figure_error, axes = plt.subplots(
        1,
        2,
        figsize=(14.8, 6.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    axis_ci_error, axis_omega_error = axes

    # ---------------------------------------------------------------
    # (a) Erreur absolue sur ci
    # ---------------------------------------------------------------

    scatter_ci_error = axis_ci_error.scatter(
        accepted["Mach"],
        accepted["alpha"],
        c=accepted["abs_delta_ci"],
        cmap="inferno",
        s=44,
        marker="o",
        edgecolors="none",
    )

    ci_threshold = add_top_error_outlines(
        axis_ci_error,
        accepted,
        "abs_delta_ci",
    )

    add_missing_roots(
        axis_ci_error,
        missing,
    )

    colorbar_ci_error = figure_error.colorbar(
        scatter_ci_error,
        ax=axis_ci_error,
        pad=0.025,
    )

    colorbar_ci_error.set_label(
        r"$|c_i^{\mathrm{classical}}-c_i^B|$"
    )

    axis_ci_error.set_xlabel(r"Mach number $M$")
    axis_ci_error.set_ylabel(r"Wavenumber $\alpha$")
    axis_ci_error.set_title(
        r"(a) Error on $c_i$ at Blumen points"
    )
    axis_ci_error.grid(True, alpha=0.20)
    axis_ci_error.legend(
        frameon=False,
        loc="best",
        fontsize=9,
    )

    # ---------------------------------------------------------------
    # (b) Erreur absolue sur omega_i = alpha ci
    # ---------------------------------------------------------------

    scatter_omega_error = axis_omega_error.scatter(
        accepted["Mach"],
        accepted["alpha"],
        c=accepted["abs_delta_omega_i"],
        cmap="inferno",
        s=44,
        marker="o",
        edgecolors="none",
    )

    omega_threshold = add_top_error_outlines(
        axis_omega_error,
        accepted,
        "abs_delta_omega_i",
    )

    add_missing_roots(
        axis_omega_error,
        missing,
    )

    colorbar_omega_error = figure_error.colorbar(
        scatter_omega_error,
        ax=axis_omega_error,
        pad=0.025,
    )

    colorbar_omega_error.set_label(
        r"$|\omega_i^{\mathrm{classical}}-\omega_i^B|$"
    )

    axis_omega_error.set_xlabel(r"Mach number $M$")
    axis_omega_error.set_title(
        r"(b) Error on $\omega_i=\alpha c_i$ at Blumen points"
    )
    axis_omega_error.grid(True, alpha=0.20)
    axis_omega_error.legend(
        frameon=False,
        loc="best",
        fontsize=9,
    )

    error_pdf = (
        output_dir
        / "Fig_supersonic_blumen_pointwise_error_maps.pdf"
    )
    error_png = (
        output_dir
        / "Fig_supersonic_blumen_pointwise_error_maps.png"
    )

    figure_error.savefig(
        error_pdf,
        bbox_inches="tight",
    )
    figure_error.savefig(
        error_png,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(figure_error)

    # ================================================================
    # CSV complet servant à la figure
    # ================================================================

    output_table = (
        output_dir.parent
        / "tables"
        / "Tab_supersonic_blumen_pointwise_errors.csv"
    )

    output_table.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    accepted.to_csv(
        output_table,
        index=False,
    )

    print(
        "=== SUPERSONIC BLUMEN POINTWISE MAPS ==="
    )
    print(
        "Positive digitized points :",
        len(positive),
    )
    print(
        "Accepted classical roots  :",
        len(accepted),
    )
    print(
        "Missing classical roots   :",
        len(missing),
    )
    print()
    print(
        "Mean |delta ci|           :",
        accepted["abs_delta_ci"].mean(),
    )
    print(
        "Maximum |delta ci|        :",
        accepted["abs_delta_ci"].max(),
    )
    print(
        "Top-10% ci threshold      :",
        ci_threshold,
    )
    print()
    print(
        "Mean |delta omega_i|      :",
        accepted["abs_delta_omega_i"].mean(),
    )
    print(
        "Maximum |delta omega_i|   :",
        accepted["abs_delta_omega_i"].max(),
    )
    print(
        "Top-10% omega threshold   :",
        omega_threshold,
    )
    print()
    print("Classical ci map :", ci_pdf)
    print("Error map        :", error_pdf)
    print("Source table     :", output_table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
