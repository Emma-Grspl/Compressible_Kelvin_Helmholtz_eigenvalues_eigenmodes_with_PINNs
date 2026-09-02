#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def contiguous_segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split an ordered Blumen curve into contiguous accepted-root segments.

    This prevents a plotted classical curve from bridging across a missing
    point, notably at the three alpha -> 0 endpoints.
    """
    segments: list[pd.DataFrame] = []
    current_indices: list[int] = []

    for index, accepted in zip(
        frame.index,
        frame["has_root"],
        strict=True,
    ):
        if bool(accepted):
            current_indices.append(index)
        elif current_indices:
            segments.append(frame.loc[current_indices].copy())
            current_indices = []

    if current_indices:
        segments.append(frame.loc[current_indices].copy())

    return segments


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)

    required = {
        "blumen_row_id",
        "curve_key",
        "Mach",
        "alpha",
        "blumen_ci",
        "status",
        "classical_cr",
        "classical_ci",
        "residual_norm",
    }

    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(
            "Colonnes absentes du CSV pointwise : "
            + ", ".join(missing)
        )

    for column in (
        "blumen_row_id",
        "source_row_id",
        "Mach",
        "alpha",
        "blumen_ci",
        "classical_cr",
        "classical_ci",
        "classical_omega_i",
        "delta_ci",
        "residual_norm",
    ):
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    # Courbes positives uniquement. La neutralité ci=0 est exclue.
    positive = data.loc[data["blumen_ci"] > 0.0].copy()

    positive["has_root"] = (
        positive["status"].astype(str).eq("accepted_root")
        & np.isfinite(positive["classical_cr"])
        & np.isfinite(positive["classical_ci"])
    )

    positive["delta_ci_recomputed"] = (
        positive["classical_ci"] - positive["blumen_ci"]
    )

    order_column = (
        "source_row_id"
        if "source_row_id" in positive.columns
        else "blumen_row_id"
    )

    positive = positive.sort_values(order_column).reset_index(drop=True)

    levels = sorted(
        float(value)
        for value in positive["blumen_ci"].dropna().unique()
    )

    cmap = plt.get_cmap("viridis")
    colors = {
        level: cmap(
            0.08
            + 0.84
            * index
            / max(1, len(levels) - 1)
        )
        for index, level in enumerate(levels)
    }

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, 4.8),
        constrained_layout=True,
    )

    ax_geometry, ax_values, ax_error = axes

    # ------------------------------------------------------------------
    # (a) Géométrie des isolignes Blumen positives
    # ------------------------------------------------------------------

    legend_used: set[float] = set()

    for curve_key, curve in positive.groupby(
        "curve_key",
        sort=False,
    ):
        curve = curve.sort_values(order_column)
        level = float(curve["blumen_ci"].iloc[0])
        color = colors[level]

        label = None
        if level not in legend_used:
            label = rf"$c_i^B={level:g}$"
            legend_used.add(level)

        # Ligne digitalisée de Blumen : tous les points, y compris ceux
        # où le solveur classique n'a pas trouvé de racine acceptée.
        ax_geometry.plot(
            curve["Mach"],
            curve["alpha"],
            linestyle="-",
            linewidth=1.55,
            color=color,
            alpha=0.82,
            label=label,
        )

        accepted = curve.loc[curve["has_root"]]
        failed = curve.loc[~curve["has_root"]]

        ax_geometry.scatter(
            accepted["Mach"],
            accepted["alpha"],
            s=21,
            marker="o",
            facecolors=color,
            edgecolors="black",
            linewidths=0.25,
            zorder=3,
        )

        if not failed.empty:
            ax_geometry.scatter(
                failed["Mach"],
                failed["alpha"],
                s=47,
                marker="x",
                color="black",
                linewidths=1.3,
                zorder=4,
            )

    ax_geometry.set_xlabel(r"Mach number $M$")
    ax_geometry.set_ylabel(r"Wavenumber $\alpha$")
    ax_geometry.set_title(
        "(a) Positive Blumen isolines\n"
        "circles: accepted classical roots"
    )
    ax_geometry.grid(True, alpha=0.22)
    ax_geometry.legend(
        frameon=False,
        fontsize=8,
        ncols=2,
    )

    # ------------------------------------------------------------------
    # (b) ci classique évalué sur chaque isoligne Blumen
    # ------------------------------------------------------------------

    target_label_used: set[float] = set()
    classical_label_used: set[float] = set()

    for curve_key, curve in positive.groupby(
        "curve_key",
        sort=False,
    ):
        curve = curve.sort_values(order_column)
        level = float(curve["blumen_ci"].iloc[0])
        color = colors[level]

        # Valeur constante annoncée par Blumen pour cette isoligne.
        ax_values.plot(
            curve["Mach"],
            np.full(len(curve), level),
            linestyle="--",
            linewidth=1.05,
            color=color,
            alpha=0.65,
            label=(
                rf"Blumen $c_i={level:g}$"
                if level not in target_label_used
                else None
            ),
        )
        target_label_used.add(level)

        # Valeurs classiques. On coupe la ligne lorsqu'une racine manque.
        for segment_index, segment in enumerate(
            contiguous_segments(curve)
        ):
            ax_values.plot(
                segment["Mach"],
                segment["classical_ci"],
                linestyle="-",
                marker="o",
                markersize=3.2,
                linewidth=1.55,
                color=color,
                label=(
                    rf"Classical on $c_i^B={level:g}$"
                    if (
                        level not in classical_label_used
                        and segment_index == 0
                    )
                    else None
                ),
            )

        classical_label_used.add(level)

    ax_values.set_xlabel(r"Mach number $M$")
    ax_values.set_ylabel(r"Phase-velocity growth rate $c_i$")
    ax_values.set_title(
        "(b) Classical values at the digitized coordinates\n"
        "dashed: nominal Blumen level"
    )
    ax_values.grid(True, alpha=0.22)

    # Une seule légende compacte par niveau : les styles sont expliqués
    # directement dans le titre.
    handles, labels = ax_values.get_legend_handles_labels()
    compact_handles = []
    compact_labels = []

    for level in levels:
        expected = rf"Classical on $c_i^B={level:g}$"
        if expected in labels:
            index = labels.index(expected)
            compact_handles.append(handles[index])
            compact_labels.append(rf"$c_i^B={level:g}$")

    ax_values.legend(
        compact_handles,
        compact_labels,
        frameon=False,
        fontsize=8,
        ncols=2,
    )

    # ------------------------------------------------------------------
    # (c) Erreur pointwise
    # ------------------------------------------------------------------

    for curve_key, curve in positive.groupby(
        "curve_key",
        sort=False,
    ):
        curve = curve.sort_values(order_column)
        level = float(curve["blumen_ci"].iloc[0])
        color = colors[level]

        for segment in contiguous_segments(curve):
            ax_error.plot(
                segment["Mach"],
                segment["delta_ci_recomputed"],
                linestyle="-",
                marker="o",
                markersize=3.2,
                linewidth=1.45,
                color=color,
            )

    ax_error.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
    )
    ax_error.set_xlabel(r"Mach number $M$")
    ax_error.set_ylabel(
        r"$\Delta c_i=c_i^{\mathrm{classical}}-c_i^B$"
    )
    ax_error.set_title("(c) Pointwise spectral error")
    ax_error.grid(True, alpha=0.22)

    pdf_path = (
        output_dir
        / "Fig_supersonic_blumen_positive_isolines_pointwise.pdf"
    )
    png_path = (
        output_dir
        / "Fig_supersonic_blumen_positive_isolines_pointwise.png"
    )

    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    # ------------------------------------------------------------------
    # Table quantitative par niveau
    # ------------------------------------------------------------------

    rows: list[dict[str, float | int]] = []

    for level, group in positive.groupby(
        "blumen_ci",
        sort=True,
    ):
        valid = group.loc[group["has_root"]].copy()
        error = valid["delta_ci_recomputed"]

        rows.append(
            {
                "blumen_ci": float(level),
                "n_digitized_points": int(len(group)),
                "n_accepted_roots": int(group["has_root"].sum()),
                "completion": float(group["has_root"].mean()),
                "classical_ci_mean": float(
                    valid["classical_ci"].mean()
                ),
                "classical_ci_std": float(
                    valid["classical_ci"].std(ddof=0)
                ),
                "mean_signed_error": float(error.mean()),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "max_absolute_error": float(error.abs().max()),
                "median_residual": float(
                    valid["residual_norm"].median()
                ),
                "max_residual": float(
                    valid["residual_norm"].max()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    summary_path = (
        output_dir.parent
        / "tables"
        / "Tab_supersonic_blumen_positive_isolines.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    failed = positive.loc[~positive["has_root"]].copy()
    failed_path = (
        output_dir.parent
        / "tables"
        / "Tab_supersonic_blumen_positive_missing_roots.csv"
    )
    failed.to_csv(failed_path, index=False)

    accepted_points_path = (
        output_dir.parent
        / "source_data"
        / "supersonic_blumen_positive_pointwise_values.csv"
    )
    accepted_points_path.parent.mkdir(parents=True, exist_ok=True)
    positive.to_csv(accepted_points_path, index=False)

    print("=== POSITIVE BLUMEN ISOLINES ===")
    print(f"Positive points : {len(positive)}")
    print(
        "Accepted roots : "
        f"{int(positive['has_root'].sum())}/{len(positive)}"
    )
    print(f"Missing roots  : {len(failed)}")
    print()
    print(summary.to_string(index=False))
    print()
    print("Figure PDF     :", pdf_path)
    print("Figure PNG     :", png_path)
    print("Summary table  :", summary_path)
    print("Missing points :", failed_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
