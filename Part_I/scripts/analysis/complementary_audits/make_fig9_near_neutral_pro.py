#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=str,
        default="assets/validation_pointwise_canonical.csv",
        help="Path to validation_pointwise_canonical.csv",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="assets/section4_results",
        help="Output directory",
    )
    p.add_argument(
        "--eta-min",
        type=float,
        default=0.86,
        help="Minimum eta kept in the near-neutral cuts",
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=4,
        help="Minimum number of points required for a Mach cut",
    )
    return p.parse_args()


def choose_mach_cuts(df, eta_min=0.86, min_points=4):
    near = df[df["eta"] >= eta_min].copy()

    rows = []
    for M, g in near.groupby("Mach"):
        if len(g) < min_points:
            continue
        scale = np.maximum(np.abs(g["ci_ref"].to_numpy(dtype=float)), 0.05)
        gep_err = np.abs(g["scalar_gep_ci"].to_numpy(dtype=float) - g["ci_ref"].to_numpy(dtype=float)) / scale
        rows.append(
            {
                "Mach": float(M),
                "n_points": int(len(g)),
                "max_gep_scaled_error": float(np.max(gep_err)),
            }
        )

    stats = pd.DataFrame(rows).sort_values(["Mach"])
    if stats.empty:
        raise RuntimeError("No valid Mach cut found with the current filters.")

    # Prefer high-Mach cuts, because this is the sensitive region.
    high = stats[stats["Mach"] >= 0.90].copy()
    if len(high) >= 3:
        chosen = high.sort_values("Mach").tail(3)["Mach"].tolist()
    elif len(stats) >= 3:
        chosen = stats.sort_values("Mach").tail(3)["Mach"].tolist()
    else:
        chosen = stats.sort_values("Mach")["Mach"].tolist()

    return chosen, stats


def main():
    args = parse_args()

    input_csv = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    png_out = outdir / "Fig_near_neutral_growth_rate_cuts_N340.png"
    pdf_out = outdir / "Fig_near_neutral_growth_rate_cuts_N340.pdf"
    csv_out = outdir / "Fig_near_neutral_growth_rate_cuts_N340_data.csv"

    df = pd.read_csv(input_csv).copy()

    required = ["Mach", "eta", "alpha", "ci_ref", "ci_pinn", "scalar_gep_ci"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    chosen_machs, stats = choose_mach_cuts(
        df,
        eta_min=args.eta_min,
        min_points=args.min_points,
    )

    plotted_rows = []
    n_panels = len(chosen_machs)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 140,
        "savefig.dpi": 300,
    })

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(5.2 * n_panels, 4.6),
        sharey=True,
        constrained_layout=False,
    )

    if n_panels == 1:
        axes = [axes]

    # Global y-range based on all selected data
    ymins = []
    ymaxs = []

    for M in chosen_machs:
        g = df[np.isclose(df["Mach"], M, atol=1e-12) & (df["eta"] >= args.eta_min)].copy()
        g = g.sort_values("eta")
        plotted_rows.append(g.assign(selected_M=M))
        ymins.extend([g["ci_ref"].min(), g["ci_pinn"].min(), g["scalar_gep_ci"].min()])
        ymaxs.extend([g["ci_ref"].max(), g["ci_pinn"].max(), g["scalar_gep_ci"].max()])

    y_min = min(ymins)
    y_max = max(ymaxs)
    y_pad = 0.06 * (y_max - y_min if y_max > y_min else 1.0)
    y_low = max(0.0, y_min - y_pad)
    y_high = y_max + y_pad

    for ax, M in zip(axes, chosen_machs):
        g = df[np.isclose(df["Mach"], M, atol=1e-12) & (df["eta"] >= args.eta_min)].copy()
        g = g.sort_values("eta")

        scale = np.maximum(np.abs(g["ci_ref"].to_numpy(dtype=float)), 0.05)
        gep_err = np.abs(g["scalar_gep_ci"].to_numpy(dtype=float) - g["ci_ref"].to_numpy(dtype=float)) / scale
        max_gep_scaled_error = float(np.max(gep_err))

        ax.plot(
            g["eta"], g["ci_ref"],
            marker="o", linewidth=2.2, markersize=6,
            label="Classical",
        )
        ax.plot(
            g["eta"], g["ci_pinn"],
            marker="s", linewidth=2.0, markersize=5.5,
            label="Direct PINN",
        )
        ax.plot(
            g["eta"], g["scalar_gep_ci"],
            marker="^", linewidth=2.0, markersize=6,
            label="Selected GEP",
        )

        # Visual marker for the more delicate near-neutral zone
        ax.axvline(0.92, linestyle="--", linewidth=1.2, alpha=0.7)

        ax.set_title(
            rf"$M={M:.2f}$",
            fontsize=13,
            pad=8,
        )

        ax.text(
            0.04,
            0.95,
            f"max scaled GEP error = {max_gep_scaled_error:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="0.75",
                alpha=0.90,
            ),
        )
        ax.set_xlabel(r"$\eta$")
        ax.set_xlim(g["eta"].min() - 0.003, g["eta"].max() + 0.003)
        ax.set_ylim(y_low, y_high)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"Growth rate $c_i$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=2.2,
        handlelength=2.4,
    )

    # Leave a dedicated strip above the panels for the common legend.
    # No global title: the manuscript caption already provides it.
    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.16,
        top=0.82,
        wspace=0.08,
    )

    fig.savefig(png_out, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)

    out_df = pd.concat(plotted_rows, ignore_index=True)
    out_df.to_csv(csv_out, index=False)

    print("=" * 90)
    print("FIGURE 9 — PROFESSIONAL VERSION")
    print("=" * 90)
    print("Input:", input_csv)
    print("Output PNG:", png_out)
    print("Output PDF:", pdf_out)
    print("Output CSV:", csv_out)
    print()
    print("Chosen Mach cuts:", [f"{m:.6f}" for m in chosen_machs])
    print()
    print("Available near-neutral Mach statistics:")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
