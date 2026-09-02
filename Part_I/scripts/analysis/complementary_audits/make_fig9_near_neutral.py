#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_input(explicit: str | None) -> Path:
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p

    cwd = Path.cwd()

    candidates = [
        cwd / "assets/pinn_subsonic/anchor_budget_runs/N340/summary/validation_pointwise_canonical.csv",
        cwd / "assets/validation_pointwise_canonical.csv",
        cwd / "assets/section4_results/validation_pointwise_canonical.csv",
    ]

    for p in candidates:
        if p.exists():
            return p.resolve()

    matches = sorted(cwd.glob("assets/**/validation_pointwise_canonical.csv"))
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        "Could not find validation_pointwise_canonical.csv.\n"
        "Please pass it explicitly with --input."
    )


def scaled_error(pred: pd.Series, ref: pd.Series) -> pd.Series:
    denom = np.maximum(np.abs(ref.to_numpy(dtype=float)), 0.05)
    return np.abs(pred.to_numpy(dtype=float) - ref.to_numpy(dtype=float)) / denom


def choose_actual_machs(all_machs: np.ndarray, targets: list[float]) -> list[float]:
    chosen = []
    for t in targets:
        m = float(all_machs[np.argmin(np.abs(all_machs - t))])
        if m not in chosen:
            chosen.append(m)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to validation_pointwise_canonical.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/section4_results",
        help="Output directory",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=0.86,
        help="Minimum eta kept in the near-neutral cuts",
    )
    args = parser.parse_args()

    input_csv = find_input(args.input)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    png_out = outdir / "Fig_near_neutral_growth_rate_cuts_N340.png"
    pdf_out = outdir / "Fig_near_neutral_growth_rate_cuts_N340.pdf"
    csv_out = outdir / "Fig_near_neutral_growth_rate_cuts_N340_data.csv"

    df = pd.read_csv(input_csv).copy()

    required = [
        "Mach",
        "eta",
        "alpha",
        "ci_ref",
        "ci_pinn",
        "scalar_gep_ci",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required).copy()

    unique_machs = np.array(sorted(df["Mach"].unique()), dtype=float)
    target_machs = choose_actual_machs(unique_machs, [0.90, 0.95, 0.98])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)

    plotted_rows = []
    global_ymax = 0.0

    for ax, M in zip(axes, target_machs):
        g = df[np.isclose(df["Mach"], M, atol=1e-12)].copy()
        g = g[g["eta"] >= args.eta_min].copy()
        g = g.sort_values("eta").reset_index(drop=True)

        if len(g) == 0:
            ax.text(0.5, 0.5, f"No data for M={M:.2f}", ha="center", va="center")
            ax.set_axis_off()
            continue

        g["err_pinn_scaled"] = scaled_error(g["ci_pinn"], g["ci_ref"])
        g["err_gep_scaled"] = scaled_error(g["scalar_gep_ci"], g["ci_ref"])

        global_ymax = max(
            global_ymax,
            float(g["ci_ref"].max()),
            float(g["ci_pinn"].max()),
            float(g["scalar_gep_ci"].max()),
        )

        ax.plot(g["eta"], g["ci_ref"], "o-", label="Classical")
        ax.plot(g["eta"], g["ci_pinn"], "s-", label="Direct PINN")
        ax.plot(g["eta"], g["scalar_gep_ci"], "^-", label="PINN-seeded GEP")

        worst_idx = g["err_gep_scaled"].idxmax()
        worst_eta = float(g.loc[worst_idx, "eta"])
        ax.axvline(worst_eta, linestyle="--", linewidth=1.0, alpha=0.5)

        ax.set_title(
            f"M = {M:.2f}\nmax scaled GEP error = {g['err_gep_scaled'].max():.3f}"
        )
        ax.set_xlabel(r"$\eta$")
        ax.grid(alpha=0.25)

        plotted_rows.append(g.assign(selected_M=M))

    axes[0].set_ylabel(r"$c_i$")

    if global_ymax > 0:
        for ax in axes:
            ax.set_ylim(bottom=0.0, top=1.05 * global_ymax)

    axes[-1].legend(loc="best")

    fig.suptitle(
        "Near-neutral growth-rate cuts for the N340 model\n"
        "High-Mach cuts selected to show the most sensitive region",
        y=1.03,
    )

    fig.tight_layout()
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)

    if plotted_rows:
        out_df = pd.concat(plotted_rows, ignore_index=True)
        out_df.to_csv(csv_out, index=False)
    else:
        pd.DataFrame().to_csv(csv_out, index=False)

    print("=" * 90)
    print("FIGURE 9 — NEAR-NEUTRAL HIGH-MACH CUTS")
    print("=" * 90)
    print("Input:", input_csv)
    print("Output PNG:", png_out)
    print("Output PDF:", pdf_out)
    print("Output CSV:", csv_out)
    print()
    print("Selected Mach cuts:", [f"{m:.6f}" for m in target_machs])
    print()

    if plotted_rows:
        summary_rows = []
        out_df = pd.concat(plotted_rows, ignore_index=True)
        for M, g in out_df.groupby("selected_M"):
            summary_rows.append({
                "Mach": M,
                "n_points": len(g),
                "eta_min": g["eta"].min(),
                "eta_max": g["eta"].max(),
                "max_pinn_scaled_error": g["err_pinn_scaled"].max(),
                "max_gep_scaled_error": g["err_gep_scaled"].max(),
                "median_gep_scaled_error": g["err_gep_scaled"].median(),
            })
        summary = pd.DataFrame(summary_rows).sort_values("Mach")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
