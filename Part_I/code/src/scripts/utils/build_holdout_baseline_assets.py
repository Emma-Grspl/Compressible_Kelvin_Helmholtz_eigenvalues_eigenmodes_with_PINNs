#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pick_column(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        for lc, orig in lower.items():
            if cand in lc:
                return orig
    return None


def summarize_errors(err):
    err = np.asarray(err, dtype=float)
    return {
        "n": int(err.size),
        "median": float(np.median(err)),
        "mean": float(np.mean(err)),
        "p95": float(np.quantile(err, 0.95)),
        "p99": float(np.quantile(err, 0.99)),
        "max": float(np.max(err)),
        "fraction_lt_1pct": float(np.mean(err < 1e-2)),
        "fraction_gt_5pct": float(np.mean(err > 5e-2)),
        "fraction_gt_10pct": float(np.mean(err > 1e-1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    out_fig = args.outdir / "figures"
    out_tab = args.outdir / "tables"
    out_src = args.outdir / "source_data"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)
    out_src.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    cols = list(df.columns)

    chart_col = pick_column(cols, ["chart_id", "chart", "chart_evaluated"])
    ref_col = pick_column(cols, ["ci_reference", "ci_ref", "classical_ci", "ci_classic"])

    required_error_cols = [
        "scaled_error_physics",
        "scaled_error_anchor_only",
    ]

    missing = [
        c for c in required_error_cols
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing scaled-error columns {missing}. "
            f"Available columns = {list(df.columns)}"
        )

    physics_err_col = "scaled_error_physics"
    anchor_err_col = "scaled_error_anchor_only"

    if chart_col is None:
        chart_col = "chart_id"
        df[chart_col] = "UNKNOWN"

    df = df.copy()
    df["physics_better"] = df[physics_err_col] < df[anchor_err_col]
    df["error_ratio_physics_over_anchor"] = df[physics_err_col] / np.maximum(df[anchor_err_col], 1e-12)
    df["error_delta_physics_minus_anchor"] = df[physics_err_col] - df[anchor_err_col]

    # global summary
    global_rows = []
    for method, col in [
        ("anchors_only", anchor_err_col),
        ("physics_informed", physics_err_col),
    ]:
        row = {"method": method}
        row.update(summarize_errors(df[col].to_numpy()))
        global_rows.append(row)

    global_summary = pd.DataFrame(global_rows)

    # chart summary
    chart_rows = []
    for chart, sub in df.groupby(chart_col):
        row = {
            "chart_id": chart,
            "n": len(sub),
            "anchors_only_median": float(np.median(sub[anchor_err_col])),
            "anchors_only_p95": float(np.quantile(sub[anchor_err_col], 0.95)),
            "anchors_only_max": float(np.max(sub[anchor_err_col])),
            "physics_median": float(np.median(sub[physics_err_col])),
            "physics_p95": float(np.quantile(sub[physics_err_col], 0.95)),
            "physics_max": float(np.max(sub[physics_err_col])),
            "physics_better_fraction": float(np.mean(sub["physics_better"])),
            "median_delta_physics_minus_anchor": float(np.median(sub["error_delta_physics_minus_anchor"])),
        }
        chart_rows.append(row)

    chart_summary = (
        pd.DataFrame(chart_rows)
        .sort_values(["physics_median", "anchors_only_median"], ascending=False)
        .reset_index(drop=True)
    )

    # worst points
    worst = (
        df.sort_values("error_delta_physics_minus_anchor", ascending=False)
        .head(40)
        .copy()
    )

    # save source data
    df.to_csv(out_src / "Holdout384_pointwise_physics_vs_anchors_only.csv", index=False)
    global_summary.to_csv(out_tab / "SuppTable_holdout_baselines_N340.csv", index=False)
    chart_summary.to_csv(out_tab / "SuppTable_holdout_by_chart_N340.csv", index=False)
    worst.to_csv(out_tab / "SuppTable_holdout_worst_physics_minus_anchor_N340.csv", index=False)

    # figure
    x = np.maximum(df[anchor_err_col].to_numpy(float), 1e-6)
    y = np.maximum(df[physics_err_col].to_numpy(float), 1e-6)

    mask_etaedge = df[chart_col].astype(str).eq("ETAEDGE_HM2B")
    mask_other = ~mask_etaedge

    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)

    ax.scatter(
        x[mask_other],
        y[mask_other],
        s=18,
        alpha=0.7,
        label="Other charts",
    )
    ax.scatter(
        x[mask_etaedge],
        y[mask_etaedge],
        s=26,
        marker="^",
        alpha=0.85,
        label="ETAEDGE_HM2B",
    )

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    grid = np.logspace(np.log10(lo), np.log10(hi), 200)
    ax.plot(grid, grid, linestyle="--", linewidth=1.2, label=r"$y=x$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Anchors-only scaled error in $c_i$")
    ax.set_ylabel(r"Physics-informed scaled error in $c_i$")
    ax.set_title("Strict holdout comparison on 384 off-anchor points")

    frac = float(np.mean(df["physics_better"]))
    txt = (
        f"n = {len(df)}\n"
        f"Physics better: {100*frac:.1f}%\n"
        f"Median anchor: {np.median(df[anchor_err_col]):.3e}\n"
        f"Median physics: {np.median(df[physics_err_col]):.3e}"
    )
    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    ax.legend()
    ax.grid(True, which="both", alpha=0.25)

    png = out_fig / "SuppFig_holdout_anchors_only_vs_physics_N340.png"
    pdf = out_fig / "SuppFig_holdout_anchors_only_vs_physics_N340.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {out_tab / 'SuppTable_holdout_baselines_N340.csv'}")
    print(f"Saved: {out_tab / 'SuppTable_holdout_by_chart_N340.csv'}")
    print(f"Saved: {out_tab / 'SuppTable_holdout_worst_physics_minus_anchor_N340.csv'}")


if __name__ == "__main__":
    main()
