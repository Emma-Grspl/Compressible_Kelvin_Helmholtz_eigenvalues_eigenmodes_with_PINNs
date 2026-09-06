#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Mach" in df.columns and "mach" not in df.columns:
        df = df.rename(columns={"Mach": "mach"})
    if "M" in df.columns and "mach" not in df.columns:
        df = df.rename(columns={"M": "mach"})
    return df


def nearest_row(df: pd.DataFrame, alpha: float, mach: float, tol: float = 1e-9) -> pd.Series:
    d = df.copy()
    d["_dist"] = (d["alpha"].astype(float) - alpha).abs() + (d["mach"].astype(float) - mach).abs()
    row = d.sort_values("_dist").iloc[0]
    if float(row["_dist"]) > tol:
        raise RuntimeError(f"No matching row for alpha={alpha}, mach={mach}; nearest dist={row['_dist']}")
    return row.drop(labels=["_dist"])


def score(row: pd.Series, cols: list[str]) -> float:
    vals = []
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            vals.append(float(row[c]))
    return float(np.mean(vals)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--pq-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mach", type=float, default=0.5)
    ap.add_argument("--eval-alphas", default="0.3 0.5 0.7")
    ap.add_argument("--alpha-switch", type=float, default=0.4)
    ap.add_argument(
        "--score-cols",
        default="p_rel p_y_rel u_rel v_rel",
        help="Columns used for aggregate comparison.",
    )
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline = norm_cols(pd.read_csv(args.baseline_csv))
    pq = norm_cols(pd.read_csv(args.pq_csv))

    alphas = [float(x) for x in args.eval_alphas.replace(",", " ").split()]
    score_cols = [x for x in args.score_cols.replace(",", " ").split() if x.strip()]

    all_rows = []
    hybrid_rows = []
    best_rows = []

    for alpha in alphas:
        b = nearest_row(baseline, alpha, args.mach)
        p = nearest_row(pq, alpha, args.mach)

        b = b.copy()
        p = p.copy()
        b["source_run"] = "baseline_pressure_only"
        p["source_run"] = "pq_bootstrapped"

        b["aggregate_score"] = score(b, score_cols)
        p["aggregate_score"] = score(p, score_cols)

        all_rows.extend([b, p])

        # Fixed, article-friendly rule.
        if alpha <= args.alpha_switch:
            h = b.copy()
        else:
            h = p.copy()
        h["source_run"] = "baseline_pressure_only" if alpha <= args.alpha_switch else "pq_bootstrapped"
        h["hybrid_rule"] = f"baseline if alpha <= {args.alpha_switch:g}, else pq_bootstrapped"
        h["aggregate_score"] = score(h, score_cols)
        hybrid_rows.append(h)

        # Oracle only for diagnostic sanity; not the article rule.
        best = b.copy() if b["aggregate_score"] <= p["aggregate_score"] else p.copy()
        best["source_run"] = "best_by_validation_score"
        best["hybrid_rule"] = "oracle diagnostic only"
        best["aggregate_score"] = score(best, score_cols)
        best_rows.append(best)

    comp = pd.DataFrame(all_rows)
    hybrid = pd.DataFrame(hybrid_rows)
    oracle = pd.DataFrame(best_rows)

    comp.to_csv(outdir / "comparison_baseline_vs_pq.csv", index=False)
    hybrid.to_csv(outdir / "diagnostics_summary.csv", index=False)
    oracle.to_csv(outdir / "diagnostics_summary_oracle.csv", index=False)

    combined = pd.concat(
        [
            comp.assign(run=comp["source_run"]),
            hybrid.assign(run="hybrid_fixed_switch"),
            oracle.assign(run="hybrid_oracle_validation_only"),
        ],
        ignore_index=True,
        sort=False,
    )

    keep_cols = [
        "run",
        "source_run",
        "hybrid_rule",
        "alpha",
        "mach",
        "ci_abs_err",
        "p_rel",
        "p_y_rel",
        "rho_rel",
        "u_rel",
        "v_rel",
        "gamma_rel",
        "aggregate_score",
    ]
    keep_cols = [c for c in keep_cols if c in combined.columns]

    table = combined[keep_cols].copy()
    table.to_csv(outdir / "summary_table.csv", index=False)

    agg = (
        table.groupby("run", dropna=False)[
            [c for c in ["p_rel", "p_y_rel", "u_rel", "v_rel", "gamma_rel", "aggregate_score"] if c in table.columns]
        ]
        .mean()
        .reset_index()
    )
    agg.to_csv(outdir / "aggregate_summary.csv", index=False)

    with open(outdir / "README.md", "w") as f:
        f.write("# Hybrid subsonic pressure-only / first-order p-q diagnostics\n\n")
        f.write(f"- baseline_csv: {args.baseline_csv}\n")
        f.write(f"- pq_csv: {args.pq_csv}\n")
        f.write(f"- mach: {args.mach}\n")
        f.write(f"- eval_alphas: {alphas}\n")
        f.write(f"- fixed rule: baseline if alpha <= {args.alpha_switch:g}, else pq_bootstrapped\n")
        f.write(f"- score_cols: {score_cols}\n\n")
        f.write("Important: this hybrid selects a complete modal package from one expert per alpha. It does not mix p from one model with q from another.\n\n")
        f.write("## Fixed-switch hybrid\n\n")
        f.write(hybrid[keep_cols[2:] if keep_cols[0] == "run" else keep_cols].to_string(index=False))
        f.write("\n\n## Aggregate summary\n\n")
        f.write(agg.to_string(index=False))
        f.write("\n")

    print("\n[OK] wrote", outdir)
    print("\nSummary table:")
    print(table.sort_values(["alpha", "run"]).to_string(index=False))
    print("\nAggregate summary:")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
