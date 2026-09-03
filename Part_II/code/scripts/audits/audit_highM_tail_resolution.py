#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


FIELD_SPECS = [
    ("p", "p_real", "p_imag"),
    ("rho", "rho_real", "rho_imag"),
    ("u", "u_real", "u_imag"),
    ("v", "v_real", "v_imag"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--points-csv", required=True)
    p.add_argument("--modal-parquet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-review", type=int, default=24)
    return p.parse_args()


def keyify(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Mach"] = out["Mach"].astype(float)
    out["alpha"] = out["alpha"].astype(float)
    out["_Mach_key"] = out["Mach"].round(12)
    out["_alpha_key"] = out["alpha"].round(12)
    return out


def normalized_complex(df: pd.DataFrame, rcol: str, icol: str):
    z = df[rcol].to_numpy(dtype=float) + 1j * df[icol].to_numpy(dtype=float)
    amp = np.abs(z)
    mx = np.nanmax(amp)
    if not np.isfinite(mx) or mx <= 0:
        return z, np.zeros_like(amp)
    return z / mx, amp / mx


def tail_samples_per_period(y: np.ndarray, z: np.ndarray, amp: np.ndarray, side: str):
    if side == "left":
        mask = (y < 0) & (amp >= 1e-3) & (amp <= 2e-2)
    else:
        mask = (y > 0) & (amp >= 1e-3) & (amp <= 2e-2)

    ys = y[mask]
    zs = z[mask]

    if len(ys) < 8:
        return np.nan, 0

    phase = np.unwrap(np.angle(zs))
    dphi = np.diff(phase)

    dphi = np.abs(dphi)
    dphi = dphi[np.isfinite(dphi) & (dphi > 1e-10)]

    if len(dphi) == 0:
        return np.nan, len(ys)

    spp = 2 * np.pi / np.median(dphi)
    return float(spp), int(len(ys))


def build_review_page(group: pd.DataFrame, title: str):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    for ax, (label, rcol, icol) in zip(axes, FIELD_SPECS):
        g = group.sort_values("y")
        y = g["y"].to_numpy(dtype=float)
        z, amp = normalized_complex(g, rcol, icol)

        ax.plot(y, np.real(z), label="real", linewidth=1.0)
        ax.plot(y, np.imag(z), label="imag", linewidth=1.0)
        ax.plot(y, amp, "--", label="modulus", linewidth=1.0)

        ax.set_title(label)
        ax.set_xlabel("y")
        ax.set_ylabel("normalized mode")
        ax.grid(True, alpha=0.25)

        if label == "p":
            ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main():
    args = parse_args()

    if not args.points_csv.strip():
        raise SystemExit(
            "--points-csv is empty."
        )

    if not args.modal_parquet.strip():
        raise SystemExit(
            "--modal-parquet is empty. "
            "Pass the explicit Parquet file path."
        )

    points_csv = Path(
        args.points_csv
    ).expanduser()

    modal_parquet = Path(
        args.modal_parquet
    ).expanduser()

    outdir = Path(
        args.output_dir
    ).expanduser()

    if not points_csv.is_file():
        raise SystemExit(
            f"Points CSV not found: {points_csv}"
        )

    if not modal_parquet.is_file():
        raise SystemExit(
            "Modal Parquet is not a regular file: "
            f"{modal_parquet}"
        )

    outdir.mkdir(parents=True, exist_ok=True)

    points = keyify(pd.read_csv(points_csv))
    modal = keyify(pd.read_parquet(modal_parquet))

    # garde uniquement les points demandés
    keys = set(zip(points["_Mach_key"], points["_alpha_key"]))
    modal = modal[
        modal[["_Mach_key", "_alpha_key"]]
        .apply(tuple, axis=1)
        .isin(keys)
    ].copy()

    metrics = []

    grouped = modal.groupby(["_Mach_key", "_alpha_key"], sort=True)

    for (mk, ak), group in grouped:
        group = group.sort_values("y").copy()
        group["y"] = pd.to_numeric(group["y"], errors="coerce")
        group = group[np.isfinite(group["y"])].copy()

        row = {
            "Mach": float(mk),
            "alpha": float(ak),
            "n_rows": len(group),
            "y_min": float(group["y"].min()),
            "y_max": float(group["y"].max()),
        }

        worst_tail = np.inf

        for label, rcol, icol in FIELD_SPECS:
            z, amp = normalized_complex(group, rcol, icol)
            y = group["y"].to_numpy(dtype=float)

            left_spp, left_n = tail_samples_per_period(y, z, amp, "left")
            right_spp, right_n = tail_samples_per_period(y, z, amp, "right")

            row[f"{label}_left_tail_samples_per_period"] = left_spp
            row[f"{label}_right_tail_samples_per_period"] = right_spp
            row[f"{label}_left_tail_n"] = left_n
            row[f"{label}_right_tail_n"] = right_n

            vals = [v for v in [left_spp, right_spp] if np.isfinite(v)]
            if vals:
                worst_tail = min(worst_tail, min(vals))

        row["worst_tail_samples_per_period"] = (
            float(worst_tail) if np.isfinite(worst_tail) else np.nan
        )
        row["tail_resolution_flag"] = (
            "needs_attention"
            if (np.isfinite(row["worst_tail_samples_per_period"]) and row["worst_tail_samples_per_period"] < 8.0)
            else "ok"
        )

        metrics.append(row)

    metrics = pd.DataFrame(metrics).sort_values(
        ["tail_resolution_flag", "worst_tail_samples_per_period", "Mach", "alpha"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    metrics_csv = outdir / "highM_tail_resolution_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)

    worst = metrics.sort_values(
        ["worst_tail_samples_per_period", "Mach", "alpha"],
        na_position="last"
    ).reset_index(drop=True)

    worst_csv = outdir / "highM_tail_resolution_worst_points.csv"
    worst.to_csv(worst_csv, index=False)

    review_pdf = outdir / "highM_tail_resolution_review.pdf"
    review_points = worst.head(args.max_review)

    with PdfPages(review_pdf) as pdf:
        for _, r in review_points.iterrows():
            mk = round(float(r["Mach"]), 12)
            ak = round(float(r["alpha"]), 12)

            group = grouped.get_group((mk, ak)).copy()
            title = (
                f"M={r['Mach']:.2f}, alpha={r['alpha']:.5f} | "
                f"worst tail spp={r['worst_tail_samples_per_period']:.2f}"
            )
            fig = build_review_page(group, title)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    summary = {
        "n_points": int(len(metrics)),
        "n_flagged": int((metrics["tail_resolution_flag"] == "needs_attention").sum()),
        "metrics_csv": str(metrics_csv),
        "worst_csv": str(worst_csv),
        "review_pdf": str(review_pdf),
    }

    summary_path = outdir / "summary.json"
    pd.Series(summary).to_json(summary_path, indent=2)

    print("Wrote:", metrics_csv)
    print("Wrote:", worst_csv)
    print("Wrote:", review_pdf)
    print("Wrote:", summary_path)
    print()
    print(metrics[["Mach", "alpha", "worst_tail_samples_per_period", "tail_resolution_flag"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
