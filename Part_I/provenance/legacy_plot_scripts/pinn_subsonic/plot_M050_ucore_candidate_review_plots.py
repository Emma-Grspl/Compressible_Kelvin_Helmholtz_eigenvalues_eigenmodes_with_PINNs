#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path("assets/pinn_subsonic/u_reconstruction_diagnostics_M050")
EXPORT = ROOT / "fields_export"
OUT = ROOT / "review_plots"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    "mini2d_pq_discrete",
    "mini2d_ucore_smooth005",
    "mini2d_ucore_smooth02",
]

LABELS = {
    "mini2d_pq_discrete": "baseline",
    "mini2d_ucore_smooth005": "u-core 0.05",
    "mini2d_ucore_smooth02": "u-core 0.20",
}

ALPHAS = [0.3, 0.5, 0.7]

def cplx(df, name):
    return df[f"{name}_real"].to_numpy(float) + 1j * df[f"{name}_imag"].to_numpy(float)

def load(method, alpha):
    tag = f"M0500_a{int(round(alpha * 1000)):04d}"
    path = EXPORT / method / f"fields_vs_classic_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path), path

def rel_l2(y, pred, ref):
    num = np.trapz(np.abs(pred - ref) ** 2, y)
    den = np.trapz(np.abs(ref) ** 2, y)
    return float(np.sqrt(num / max(den, 1e-300)))

pdf_path = OUT / "M050_ucore_candidate_review.pdf"

with PdfPages(pdf_path) as pdf:
    # Page 1: global/zoned u metrics table-like plot
    metrics_path = ROOT / "u_zone_error_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        sub = metrics[metrics["method"].isin(METHODS)].copy()
        cols = [
            "u_rel_full",
            "u_rel_core_abs_y_le_2",
            "u_rel_inner_2_lt_abs_y_le_5",
            "u_rel_outer_abs_y_gt_5",
        ]

        for col in cols:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for method in METHODS:
                g = sub[sub["method"] == method].sort_values("alpha")
                if len(g):
                    ax.plot(g["alpha"], g[col], marker="o", label=LABELS[method])
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(col)
            ax.set_title(f"M=0.5 — {col}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(OUT / f"{col}_comparison.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

    # Profile pages: u, v, p, q for each alpha.
    variables = [
        ("p", "pressure p"),
        ("q", "derivative q=p_y"),
        ("u", "velocity u"),
        ("v", "velocity v"),
        ("gamma", "gamma"),
    ]

    for alpha in ALPHAS:
        for var, title in variables:
            fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

            ref_done = False
            for method in METHODS:
                df, _ = load(method, alpha)
                y = df["y"].to_numpy(float)

                ref = cplx(df, f"{var}_ref")
                pred = cplx(df, f"{var}_pred")

                if not ref_done:
                    axes[0].plot(y, ref.real, linestyle="--", linewidth=2, label="classic ref")
                    axes[1].plot(y, ref.imag, linestyle="--", linewidth=2, label="classic ref")
                    ref_done = True

                err = rel_l2(y, pred, ref)
                lab = f"{LABELS[method]}  rel={err:.3g}"

                axes[0].plot(y, pred.real, label=lab)
                axes[1].plot(y, pred.imag, label=lab)

            axes[0].set_title(f"M=0.5, alpha={alpha:g} — Re({title})")
            axes[1].set_title(f"M=0.5, alpha={alpha:g} — Im({title})")
            axes[1].set_xlabel("y")
            axes[0].set_ylabel("real")
            axes[1].set_ylabel("imag")

            for ax in axes:
                ax.set_xlim(-15, 15)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(OUT / f"M050_a{int(round(alpha*1000)):04d}_{var}_profiles.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

print("[OK] wrote", pdf_path)
