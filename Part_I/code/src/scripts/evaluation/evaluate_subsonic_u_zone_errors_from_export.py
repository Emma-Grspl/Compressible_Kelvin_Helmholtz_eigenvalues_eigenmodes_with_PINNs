#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path("assets/pinn_subsonic/u_reconstruction_diagnostics_M050")
EXPORT_ROOT = ROOT / "fields_export"
OUT = ROOT
OUT.mkdir(parents=True, exist_ok=True)

def cplx(df, re_col, im_col):
    return df[re_col].to_numpy(float) + 1j * df[im_col].to_numpy(float)

def rel_l2(y, pred, ref, mask):
    if int(mask.sum()) < 3:
        return np.nan
    yy = y[mask]
    num = np.trapz(np.abs(pred[mask] - ref[mask]) ** 2, yy)
    den = np.trapz(np.abs(ref[mask]) ** 2, yy)
    return float(np.sqrt(num / max(den, 1e-300)))

def complex_align(pred, ref, mask):
    if int(mask.sum()) < 3:
        return 1.0 + 0j
    den = np.vdot(pred[mask], pred[mask])
    if abs(den) < 1e-300:
        return 1.0 + 0j
    return np.vdot(pred[mask], ref[mask]) / den

zones = {
    "core_abs_y_le_0p5": lambda y: np.abs(y) <= 0.5,
    "core_abs_y_le_1": lambda y: np.abs(y) <= 1.0,
    "core_abs_y_le_2": lambda y: np.abs(y) <= 2.0,
    "inner_2_lt_abs_y_le_5": lambda y: (np.abs(y) > 2.0) & (np.abs(y) <= 5.0),
    "outer_abs_y_gt_5": lambda y: np.abs(y) > 5.0,
    "diagnostic_abs_y_le_15": lambda y: np.abs(y) <= 15.0,
    "full": lambda y: np.isfinite(y),
}

rows = []

files = sorted(EXPORT_ROOT.glob("*/fields_vs_classic_*.csv"))
if not files:
    raise SystemExit(f"[FAIL] no fields_vs_classic_*.csv found under {EXPORT_ROOT}")

for f in files:
    method = f.parent.name
    df = pd.read_csv(f)

    y = df["y"].to_numpy(float)
    alpha = float(df["alpha"].iloc[0])
    Mach = float(df["Mach"].iloc[0])

    u_ref = cplx(df, "u_ref_real", "u_ref_imag")
    u_pred_raw = cplx(df, "u_pred_real", "u_pred_imag")

    align_mask = np.abs(y) <= 15.0
    scale = complex_align(u_pred_raw, u_ref, align_mask)
    u_pred = scale * u_pred_raw

    row = {
        "method": method,
        "alpha": alpha,
        "Mach": Mach,
        "csv": str(f),
        "n": len(y),
        "align_scale_abs": abs(scale),
        "max_abs_u_ref": float(np.max(np.abs(u_ref))),
        "max_abs_u_pred_raw": float(np.max(np.abs(u_pred_raw))),
        "max_abs_u_pred_aligned": float(np.max(np.abs(u_pred))),
    }

    for name, fun in zones.items():
        mask = fun(y)
        row[f"u_rel_{name}"] = rel_l2(y, u_pred, u_ref, mask)

    rows.append(row)

metrics = pd.DataFrame(rows).sort_values(["alpha", "method"])
metrics.to_csv(OUT / "u_zone_error_metrics.csv", index=False)

print("[OK] wrote", OUT / "u_zone_error_metrics.csv")
print(metrics.to_string(index=False))

plot_cols = [
    "u_rel_core_abs_y_le_0p5",
    "u_rel_core_abs_y_le_1",
    "u_rel_core_abs_y_le_2",
    "u_rel_inner_2_lt_abs_y_le_5",
    "u_rel_outer_abs_y_gt_5",
    "u_rel_diagnostic_abs_y_le_15",
]

with PdfPages(OUT / "M050_u_zone_error_report.pdf") as pdf:
    for col in plot_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, g in metrics.groupby("method"):
            g = g.sort_values("alpha")
            ax.plot(g["alpha"], g[col], marker="o", label=method)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(col)
        ax.set_title(f"M=0.5 — {col}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(OUT / f"{col}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

print("[OK] wrote", OUT / "M050_u_zone_error_report.pdf")
