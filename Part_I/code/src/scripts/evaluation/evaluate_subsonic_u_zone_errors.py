#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RUNS = {
    "mini2d_pq_discrete": Path("assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_discrete"),
    "pq_detach_old": Path("assets/pinn_subsonic/permach_pq_bootp_M050_a010070_cilearned_nodetach"),
    "pq_true_nodetach": Path("assets/pinn_subsonic/permach_pq_bootp_M050_a010070_TRUE_cilearned_nodetach"),
}

OUT = Path("assets/pinn_subsonic/u_reconstruction_diagnostics_M050")
OUT.mkdir(parents=True, exist_ok=True)

def infer_alpha(path):
    m = re.search(r"a(\d{4})", path.name)
    if m:
        return int(m.group(1)) / 1000.0
    m = re.search(r"alpha[_=]?([0-9.]+)", path.name)
    if m:
        return float(m.group(1))
    return np.nan

def cplx(df, re_col, im_col):
    return df[re_col].to_numpy(float) + 1j * df[im_col].to_numpy(float)

def rel_l2(y, pred, ref, mask):
    if int(mask.sum()) < 3:
        return np.nan
    yy = y[mask]
    num = np.trapz(np.abs(pred[mask] - ref[mask])**2, yy)
    den = np.trapz(np.abs(ref[mask])**2, yy)
    return float(np.sqrt(num / max(den, 1e-300)))

def complex_align(pred, ref, mask):
    if int(mask.sum()) < 3:
        return 1.0 + 0j
    den = np.vdot(pred[mask], pred[mask])
    if abs(den) < 1e-300:
        return 1.0 + 0j
    return np.vdot(pred[mask], ref[mask]) / den

zones = {
    "core_|y|<=0.5": lambda y: np.abs(y) <= 0.5,
    "core_|y|<=1": lambda y: np.abs(y) <= 1.0,
    "core_|y|<=2": lambda y: np.abs(y) <= 2.0,
    "inner_2<|y|<=5": lambda y: (np.abs(y) > 2.0) & (np.abs(y) <= 5.0),
    "outer_|y|>5": lambda y: np.abs(y) > 5.0,
    "diagnostic_|y|<=15": lambda y: np.abs(y) <= 15.0,
    "full": lambda y: np.isfinite(y),
}

rows = []
field_files = []

for method, run in RUNS.items():
    for f in sorted(run.glob("fields_vs_classic_*.csv")):
        field_files.append((method, f))

if not field_files:
    raise SystemExit(
        "[FAIL] Aucun fields_vs_classic_*.csv trouvé. "
        "Il faut d'abord relancer/exporter les diagnostics numériques."
    )

for method, f in field_files:
    df = pd.read_csv(f)
    y = df["y"].to_numpy(float)

    alpha = float(df["alpha"].iloc[0]) if "alpha" in df.columns else infer_alpha(f)
    Mach = float(df["Mach"].iloc[0]) if "Mach" in df.columns else 0.5

    u_ref = cplx(df, "u_ref_real", "u_ref_imag")
    u_pred_raw = cplx(df, "u_pred_real", "u_pred_imag")

    # alignement global sur |y|<=15, cohérent avec les plots.
    align_mask = np.abs(y) <= 15
    scale = complex_align(u_pred_raw, u_ref, align_mask)
    u_pred = scale * u_pred_raw

    row = {
        "method": method,
        "alpha": alpha,
        "Mach": Mach,
        "csv": str(f),
        "n": len(y),
        "align_scale_real": scale.real,
        "align_scale_imag": scale.imag,
        "align_scale_abs": abs(scale),
        "max_abs_u_ref": float(np.max(np.abs(u_ref))),
        "max_abs_u_pred_raw": float(np.max(np.abs(u_pred_raw))),
        "max_abs_u_pred_aligned": float(np.max(np.abs(u_pred))),
    }

    for zname, zfun in zones.items():
        mask = zfun(y)
        row[f"u_rel_{zname}"] = rel_l2(y, u_pred, u_ref, mask)
        row[f"u_ref_l2_{zname}"] = float(np.sqrt(np.trapz(np.abs(u_ref[mask])**2, y[mask]))) if mask.sum() >= 3 else np.nan
        row[f"u_err_l2_{zname}"] = float(np.sqrt(np.trapz(np.abs((u_pred-u_ref)[mask])**2, y[mask]))) if mask.sum() >= 3 else np.nan

    rows.append(row)

metrics = pd.DataFrame(rows).sort_values(["alpha", "method"])
metrics.to_csv(OUT / "u_zone_error_metrics.csv", index=False)

print("[OK] wrote", OUT / "u_zone_error_metrics.csv")
print(metrics.to_string(index=False))

# Plots.
with PdfPages(OUT / "M050_u_zone_error_report.pdf") as pdf:
    for metric in [
        "u_rel_core_|y|<=0.5",
        "u_rel_core_|y|<=1",
        "u_rel_core_|y|<=2",
        "u_rel_inner_2<|y|<=5",
        "u_rel_outer_|y|>5",
        "u_rel_diagnostic_|y|<=15",
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, g in metrics.groupby("method"):
            g = g.sort_values("alpha")
            ax.plot(g["alpha"], g[metric], marker="o", label=method)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(metric)
        ax.set_title(f"M=0.5 — {metric}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(OUT / f"{metric.replace('|','abs').replace('<','lt').replace('>','gt').replace('=','eq').replace('/','_').replace(' ','_')}.png", dpi=220)
        plt.close(fig)

print("[OK] wrote", OUT / "M050_u_zone_error_report.pdf")
