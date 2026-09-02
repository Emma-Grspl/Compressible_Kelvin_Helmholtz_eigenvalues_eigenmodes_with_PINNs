#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("assets/pinn_subsonic/u_reconstruction_diagnostics_M050")
EXPORT_ROOT = ROOT / "fields_export"
OUT = ROOT
OUT.mkdir(parents=True, exist_ok=True)

def cplx(df, re_col, im_col):
    return df[re_col].to_numpy(float) + 1j * df[im_col].to_numpy(float)

def int_l2_sq(y, z, mask):
    if int(mask.sum()) < 3:
        return np.nan
    return float(np.trapz(np.abs(z[mask]) ** 2, y[mask]))

def rel_l2(y, pred, ref, mask):
    num = int_l2_sq(y, pred - ref, mask)
    den = int_l2_sq(y, ref, mask)
    if not np.isfinite(num) or not np.isfinite(den):
        return np.nan
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
    err = u_pred - u_ref

    full_err2 = int_l2_sq(y, err, zones["full"](y))
    full_ref2 = int_l2_sq(y, u_ref, zones["full"](y))

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
        err2 = int_l2_sq(y, err, mask)
        ref2 = int_l2_sq(y, u_ref, mask)

        row[f"u_rel_{name}"] = rel_l2(y, u_pred, u_ref, mask)
        row[f"u_err_frac_{name}"] = err2 / max(full_err2, 1e-300) if np.isfinite(err2) else np.nan
        row[f"u_ref_frac_{name}"] = ref2 / max(full_ref2, 1e-300) if np.isfinite(ref2) else np.nan

    rows.append(row)

metrics = pd.DataFrame(rows).sort_values(["alpha", "method"])
metrics.to_csv(OUT / "u_zone_error_metrics.csv", index=False)

cols = [
    "method", "alpha",
    "u_rel_core_abs_y_le_0p5",
    "u_rel_core_abs_y_le_1",
    "u_rel_core_abs_y_le_2",
    "u_rel_inner_2_lt_abs_y_le_5",
    "u_rel_outer_abs_y_gt_5",
    "u_rel_full",
    "u_err_frac_core_abs_y_le_0p5",
    "u_err_frac_core_abs_y_le_1",
    "u_err_frac_core_abs_y_le_2",
    "u_err_frac_inner_2_lt_abs_y_le_5",
    "u_err_frac_outer_abs_y_gt_5",
]

print("[OK] wrote", OUT / "u_zone_error_metrics.csv")
print(metrics[[c for c in cols if c in metrics.columns]].to_string(index=False))
