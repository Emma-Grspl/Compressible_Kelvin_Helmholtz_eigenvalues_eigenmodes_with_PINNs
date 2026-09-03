#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


ROOT = Path("assets/classic_supersonic/scan_M18_M19_alpha_0_03_strict/refined_near_valid_branch")

IN_FIELDS = ROOT / "refined_near_valid_modal_fields.csv"
IN_SPEC = ROOT / "refined_near_valid_candidates.csv"

OUT_DIR = ROOT / "dense_y_export"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIELDS = OUT_DIR / "refined_near_valid_modal_fields_dense_y_24001.csv"
OUT_SPEC = OUT_DIR / "refined_near_valid_candidates_dense_y_24001.csv"
OUT_SUMMARY = OUT_DIR / "summary_dense_y_24001.json"

N_Y = 24001

field_cols = [
    "p_real", "p_imag",
    "rho_real", "rho_imag",
    "u_real", "u_imag",
    "v_real", "v_imag",
]

fields = pd.read_csv(IN_FIELDS, low_memory=False)
spec = pd.read_csv(IN_SPEC)

for df in [fields, spec]:
    if "Mach" not in df.columns and "M" in df.columns:
        df.rename(columns={"M": "Mach"}, inplace=True)

fields["Mach"] = fields["Mach"].astype(float)
fields["alpha"] = fields["alpha"].astype(float)
fields["y"] = fields["y"].astype(float)

for c in field_cols:
    fields[c] = pd.to_numeric(fields[c], errors="coerce")

dense_parts = []
diag_rows = []

for (mach, alpha), g in fields.groupby(["Mach", "alpha"], sort=True):
    g = g.sort_values("y").copy()

    # Retire les doublons exacts en y.
    g = g.drop_duplicates("y", keep="first")

    y_old = g["y"].to_numpy(dtype=float)
    y_min = float(np.nanmin(y_old))
    y_max = float(np.nanmax(y_old))

    y_new = np.linspace(y_min, y_max, N_Y)

    out = pd.DataFrame({
        "Mach": mach,
        "alpha": alpha,
        "y": y_new,
    })

    # Colonnes métadonnées constantes.
    meta_cols = [
        c for c in g.columns
        if c not in field_cols and c not in ["y", "Mach", "alpha"]
    ]

    for c in meta_cols:
        vals = g[c].dropna()
        if len(vals):
            out[c] = vals.iloc[0]
        else:
            out[c] = np.nan

    for c in field_cols:
        v_old = g[c].to_numpy(dtype=float)

        mask = np.isfinite(y_old) & np.isfinite(v_old)
        yy = y_old[mask]
        vv = v_old[mask]

        if len(yy) < 4:
            out[c] = np.interp(y_new, yy, vv)
            continue

        # PCHIP évite les oscillations artificielles de spline cubique classique.
        if HAS_SCIPY:
            interp = PchipInterpolator(yy, vv, extrapolate=False)
            v_new = interp(y_new)
            # Les NaN éventuels aux bords sont remplacés par interpolation linéaire.
            if np.isnan(v_new).any():
                v_lin = np.interp(y_new, yy, vv)
                v_new = np.where(np.isfinite(v_new), v_new, v_lin)
        else:
            v_new = np.interp(y_new, yy, vv)

        out[c] = v_new

    out["source"] = "refined_near_valid_branch_dense_y_pchip"
    out["validation_status"] = "dense_y_resampled_requires_visual_confirmation"
    out["dense_y_n_points"] = N_Y

    dense_parts.append(out)

    dy_old = np.diff(y_old)
    dy_new = np.diff(y_new)
    diag_rows.append({
        "Mach": mach,
        "alpha": alpha,
        "n_old": int(len(y_old)),
        "n_new": int(len(y_new)),
        "dy_old_min": float(np.nanmin(dy_old)),
        "dy_old_median": float(np.nanmedian(dy_old)),
        "dy_old_max": float(np.nanmax(dy_old)),
        "dy_new": float(dy_new[0]),
        "y_min": y_min,
        "y_max": y_max,
    })

dense = pd.concat(dense_parts, ignore_index=True)
dense = dense.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)

dense.to_csv(OUT_FIELDS, index=False)

spec = spec.copy()
spec["source"] = "refined_near_valid_branch_dense_y_pchip"
spec["validation_status"] = "dense_y_resampled_requires_visual_confirmation"
spec["dense_y_n_points"] = N_Y
spec.to_csv(OUT_SPEC, index=False)

diag = pd.DataFrame(diag_rows)
diag.to_csv(OUT_DIR / "dense_y_diagnostics.csv", index=False)

summary = {
    "status": "dense_y_resampled_requires_visual_confirmation",
    "n_points": int(dense[["Mach", "alpha"]].drop_duplicates().shape[0]),
    "n_rows": int(len(dense)),
    "n_y_per_point": N_Y,
    "interpolation": "PCHIP" if HAS_SCIPY else "linear",
    "important_note": (
        "This densifies the modal-field export on a uniform y grid. "
        "It does not change cr/ci. Use for visual review/export; final validation still requires modal inspection."
    ),
    "by_Mach": spec.groupby("Mach").size().to_dict(),
}

OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2))
print("wrote:", OUT_FIELDS)
print("wrote:", OUT_SPEC)
print("wrote:", OUT_DIR / "dense_y_diagnostics.csv")
