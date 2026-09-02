#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def rel_l2(y, pred, ref, mask):
    if mask.sum() < 2:
        return np.nan
    yy = y[mask]
    num = np.trapz(np.abs(pred[mask] - ref[mask]) ** 2, yy)
    den = np.trapz(np.abs(ref[mask]) ** 2, yy)
    return float(np.sqrt(num / max(den, 1e-300)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    p = Path(args.run_dir) / "validated_csv_diagnostics/fields_vs_validated_csv.csv"
    df = pd.read_csv(p)

    y = df["y"].to_numpy(float)

    p_ref = df["p_ref_real"].to_numpy(float) + 1j * df["p_ref_imag"].to_numpy(float)
    p_pred = df["p_pred_aligned_real"].to_numpy(float) + 1j * df["p_pred_aligned_imag"].to_numpy(float)

    q_ref = df["q_ref_real"].to_numpy(float) + 1j * df["q_ref_imag"].to_numpy(float)
    q_pred = df["q_pred_aligned_real"].to_numpy(float) + 1j * df["q_pred_aligned_imag"].to_numpy(float)

    zones = {
        "center_|y|<=50": np.abs(y) <= 50,
        "center_|y|<=60": np.abs(y) <= 60,
        "diagnostic_|y|<=80": np.abs(y) <= 80,
        "left_-120_-80": (y >= -120) & (y <= -80),
        "right_50_120": (y >= 50) & (y <= 120),
        "right_y>80": y > 80,
        "right_y>120": y > 120,
        "left_y<-80": y < -80,
        "outside_|y|>80": np.abs(y) > 80,
        "full_domain": np.isfinite(y),
    }

    rows = []
    for name, mask in zones.items():
        rows.append({
            "zone": name,
            "n": int(mask.sum()),
            "p_rel": rel_l2(y, p_pred, p_ref, mask),
            "q_rel": rel_l2(y, q_pred, q_ref, mask),
            "max_abs_p_ref": float(np.nanmax(np.abs(p_ref[mask]))) if mask.any() else np.nan,
            "max_abs_p_pred": float(np.nanmax(np.abs(p_pred[mask]))) if mask.any() else np.nan,
            "mean_abs_p_ref": float(np.nanmean(np.abs(p_ref[mask]))) if mask.any() else np.nan,
            "mean_abs_p_pred": float(np.nanmean(np.abs(p_pred[mask]))) if mask.any() else np.nan,
        })

    out = Path(args.run_dir) / "validated_csv_diagnostics/zone_errors_vs_validated_csv.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("[OK] wrote", out)

if __name__ == "__main__":
    main()
