#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pick_col(df, names, required=True):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    if required:
        raise KeyError(f"Missing one of {names}. Columns={list(df.columns)}")
    return None


def get_meta_value(row, names):
    for n in names:
        if n in row.index:
            return row[n]
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mach", type=float, default=1.6)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--fields-csv", default="assets/classic_supersonic/shooting/supersonic_shooting_core_band_M16_fields.csv")
    ap.add_argument("--summary-csv", default="assets/classic_supersonic/shooting/supersonic_shooting_core_band_M16_summary.csv")
    args = ap.parse_args()

    fields_path = Path(args.fields_csv)
    summary_path = Path(args.summary_csv)

    if not fields_path.exists():
        raise FileNotFoundError(fields_path)

    fields = pd.read_csv(fields_path)

    mach_col = pick_col(fields, ["Mach", "mach", "M"], required=False)
    alpha_col = pick_col(fields, ["alpha", "a"])
    y_col = pick_col(fields, ["y"])
    pre_col = pick_col(fields, ["p_real", "p_re", "real_p"])
    pim_col = pick_col(fields, ["p_imag", "p_im", "imag_p"], required=False)

    fields[alpha_col] = pd.to_numeric(fields[alpha_col], errors="coerce")
    if mach_col is not None:
        fields[mach_col] = pd.to_numeric(fields[mach_col], errors="coerce")
        sub = fields[
            np.isclose(fields[mach_col], args.mach, atol=1e-10)
            & np.isclose(fields[alpha_col], args.alpha, atol=1e-10)
        ].copy()
    else:
        sub = fields[np.isclose(fields[alpha_col], args.alpha, atol=1e-10)].copy()

    if sub.empty:
        cols = [c for c in [mach_col, alpha_col] if c is not None]
        avail = fields[cols].drop_duplicates().sort_values(cols)
        print("[DEBUG] available field points:")
        print(avail.to_string(index=False))
        raise RuntimeError(f"No M16 field rows for Mach={args.mach}, alpha={args.alpha}")

    sub = sub.sort_values(y_col).reset_index(drop=True)

    meta = {}
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        s_alpha = pick_col(summary, ["alpha", "a"], required=False)
        s_mach = pick_col(summary, ["Mach", "mach", "M"], required=False)
        if s_alpha is not None:
            summary[s_alpha] = pd.to_numeric(summary[s_alpha], errors="coerce")
            smask = np.isclose(summary[s_alpha], args.alpha, atol=1e-10)
            if s_mach is not None:
                summary[s_mach] = pd.to_numeric(summary[s_mach], errors="coerce")
                smask &= np.isclose(summary[s_mach], args.mach, atol=1e-10)
            if smask.any():
                meta = summary[smask].iloc[0]
            else:
                print("[WARN] no exact summary row; using nearest summary row.")
                summary["_dist"] = (summary[s_alpha] - args.alpha).abs()
                if s_mach is not None:
                    summary["_dist"] += (summary[s_mach] - args.mach).abs()
                meta = summary.sort_values("_dist").iloc[0]

    y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(float)
    p_imag = np.zeros(len(sub)) if pim_col is None else pd.to_numeric(sub[pim_col], errors="coerce").to_numpy(float)
    p = pd.to_numeric(sub[pre_col], errors="coerce").to_numpy(float) + 1j * p_imag

    norm = np.nanmax(np.abs(p))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Invalid pressure norm.")

    out = pd.DataFrame({
        "Mach": args.mach,
        "alpha": args.alpha,
        "y": y,
        "p_real": p.real,
        "p_imag": p.imag,
        "p_abs": np.abs(p),
        "p_real_norm": p.real / norm,
        "p_imag_norm": p.imag / norm,
        "p_abs_norm": np.abs(p) / norm,
        "reference_cr": get_meta_value(meta, ["reference_cr", "cr", "best_cr"]),
        "reference_ci": get_meta_value(meta, ["reference_ci", "ci", "best_ci"]),
        "reference_omega_i": get_meta_value(meta, ["reference_omega_i", "omega_i", "best_omega_i"]),
        "status": get_meta_value(meta, ["status", "best_status"]),
        "source_fields_csv": str(fields_path),
        "source_summary_csv": str(summary_path) if summary_path.exists() else "",
    })

    for col in ["rho_real", "rho_imag", "u_real", "u_imag", "v_real", "v_imag"]:
        if col in sub.columns:
            out[col] = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("[OK] wrote", out_path)
    print("fields:", fields_path)
    print("summary:", summary_path if summary_path.exists() else "MISSING")
    print(f"M={args.mach}, alpha={args.alpha}")
    print("cr=", out["reference_cr"].iloc[0])
    print("ci=", out["reference_ci"].iloc[0])
    print("rows=", len(out), "y=[", out["y"].min(), ",", out["y"].max(), "]")
    print("max|p|=", norm)


if __name__ == "__main__":
    main()
