#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


POINTS = Path("assets/classic_supersonic/csv/modal_reconstruction/validated_modal_points/table_supersonic_validated_modal_points.csv")
FIELDS = Path("assets/classic_supersonic/shooting/supersonic_reference_core_local_modal_fields.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mach", type=float, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    pts = pd.read_csv(POINTS)
    fld = pd.read_csv(FIELDS)

    pts["Mach"] = pd.to_numeric(pts["Mach"], errors="coerce")
    pts["alpha"] = pd.to_numeric(pts["alpha"], errors="coerce")

    print("[INFO] strict PDF validated points near requested Mach:")
    near = pts[(pts["Mach"] - args.mach).abs() < 0.35].copy()
    print(
        near[["Mach", "alpha", "line_id", "reference_cr", "reference_ci", "status", "source_group"]]
        .sort_values(["Mach", "alpha"])
        .to_string(index=False)
    )

    pts["_dist"] = (pts["Mach"] - args.mach).abs() + (pts["alpha"] - args.alpha).abs()
    row = pts.sort_values("_dist").iloc[0]

    M = float(row["Mach"])
    a = float(row["alpha"])
    line_id = str(row["line_id"])

    if abs(M - args.mach) > 1e-10 or abs(a - args.alpha) > 1e-10:
        print(f"[WARN] exact strict PDF point not found; selected nearest M={M}, alpha={a}, line_id={line_id}")

    fld["Mach"] = pd.to_numeric(fld["Mach"], errors="coerce")
    fld["alpha"] = pd.to_numeric(fld["alpha"], errors="coerce")

    sub = fld[
        np.isclose(fld["Mach"], M, atol=1e-10)
        & np.isclose(fld["alpha"], a, atol=1e-10)
        & (fld["line_id"].astype(str) == line_id)
    ].copy()

    if sub.empty:
        print("[WARN] no line_id match; falling back to Mach/alpha only.")
        sub = fld[
            np.isclose(fld["Mach"], M, atol=1e-10)
            & np.isclose(fld["alpha"], a, atol=1e-10)
        ].copy()

    if sub.empty:
        avail = fld[["Mach", "alpha", "line_id"]].drop_duplicates().sort_values(["Mach", "alpha"])
        print("[DEBUG] available fields near requested Mach:")
        print(avail[(avail["Mach"] - args.mach).abs() < 0.35].to_string(index=False))
        raise RuntimeError(f"No field rows found for selected M={M}, alpha={a}, line_id={line_id}")

    sub = sub.sort_values("y").reset_index(drop=True)

    y = pd.to_numeric(sub["y"], errors="coerce").to_numpy(float)
    p = (
        pd.to_numeric(sub["p_real"], errors="coerce").to_numpy(float)
        + 1j * pd.to_numeric(sub["p_imag"], errors="coerce").to_numpy(float)
    )

    norm = np.nanmax(np.abs(p))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Invalid pressure norm.")

    out = pd.DataFrame({
        "Mach": M,
        "alpha": a,
        "line_id": line_id,
        "point_id": row.get("point_id", ""),
        "source_group": row.get("source_group", ""),
        "y": y,
        "p_real": p.real,
        "p_imag": p.imag,
        "p_abs": np.abs(p),
        "p_real_norm": p.real / norm,
        "p_imag_norm": p.imag / norm,
        "p_abs_norm": np.abs(p) / norm,
        "rho_real": pd.to_numeric(sub["rho_real"], errors="coerce"),
        "rho_imag": pd.to_numeric(sub["rho_imag"], errors="coerce"),
        "u_real": pd.to_numeric(sub["u_real"], errors="coerce"),
        "u_imag": pd.to_numeric(sub["u_imag"], errors="coerce"),
        "v_real": pd.to_numeric(sub["v_real"], errors="coerce"),
        "v_imag": pd.to_numeric(sub["v_imag"], errors="coerce"),
        "reference_cr": row.get("reference_cr", np.nan),
        "reference_ci": row.get("reference_ci", np.nan),
        "stage1_mismatch": row.get("stage1_mismatch", np.nan),
        "stage2_mismatch": row.get("stage2_mismatch", np.nan),
        "status": row.get("status", ""),
        "source_csv": row.get("source_csv", ""),
        "source_label": row.get("source_label", ""),
    })

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("\n[OK] wrote", out_path)
    print(f"requested M={args.mach}, alpha={args.alpha}")
    print(f"selected  M={M}, alpha={a}, line_id={line_id}")
    print(f"cr={row.get('reference_cr', np.nan)}")
    print(f"ci={row.get('reference_ci', np.nan)}")
    print(f"rows={len(out)}")
    print(f"y=[{out['y'].min():.6g}, {out['y'].max():.6g}]")
    print(f"max|p|={norm:.6e}")


if __name__ == "__main__":
    main()
