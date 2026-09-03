#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

OUTDIR = Path("assets/classic_supersonic/KH_shoot_collect/reports")
cand_path = OUTDIR / "KH_shoot_best_point_per_alpha_mach.csv"

refs = [
    Path("assets/classic_supersonic/csv/modal_reconstruction/validated_modal_points/table_supersonic_validated_modal_points.csv"),
    Path("assets/classic_supersonic/shooting/supersonic_reference_core_local_modal.csv"),
    Path("assets/classic_supersonic/shooting/supersonic_reference_core_local_spectral.csv"),
]

if not cand_path.exists():
    raise SystemExit(f"[FAIL] missing {cand_path}")

cand = pd.read_csv(cand_path)

def pick_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

allrefs = []
for rp in refs:
    if not rp.exists():
        continue
    df = pd.read_csv(rp)
    ca = pick_col(df, ["alpha", "a"])
    cm = pick_col(df, ["Mach", "M", "mach"])
    ccr = pick_col(df, ["cr", "c_r", "c_real", "phase_speed_real"])
    cci = pick_col(df, ["ci", "c_i", "c_imag", "phase_speed_imag"])
    if ca is None or cm is None or cci is None:
        continue
    sub = pd.DataFrame({
        "ref_source": str(rp),
        "Mach": pd.to_numeric(df[cm], errors="coerce"),
        "alpha": pd.to_numeric(df[ca], errors="coerce"),
        "cr_ref": pd.to_numeric(df[ccr], errors="coerce") if ccr is not None else np.nan,
        "ci_ref": pd.to_numeric(df[cci], errors="coerce"),
    })
    allrefs.append(sub)

if not allrefs:
    raise SystemExit("[WARN] no validated reference files found")

ref = pd.concat(allrefs, ignore_index=True).dropna(subset=["Mach", "alpha", "ci_ref"])
ref["Mach_round"] = ref["Mach"].round(6)
ref["alpha_round"] = ref["alpha"].round(6)
ref = ref.drop_duplicates(["Mach_round", "alpha_round"], keep="first")

cand["Mach_round"] = pd.to_numeric(cand["Mach"], errors="coerce").round(6)
cand["alpha_round"] = pd.to_numeric(cand["alpha"], errors="coerce").round(6)
cand["cr"] = pd.to_numeric(cand["cr"], errors="coerce")
cand["ci"] = pd.to_numeric(cand["ci"], errors="coerce")

m = cand.merge(ref, on=["Mach_round", "alpha_round"], how="left", suffixes=("", "_refrow"))

m["cr_abs_err"] = (m["cr"] - m["cr_ref"]).abs()
m["ci_abs_err"] = (m["ci"] - m["ci_ref"]).abs()
m["ci_rel_err"] = m["ci_abs_err"] / np.maximum(m["ci_ref"].abs(), 1e-12)

m["validated_match"] = (
    m["ci_abs_err"].notna()
    & (m["ci_abs_err"] < 1e-5)
    & (m["cr_abs_err"].fillna(0) < 1e-5)
)

out = OUTDIR / "KH_shoot_vs_validated.csv"
m.to_csv(out, index=False)

print("[OK] wrote", out)
cols = [
    "quality_flag", "Mach", "alpha", "cr", "ci",
    "cr_ref", "ci_ref", "cr_abs_err", "ci_abs_err", "ci_rel_err",
    "validated_match", "source", "ref_source"
]
cols = [c for c in cols if c in m.columns]
print(m[cols].to_string(index=False))
