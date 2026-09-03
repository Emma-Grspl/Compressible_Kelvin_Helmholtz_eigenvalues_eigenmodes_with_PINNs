#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

OUTDIR = Path("assets/classic_supersonic/KH_shoot_collect/reports")
cand_path = OUTDIR / "KH_shoot_strict_best_supersonic_points.csv"

refs = [
    Path("assets/classic_supersonic/csv/modal_reconstruction/validated_modal_points/table_supersonic_validated_modal_points.csv"),
    Path("assets/classic_supersonic/shooting/supersonic_reference_core_local_modal.csv"),
    Path("assets/classic_supersonic/shooting/supersonic_reference_core_local_spectral.csv"),
]

def pick_col(df, exact=(), contains=()):
    lower = {c.lower(): c for c in df.columns}
    for e in exact:
        if e.lower() in lower:
            return lower[e.lower()]
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in contains):
            return c
    return None

if not cand_path.exists():
    raise SystemExit(f"[FAIL] missing {cand_path}")

cand = pd.read_csv(cand_path)

ref_rows = []

for rp in refs:
    if not rp.exists():
        print("[WARN] missing", rp)
        continue

    df = pd.read_csv(rp)
    print("\n[REF]", rp)
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    alpha_col = pick_col(df, exact=["alpha", "a", "alpha_target"], contains=["alpha"])
    mach_col  = pick_col(df, exact=["Mach", "mach", "M", "mach_target"], contains=["mach"])

    cr_col = pick_col(
        df,
        exact=["cr", "cr_ref", "cr_final", "cr_modal", "cr_spectral", "c_real", "phase_speed_real"],
        contains=["cr"]
    )
    ci_col = pick_col(
        df,
        exact=["ci", "ci_ref", "ci_final", "ci_modal", "ci_spectral", "c_imag", "phase_speed_imag"],
        contains=["ci"]
    )

    print("picked:", alpha_col, mach_col, cr_col, ci_col)

    if alpha_col is None or mach_col is None or ci_col is None:
        print("[SKIP] cannot identify columns")
        continue

    sub = pd.DataFrame({
        "ref_source": str(rp),
        "Mach": pd.to_numeric(df[mach_col], errors="coerce"),
        "alpha": pd.to_numeric(df[alpha_col], errors="coerce"),
        "cr_ref": pd.to_numeric(df[cr_col], errors="coerce") if cr_col else np.nan,
        "ci_ref": pd.to_numeric(df[ci_col], errors="coerce"),
    }).dropna(subset=["Mach", "alpha", "ci_ref"])

    ref_rows.append(sub)

if not ref_rows:
    raise SystemExit("[FAIL] no usable reference rows")

ref = pd.concat(ref_rows, ignore_index=True)
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

m["validated_match_1e5"] = (
    m["ci_abs_err"].notna()
    & (m["ci_abs_err"] < 1e-5)
    & (m["cr_abs_err"].fillna(0) < 1e-5)
)

m["validated_match_1e3"] = (
    m["ci_abs_err"].notna()
    & (m["ci_abs_err"] < 1e-3)
    & (m["cr_abs_err"].fillna(0) < 1e-3)
)

out = OUTDIR / "KH_shoot_vs_validated_v2.csv"
m.to_csv(out, index=False)

print("\n[OK] wrote", out)
cols = [
    "quality_flag", "Mach", "alpha", "cr", "ci",
    "cr_ref", "ci_ref", "cr_abs_err", "ci_abs_err", "ci_rel_err",
    "validated_match_1e5", "validated_match_1e3", "source", "ref_source"
]
cols = [c for c in cols if c in m.columns]
print(m[cols].to_string(index=False))
