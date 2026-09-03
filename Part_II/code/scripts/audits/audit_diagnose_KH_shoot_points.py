#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
OUTDIR = Path("assets/classic_supersonic/KH_shoot_collect/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

INV = OUTDIR / "KH_shoot_csv_inventory.csv"
if not INV.exists():
    raise SystemExit(f"[FAIL] missing inventory: {INV}")

inv = pd.read_csv(INV)

def pick_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

rows = []

for _, r in inv.iterrows():
    path = Path(r["path"])
    if not path.exists():
        continue
    if not bool(r.get("has_alpha", False)) or not bool(r.get("has_ci", False)):
        continue

    try:
        df = pd.read_csv(path)
    except Exception:
        continue

    ca = pick_col(df, ["alpha", "a"])
    cm = pick_col(df, ["Mach", "M", "mach"])
    ccr = pick_col(df, ["cr", "c_r", "c_real", "phase_speed_real"])
    cci = pick_col(df, ["ci", "c_i", "c_imag", "phase_speed_imag"])

    if ca is None or cm is None or cci is None:
        continue

    for i, row in df.iterrows():
        alpha = row.get(ca, np.nan)
        Mach = row.get(cm, np.nan)
        cr = row.get(ccr, np.nan) if ccr is not None else np.nan
        ci = row.get(cci, np.nan)

        item = {
            "source": str(path),
            "row": int(i),
            "alpha": alpha,
            "Mach": Mach,
            "cr": cr,
            "ci": ci,
        }

        # Pull useful diagnostic columns if present.
        for col in df.columns:
            lc = col.lower()
            if any(k in lc for k in [
                "res", "resid", "mismatch", "det", "bc", "tail", "cond",
                "p_rel", "q_rel", "u_rel", "v_rel", "gamma_rel",
                "err", "error", "success", "status"
            ]):
                item[col] = row[col]

        rows.append(item)

allpts = pd.DataFrame(rows)
raw_out = OUTDIR / "KH_shoot_all_spectral_points_raw.csv"
allpts.to_csv(raw_out, index=False)

if allpts.empty:
    print("[WARN] no spectral points found")
    print("[OK] wrote", raw_out)
    raise SystemExit(0)

for c in ["alpha", "Mach", "cr", "ci"]:
    allpts[c] = pd.to_numeric(allpts[c], errors="coerce")

allpts["finite_alpha_M_ci"] = np.isfinite(allpts["alpha"]) & np.isfinite(allpts["Mach"]) & np.isfinite(allpts["ci"])
allpts["finite_cr"] = np.isfinite(allpts["cr"])
allpts["unstable"] = allpts["ci"] > 0
allpts["cr_in_reasonable_range"] = allpts["cr"].between(-2.5, 2.5) | ~allpts["finite_cr"]

# Generic residual/error score from available columns.
score_cols = []
for c in allpts.columns:
    lc = c.lower()
    if any(k in lc for k in ["res", "resid", "mismatch", "det", "bc", "tail", "err", "error"]):
        if c not in ["ci_abs_err", "ci_rel_err"]:
            val = pd.to_numeric(allpts[c], errors="coerce")
            if val.notna().any():
                score_cols.append(c)

if score_cols:
    score = np.zeros(len(allpts), dtype=float)
    used = np.zeros(len(allpts), dtype=bool)
    for c in score_cols:
        v = pd.to_numeric(allpts[c], errors="coerce").abs().to_numpy()
        good = np.isfinite(v)
        # log score: smaller is better
        score[good] += np.log10(np.maximum(v[good], 1e-300))
        used |= good
    allpts["generic_log_score"] = np.where(used, score, np.nan)
else:
    allpts["generic_log_score"] = np.nan

# Main classification, deliberately conservative.
allpts["quality_flag"] = "suspicious"

allpts.loc[
    allpts["finite_alpha_M_ci"]
    & allpts["unstable"]
    & allpts["cr_in_reasonable_range"],
    "quality_flag"
] = "candidate"

allpts.loc[
    (~allpts["finite_alpha_M_ci"])
    | (~allpts["unstable"])
    | (~allpts["cr_in_reasonable_range"]),
    "quality_flag"
] = "bad"

# If clear relative modal errors exist, upgrade/downgrade.
rel_cols = [c for c in ["p_rel", "q_rel", "u_rel", "v_rel", "gamma_rel"] if c in allpts.columns]
for c in rel_cols:
    allpts[c] = pd.to_numeric(allpts[c], errors="coerce")

if "p_rel" in rel_cols and "q_rel" in rel_cols:
    allpts.loc[
        (allpts["quality_flag"] == "candidate")
        & (allpts["p_rel"] < 0.05)
        & (allpts["q_rel"] < 0.08),
        "quality_flag"
    ] = "good_modal"

    allpts.loc[
        (allpts["p_rel"] > 0.2) | (allpts["q_rel"] > 0.3),
        "quality_flag"
    ] = "suspicious"

# Deduplicate by rounded Mach/alpha, keeping strongest ci or best modal.
allpts["Mach_round"] = allpts["Mach"].round(6)
allpts["alpha_round"] = allpts["alpha"].round(6)

sort_cols = ["Mach_round", "alpha_round", "quality_rank", "ci"]
rank = {
    "good_modal": 0,
    "candidate": 1,
    "suspicious": 2,
    "bad": 3,
}
allpts["quality_rank"] = allpts["quality_flag"].map(rank).fillna(9)
allpts = allpts.sort_values(["Mach_round", "alpha_round", "quality_rank", "ci"], ascending=[True, True, True, False])

best = allpts.drop_duplicates(["Mach_round", "alpha_round"], keep="first").copy()

all_out = OUTDIR / "KH_shoot_all_spectral_points_diagnosed.csv"
best_out = OUTDIR / "KH_shoot_best_point_per_alpha_mach.csv"
good_out = OUTDIR / "KH_shoot_good_or_candidate_points.csv"
bad_out = OUTDIR / "KH_shoot_bad_or_suspicious_points.csv"

allpts.to_csv(all_out, index=False)
best.to_csv(best_out, index=False)
allpts[allpts["quality_flag"].isin(["good_modal", "candidate"])].to_csv(good_out, index=False)
allpts[allpts["quality_flag"].isin(["suspicious", "bad"])].to_csv(bad_out, index=False)

print("[OK] wrote", all_out)
print("[OK] wrote", best_out)
print("[OK] wrote", good_out)
print("[OK] wrote", bad_out)

print("\n[SUMMARY quality_flag]")
print(allpts["quality_flag"].value_counts(dropna=False).to_string())

print("\n[BEST POINTS]")
cols = ["quality_flag", "Mach", "alpha", "cr", "ci", "source"]
extra = [c for c in ["p_rel", "q_rel", "u_rel", "v_rel", "generic_log_score"] if c in best.columns]
print(best[cols + extra].to_string(index=False))
