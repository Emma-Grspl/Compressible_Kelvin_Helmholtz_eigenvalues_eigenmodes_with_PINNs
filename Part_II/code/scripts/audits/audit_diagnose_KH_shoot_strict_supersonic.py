#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import pandas as pd

OUTDIR = Path("assets/classic_supersonic/KH_shoot_collect/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = [
    Path("assets/classic_supersonic/shooting"),
    Path("assets/classic_supersonic/validated_modal_points"),
]

EXCLUDE = [
    "backups",
    "_dryrun",
    "logs",
    "KH_shoot_collect",
    "fields",
    "profiles",
]

def pick_col(df, candidates, contains_any=None):
    lower = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    if contains_any:
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in contains_any):
                return c

    return None

def read_csv_safe(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return None

csvs = []
for root in SEARCH_ROOTS:
    if root.exists():
        for p in root.rglob("*.csv"):
            sp = str(p)
            if any(x in sp for x in EXCLUDE):
                continue
            csvs.append(p)

rows = []

for p in sorted(set(csvs)):
    df = read_csv_safe(p)
    if df is None or df.empty:
        continue

    alpha_col = pick_col(df, ["alpha", "a", "alpha_target"], ["alpha"])
    mach_col  = pick_col(df, ["Mach", "mach", "M", "mach_target"], ["mach"])
    cr_col    = pick_col(df, ["cr", "c_r", "c_real", "cr_ref", "cr_validated", "cr_modal", "cr_spectral", "cr_final"], ["cr"])
    ci_col    = pick_col(df, ["ci", "c_i", "c_imag", "ci_ref", "ci_validated", "ci_modal", "ci_spectral", "ci_final"], ["ci"])

    if alpha_col is None or mach_col is None:
        continue
    if ci_col is None:
        continue

    for i, r in df.iterrows():
        alpha = pd.to_numeric(pd.Series([r.get(alpha_col)]), errors="coerce").iloc[0]
        Mach  = pd.to_numeric(pd.Series([r.get(mach_col)]), errors="coerce").iloc[0]
        cr    = pd.to_numeric(pd.Series([r.get(cr_col)]), errors="coerce").iloc[0] if cr_col else np.nan
        ci    = pd.to_numeric(pd.Series([r.get(ci_col)]), errors="coerce").iloc[0]

        if not np.isfinite(alpha) or not np.isfinite(Mach) or not np.isfinite(ci):
            continue
        if Mach < 1.0:
            continue

        row = {
            "source": str(p),
            "row": i,
            "alpha": float(alpha),
            "Mach": float(Mach),
            "cr": float(cr) if np.isfinite(cr) else np.nan,
            "ci": float(ci),
            "alpha_col": alpha_col,
            "mach_col": mach_col,
            "cr_col": cr_col,
            "ci_col": ci_col,
        }

        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in [
                "status", "accepted", "valid", "resolved",
                "p_rel", "q_rel", "u_rel", "v_rel", "gamma_rel",
                "tail", "res", "resid", "mismatch", "err", "error",
                "source_kind", "branch", "anchor"
            ]):
                row[c] = r[c]

        rows.append(row)

allpts = pd.DataFrame(rows)

if allpts.empty:
    raise SystemExit("[FAIL] no strict supersonic spectral points found")

for c in ["alpha", "Mach", "cr", "ci"]:
    allpts[c] = pd.to_numeric(allpts[c], errors="coerce")

# Remove obvious non-physical / failed points.
allpts["finite"] = np.isfinite(allpts["alpha"]) & np.isfinite(allpts["Mach"]) & np.isfinite(allpts["ci"])
allpts["unstable"] = allpts["ci"] > 1e-8
allpts["reasonable_cr"] = allpts["cr"].between(-2.5, 2.5) | allpts["cr"].isna()

# Detect status-like columns.
status_text = pd.Series([""] * len(allpts), index=allpts.index)
for c in allpts.columns:
    lc = c.lower()
    if any(k in lc for k in ["status", "accepted", "valid", "resolved"]):
        status_text = status_text + " " + allpts[c].astype(str).str.lower()

allpts["status_text"] = status_text

bad_words = ["rejected", "dry_run", "unresolved", "failure", "false", "nan"]
good_words = ["validated", "accepted", "resolved", "true"]

allpts["has_bad_status"] = allpts["status_text"].apply(lambda s: any(w in s for w in bad_words))
allpts["has_good_status"] = allpts["status_text"].apply(lambda s: any(w in s for w in good_words))

allpts["quality_flag"] = "candidate"
allpts.loc[~allpts["finite"] | ~allpts["unstable"] | ~allpts["reasonable_cr"], "quality_flag"] = "bad_numeric"
allpts.loc[allpts["has_bad_status"], "quality_flag"] = "bad_status"
allpts.loc[
    (allpts["quality_flag"] == "candidate") & allpts["has_good_status"],
    "quality_flag"
] = "validated_or_resolved"

# If modal errors exist, use them.
for c in ["p_rel", "q_rel", "u_rel", "v_rel", "gamma_rel"]:
    if c in allpts.columns:
        allpts[c] = pd.to_numeric(allpts[c], errors="coerce")

if "p_rel" in allpts.columns and "q_rel" in allpts.columns:
    good_modal = (
        (allpts["quality_flag"].isin(["candidate", "validated_or_resolved"]))
        & (allpts["p_rel"].notna())
        & (allpts["q_rel"].notna())
        & (allpts["p_rel"] < 0.05)
        & (allpts["q_rel"] < 0.08)
    )
    allpts.loc[good_modal, "quality_flag"] = "good_modal"

    suspicious_modal = (
        (allpts["p_rel"].notna())
        & (allpts["q_rel"].notna())
        & ((allpts["p_rel"] > 0.20) | (allpts["q_rel"] > 0.30))
    )
    allpts.loc[suspicious_modal, "quality_flag"] = "suspicious_modal"

# Deduplicate by (Mach, alpha). Prefer validated/good, then higher ci.
rank = {
    "good_modal": 0,
    "validated_or_resolved": 1,
    "candidate": 2,
    "suspicious_modal": 3,
    "bad_status": 4,
    "bad_numeric": 5,
}
allpts["rank"] = allpts["quality_flag"].map(rank).fillna(9)
allpts["Mach_round"] = allpts["Mach"].round(6)
allpts["alpha_round"] = allpts["alpha"].round(6)

allpts = allpts.sort_values(
    ["Mach_round", "alpha_round", "rank", "ci"],
    ascending=[True, True, True, False]
)

best = allpts.drop_duplicates(["Mach_round", "alpha_round"], keep="first").copy()

all_out = OUTDIR / "KH_shoot_strict_all_supersonic_points.csv"
best_out = OUTDIR / "KH_shoot_strict_best_supersonic_points.csv"
good_out = OUTDIR / "KH_shoot_strict_good_supersonic_points.csv"
bad_out = OUTDIR / "KH_shoot_strict_bad_supersonic_points.csv"

allpts.to_csv(all_out, index=False)
best.to_csv(best_out, index=False)
allpts[allpts["quality_flag"].isin(["good_modal", "validated_or_resolved", "candidate"])].to_csv(good_out, index=False)
allpts[allpts["quality_flag"].isin(["suspicious_modal", "bad_status", "bad_numeric"])].to_csv(bad_out, index=False)

print("[OK] wrote", all_out)
print("[OK] wrote", best_out)
print("[OK] wrote", good_out)
print("[OK] wrote", bad_out)

print("\n[SUMMARY]")
print(allpts["quality_flag"].value_counts(dropna=False).to_string())

print("\n[BEST]")
cols = ["quality_flag", "Mach", "alpha", "cr", "ci", "source"]
for c in ["p_rel", "q_rel", "u_rel", "v_rel", "gamma_rel", "status_text"]:
    if c in best.columns:
        cols.append(c)
print(best[cols].to_string(index=False))
