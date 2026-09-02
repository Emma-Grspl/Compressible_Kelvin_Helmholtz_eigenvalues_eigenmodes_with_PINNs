#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

OUTDIR = Path("assets/classic_supersonic/KH_shoot_collect/reports")
FINAL = OUTDIR / "KH_shoot_FINAL23_curated_reference.csv"

if not FINAL.exists():
    raise SystemExit(f"[FAIL] missing {FINAL}. Run curate_KH_shoot_FINAL23.py first.")

final = pd.read_csv(FINAL)
final["Mach_key"] = pd.to_numeric(final["Mach"], errors="coerce").round(6)
final["alpha_key"] = pd.to_numeric(final["alpha"], errors="coerce").round(6)

search_roots = [
    Path("assets/classic_supersonic/shooting"),
    Path("assets/classic_supersonic/validated_modal_points"),
]

exclude_parts = [
    "KH_shoot_collect",
    "logs",
    "backups",
    "_dryrun",
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

def classify_source(path):
    s = str(path)
    if s.endswith("supersonic_validated_modal_points.csv"):
        return "FINAL23_source"
    if s.endswith("supersonic_reference_core_local_modal.csv"):
        return "FINAL23_source"
    if s.endswith("supersonic_reference_core_local_spectral.csv"):
        return "FINAL23_source"
    if "selected_reference_points" in s or "manual_single_selected_points" in s:
        return "selected_extension"
    if "dense" in s or "candidate" in s:
        return "candidate_dense"
    if "rebuilt_aggregates_latest" in s:
        return "rebuilt_aggregate_polluted"
    return "other"

rows = []

for root in search_roots:
    if not root.exists():
        continue

    for path in sorted(root.rglob("*.csv")):
        sp = str(path)
        if any(x in sp for x in exclude_parts):
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        alpha_col = pick_col(df, exact=["alpha", "a"], contains=["alpha"])
        mach_col = pick_col(df, exact=["Mach", "mach", "M"], contains=["mach"])

        cr_col = pick_col(
            df,
            exact=["reference_cr", "cr", "cr_ref", "cr_final", "cr_modal", "cr_spectral"],
            contains=["cr"],
        )
        ci_col = pick_col(
            df,
            exact=["reference_ci", "ci", "ci_ref", "ci_final", "ci_modal", "ci_spectral"],
            contains=["ci"],
        )

        if alpha_col is None or mach_col is None or ci_col is None:
            continue

        for i, r in df.iterrows():
            alpha = pd.to_numeric(pd.Series([r.get(alpha_col)]), errors="coerce").iloc[0]
            Mach = pd.to_numeric(pd.Series([r.get(mach_col)]), errors="coerce").iloc[0]
            cr = pd.to_numeric(pd.Series([r.get(cr_col)]), errors="coerce").iloc[0] if cr_col else np.nan
            ci = pd.to_numeric(pd.Series([r.get(ci_col)]), errors="coerce").iloc[0]

            if not np.isfinite(alpha) or not np.isfinite(Mach) or not np.isfinite(ci):
                continue
            if Mach < 1.0:
                continue

            item = {
                "source_kind": classify_source(path),
                "source": sp,
                "row": int(i),
                "Mach": float(Mach),
                "alpha": float(alpha),
                "cr": float(cr) if np.isfinite(cr) else np.nan,
                "ci": float(ci),
            }

            for c in df.columns:
                lc = c.lower()
                if c in ["best_status", "trusted_modal", "trusted_spectral", "valid_modal_candidate", "valid_spectral_candidate"]:
                    item[c] = r[c]
                elif any(k in lc for k in ["p_rel", "q_rel", "u_rel", "v_rel", "gamma_rel", "mismatch", "residual"]):
                    item[c] = r[c]

            rows.append(item)

cand = pd.DataFrame(rows)

if cand.empty:
    raise SystemExit("[FAIL] no candidate/reference rows found")

cand["Mach_key"] = pd.to_numeric(cand["Mach"], errors="coerce").round(6)
cand["alpha_key"] = pd.to_numeric(cand["alpha"], errors="coerce").round(6)
cand["cr"] = pd.to_numeric(cand["cr"], errors="coerce")
cand["ci"] = pd.to_numeric(cand["ci"], errors="coerce")

m = cand.merge(
    final[["Mach_key", "alpha_key", "reference_cr", "reference_ci", "reference_omega_i"]],
    on=["Mach_key", "alpha_key"],
    how="left",
)

m["has_FINAL23_match"] = m["reference_ci"].notna()
m["cr_abs_err"] = (m["cr"] - m["reference_cr"]).abs()
m["ci_abs_err"] = (m["ci"] - m["reference_ci"]).abs()
m["ci_rel_err"] = m["ci_abs_err"] / np.maximum(m["reference_ci"].abs(), 1e-12)

m["decision"] = "extension_or_unmatched"
m.loc[m["source_kind"] == "rebuilt_aggregate_polluted", "decision"] = "ignore_polluted_aggregate"
m.loc[m["has_FINAL23_match"] & ((m["cr_abs_err"] > 5e-3) | (m["ci_abs_err"] > 5e-3)), "decision"] = "reject_off_FINAL23_branch"
m.loc[m["has_FINAL23_match"] & (m["cr_abs_err"] <= 5e-3) & (m["ci_abs_err"] <= 5e-3), "decision"] = "near_FINAL23"
m.loc[m["has_FINAL23_match"] & (m["cr_abs_err"] <= 1e-5) & (m["ci_abs_err"] <= 1e-5), "decision"] = "exact_FINAL23"
m.loc[m["source_kind"] == "FINAL23_source", "decision"] = "canonical_FINAL23"

# Keep a compact useful table.
m = m.sort_values(["Mach", "alpha", "decision", "source_kind", "ci"], ascending=[True, True, True, True, False])

all_out = OUTDIR / "KH_shoot_candidates_vs_FINAL23_all.csv"
m.to_csv(all_out, index=False)

keep = m[m["decision"].isin(["canonical_FINAL23", "exact_FINAL23", "near_FINAL23", "extension_or_unmatched"])].copy()
keep_out = OUTDIR / "KH_shoot_candidates_vs_FINAL23_keep_or_review.csv"
keep.to_csv(keep_out, index=False)

reject = m[m["decision"].isin(["reject_off_FINAL23_branch", "ignore_polluted_aggregate"])].copy()
reject_out = OUTDIR / "KH_shoot_candidates_vs_FINAL23_reject.csv"
reject.to_csv(reject_out, index=False)

print("[OK] wrote", all_out)
print("[OK] wrote", keep_out)
print("[OK] wrote", reject_out)

print("\n[DECISION COUNTS]")
print(m["decision"].value_counts(dropna=False).to_string())

print("\n[CANONICAL FINAL23]")
canon = m[m["decision"] == "canonical_FINAL23"].drop_duplicates(["Mach_key", "alpha_key"])
print(canon[["Mach", "alpha", "cr", "ci", "reference_cr", "reference_ci", "source_kind", "source"]].to_string(index=False))

print("\n[NEAR OR EXACT NON-CANONICAL MATCHES]")
near = m[(m["decision"].isin(["exact_FINAL23", "near_FINAL23"])) & (m["source_kind"] != "FINAL23_source")]
cols = ["decision", "Mach", "alpha", "cr", "ci", "reference_cr", "reference_ci", "cr_abs_err", "ci_abs_err", "source_kind", "source"]
print(near[cols].to_string(index=False))

print("\n[EXTENSIONS / UNMATCHED TO REVIEW]")
ext = m[m["decision"] == "extension_or_unmatched"]
cols = ["Mach", "alpha", "cr", "ci", "source_kind", "source"]
print(ext[cols].to_string(index=False))
