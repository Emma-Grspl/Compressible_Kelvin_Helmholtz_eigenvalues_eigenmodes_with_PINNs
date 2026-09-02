#!/usr/bin/env python3
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(".")
OUT = Path("assets/classic_supersonic/KH_shoot_prof_bundle")
OUT.mkdir(parents=True, exist_ok=True)

PDF = OUT / "KH_shoot_review_summary_PROF.pdf"

REFS = [
    Path("assets/classic_supersonic/csv/modal_reconstruction/validated_modal_points/table_supersonic_validated_modal_points.csv"),
    Path("assets/classic_supersonic/shooting/supersonic_reference_core_local_modal.csv"),
    Path("assets/classic_supersonic/shooting/supersonic_reference_core_local_spectral.csv"),
]

SCAN_ROOTS = [
    Path("assets/classic_supersonic/shooting"),
    Path("assets/classic_supersonic/validated_modal_points"),
]

EXCLUDE_PARTS = [
    "KH_shoot_collect",
    "KH_shoot_prof_bundle",
    "review_bundle",
    "logs",
    "backups",
    "_dryrun",
    "rebuilt_aggregates_latest",
    "modal_fields",
    "fields_vs",
    "profiles",
    "history",
]

MAX_CSV_BYTES = 80 * 1024 * 1024

def read_csv(path):
    return pd.read_csv(path, low_memory=False)

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

def nfloat(x):
    return pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]

def classify_source(path):
    s = str(path)
    if s.endswith("supersonic_validated_modal_points.csv"):
        return "canonical_FINAL23_source"
    if s.endswith("supersonic_reference_core_local_modal.csv"):
        return "canonical_FINAL23_source"
    if s.endswith("supersonic_reference_core_local_spectral.csv"):
        return "canonical_FINAL23_source"
    if "selected_reference_points" in s or "manual_single_selected_points" in s:
        return "selected_extension"
    if "dense" in s or "candidate" in s or "front" in s or "branch_guided" in s:
        return "candidate_dense"
    if "sparse_supersonic_expand" in s:
        return "sparse_extension"
    return "other_csv"

def compact_path(p):
    return str(p).replace("assets/classic_supersonic/", "")

# ---------------------------------------------------------------------
# 1. Build FINAL23 clean reference
# ---------------------------------------------------------------------
base_ref = None
for p in REFS:
    if p.exists():
        base_ref = p
        break

if base_ref is None:
    raise SystemExit("[FAIL] no FINAL23 reference file found")

base = read_csv(base_ref)
required = ["alpha", "Mach", "reference_cr", "reference_ci", "reference_omega_i"]
missing = [c for c in required if c not in base.columns]
if missing:
    raise SystemExit(f"[FAIL] {base_ref} missing required columns: {missing}")

final23 = base.copy()
for c in ["alpha", "Mach", "reference_cr", "reference_ci", "reference_omega_i"]:
    final23[c] = pd.to_numeric(final23[c], errors="coerce")

final23 = final23.dropna(subset=["alpha", "Mach", "reference_cr", "reference_ci"])
final23 = final23.sort_values(["Mach", "alpha"]).reset_index(drop=True)
final23["cr"] = final23["reference_cr"]
final23["ci"] = final23["reference_ci"]
final23["omega_i"] = final23["reference_omega_i"]
final23["canonical_source_file"] = str(base_ref)

final_cols = [
    "Mach", "alpha", "cr", "ci", "omega_i",
    "reference_cr", "reference_ci", "reference_omega_i",
    "line_id", "best_status",
    "best_stage1_mismatch", "best_stage2_mismatch",
    "best_spectral_success", "best_mode_success",
    "trusted_spectral", "trusted_modal",
    "valid_spectral_candidate", "valid_modal_candidate",
    "source_csv", "source_label", "source_group",
    "canonical_source_file",
]
final_cols = [c for c in final_cols if c in final23.columns]

final_out = OUT / "KH_shoot_FINAL23_points_complete.csv"
final23[final_cols].to_csv(final_out, index=False)

# Consistency across the three FINAL23 copies
consistency_rows = []
for p in REFS:
    if not p.exists():
        consistency_rows.append({"file": str(p), "status": "missing"})
        continue
    df = read_csv(p)
    if not all(c in df.columns for c in required):
        consistency_rows.append({"file": str(p), "status": "missing_required_columns"})
        continue

    tmp = df[required].copy()
    for c in required:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp = tmp.dropna(subset=["alpha", "Mach", "reference_cr", "reference_ci"])
    tmp["Mach_key"] = tmp["Mach"].round(6)
    tmp["alpha_key"] = tmp["alpha"].round(6)

    b = final23[required].copy()
    b["Mach_key"] = b["Mach"].round(6)
    b["alpha_key"] = b["alpha"].round(6)

    m = b.merge(tmp, on=["Mach_key", "alpha_key"], how="outer", suffixes=("_base", "_cmp"), indicator=True)
    both = m[m["_merge"] == "both"]

    row = {
        "file": str(p),
        "status": "ok",
        "rows": len(tmp),
        "merged_rows": len(both),
        "max_abs_dcr": np.nan,
        "max_abs_dci": np.nan,
        "max_abs_domega_i": np.nan,
        "nonmatching_rows": int((m["_merge"] != "both").sum()),
    }
    if len(both):
        row["max_abs_dcr"] = float(np.nanmax(np.abs(both["reference_cr_base"] - both["reference_cr_cmp"])))
        row["max_abs_dci"] = float(np.nanmax(np.abs(both["reference_ci_base"] - both["reference_ci_cmp"])))
        row["max_abs_domega_i"] = float(np.nanmax(np.abs(both["reference_omega_i_base"] - both["reference_omega_i_cmp"])))
    consistency_rows.append(row)

consistency = pd.DataFrame(consistency_rows)
consistency_out = OUT / "KH_shoot_FINAL23_consistency_check.csv"
consistency.to_csv(consistency_out, index=False)

# ---------------------------------------------------------------------
# 2. Scan useful candidate/reference CSVs
# ---------------------------------------------------------------------
skipped = []
rows = []

for root in SCAN_ROOTS:
    if not root.exists():
        skipped.append({"path": str(root), "reason": "missing_root"})
        continue

    for path in sorted(root.rglob("*.csv")):
        sp = str(path)

        if any(x in sp for x in EXCLUDE_PARTS):
            skipped.append({"path": sp, "reason": "excluded_by_path"})
            continue

        if path.stat().st_size > MAX_CSV_BYTES:
            skipped.append({"path": sp, "reason": f"too_large_{path.stat().st_size}_bytes"})
            continue

        try:
            df = read_csv(path)
        except Exception as e:
            skipped.append({"path": sp, "reason": f"read_error_{repr(e)}"})
            continue

        if df.empty:
            skipped.append({"path": sp, "reason": "empty"})
            continue

        alpha_col = pick_col(df, exact=["alpha", "a"], contains=["alpha"])
        mach_col = pick_col(df, exact=["Mach", "mach", "M"], contains=["mach"])
        cr_col = pick_col(
            df,
            exact=["reference_cr", "cr", "cr_ref", "cr_final", "cr_modal", "cr_spectral", "selected_cr"],
            contains=["cr"],
        )
        ci_col = pick_col(
            df,
            exact=["reference_ci", "ci", "ci_ref", "ci_final", "ci_modal", "ci_spectral", "selected_ci"],
            contains=["ci"],
        )

        if alpha_col is None or mach_col is None or ci_col is None:
            skipped.append({"path": sp, "reason": "missing_alpha_mach_or_ci"})
            continue

        for i, r in df.iterrows():
            alpha = nfloat(r.get(alpha_col))
            Mach = nfloat(r.get(mach_col))
            cr = nfloat(r.get(cr_col)) if cr_col else np.nan
            ci = nfloat(r.get(ci_col))

            if not np.isfinite(alpha) or not np.isfinite(Mach) or not np.isfinite(ci):
                continue
            if Mach < 1.0:
                continue

            item = {
                "source_kind": classify_source(path),
                "source": sp,
                "source_short": compact_path(path),
                "row": int(i),
                "Mach": float(Mach),
                "alpha": float(alpha),
                "cr": float(cr) if np.isfinite(cr) else np.nan,
                "ci": float(ci),
                "alpha_col": alpha_col,
                "mach_col": mach_col,
                "cr_col": cr_col,
                "ci_col": ci_col,
            }

            for c in [
                "reference_omega_i", "omega_i",
                "best_status", "best_stage1_mismatch", "best_stage2_mismatch",
                "best_spectral_success", "best_mode_success",
                "trusted_spectral", "trusted_modal",
                "valid_spectral_candidate", "valid_modal_candidate",
                "p_rel", "q_rel", "u_rel", "v_rel", "gamma_rel",
            ]:
                if c in df.columns:
                    item[c] = r[c]

            rows.append(item)

allpts = pd.DataFrame(rows)
skip_df = pd.DataFrame(skipped)
skip_out = OUT / "KH_shoot_scan_skipped_files.csv"
skip_df.to_csv(skip_out, index=False)

if allpts.empty:
    raise SystemExit("[FAIL] no points found while scanning")

for c in ["Mach", "alpha", "cr", "ci"]:
    allpts[c] = pd.to_numeric(allpts[c], errors="coerce")

# Match against FINAL23
ref = final23[["Mach", "alpha", "reference_cr", "reference_ci", "reference_omega_i"]].copy()
ref["Mach_key"] = ref["Mach"].round(6)
ref["alpha_key"] = ref["alpha"].round(6)

allpts["Mach_key"] = allpts["Mach"].round(6)
allpts["alpha_key"] = allpts["alpha"].round(6)

m = allpts.merge(
    ref[["Mach_key", "alpha_key", "reference_cr", "reference_ci", "reference_omega_i"]],
    on=["Mach_key", "alpha_key"],
    how="left",
)

m["has_FINAL23_match"] = m["reference_ci"].notna()
m["cr_abs_err"] = (m["cr"] - m["reference_cr"]).abs()
m["ci_abs_err"] = (m["ci"] - m["reference_ci"]).abs()
m["ci_rel_err"] = m["ci_abs_err"] / np.maximum(m["reference_ci"].abs(), 1e-12)

m["decision"] = "extension_or_unmatched"
m.loc[~np.isfinite(m["ci"]) | (m["ci"] <= 0), "decision"] = "bad_numeric"
m.loc[m["source_kind"] == "canonical_FINAL23_source", "decision"] = "canonical_FINAL23"
m.loc[
    m["has_FINAL23_match"] & (m["source_kind"] != "canonical_FINAL23_source")
    & (m["cr_abs_err"] <= 1e-5) & (m["ci_abs_err"] <= 1e-5),
    "decision"
] = "exact_FINAL23_duplicate"
m.loc[
    m["has_FINAL23_match"] & (m["source_kind"] != "canonical_FINAL23_source")
    & (m["cr_abs_err"] <= 5e-3) & (m["ci_abs_err"] <= 5e-3)
    & (m["decision"] != "exact_FINAL23_duplicate"),
    "decision"
] = "near_FINAL23"
m.loc[
    m["has_FINAL23_match"] & (m["source_kind"] != "canonical_FINAL23_source")
    & ((m["cr_abs_err"] > 5e-3) | (m["ci_abs_err"] > 5e-3)),
    "decision"
] = "off_branch_vs_FINAL23"

m = m.sort_values(["Mach", "alpha", "decision", "source_kind", "ci"], ascending=[True, True, True, True, False])

all_out = OUT / "KH_shoot_ALL_points_scanned_complete.csv"
m.to_csv(all_out, index=False)

keep = m[m["decision"].isin([
    "canonical_FINAL23",
    "exact_FINAL23_duplicate",
    "near_FINAL23",
    "extension_or_unmatched",
])].copy()

reject = m[m["decision"].isin([
    "off_branch_vs_FINAL23",
    "bad_numeric",
])].copy()

keep_out = OUT / "KH_shoot_points_keep_or_review.csv"
reject_out = OUT / "KH_shoot_points_reject_or_ignore.csv"

keep.to_csv(keep_out, index=False)
reject.to_csv(reject_out, index=False)

# ---------------------------------------------------------------------
# 3. Fast PDF - text only, no matplotlib tables
# ---------------------------------------------------------------------
def fmt_table(df, cols, max_rows=None, max_col_width=42):
    if df is None or df.empty:
        return "No rows."
    cols = [c for c in cols if c in df.columns]
    d = df[cols].copy()
    if max_rows is not None:
        d = d.head(max_rows)
    for c in d.columns:
        if d[c].dtype.kind in "fc":
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            d[c] = d[c].astype(str).map(lambda x: x[:max_col_width])
    return d.to_string(index=False)

def add_text_pages(pdf, title, text, fontsize=7.4, lines_per_page=42):
    lines = str(text).splitlines()
    if not lines:
        lines = [""]
    for start in range(0, len(lines), lines_per_page):
        part = lines[start:start + lines_per_page]
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(111)
        ax.axis("off")
        suffix = "" if start == 0 else f" continued {start//lines_per_page + 1}"
        ax.text(0.02, 0.97, title + suffix, fontsize=15, weight="bold", va="top")
        y = 0.91
        for line in part:
            ax.text(0.02, y, line, fontsize=fontsize, family="monospace", va="top")
            y -= 0.021
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

with PdfPages(PDF) as pdf:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    intro = []
    intro.append(f"Generated: {now}")
    intro.append(f"Output dir: {OUT}")
    intro.append("")
    intro.append("Main deliverables:")
    for p in [PDF, final_out, all_out, keep_out, reject_out, consistency_out, skip_out]:
        intro.append(f"- {p.name} : exists={p.exists()} size={p.stat().st_size/1024:.1f} KiB")
    add_text_pages(pdf, "KH_shoot - professor review bundle", "\n".join(intro), fontsize=8)

    final_text = "FINAL23 clean reference, complete canonical points.\n\n"
    final_text += fmt_table(
        final23,
        ["Mach", "alpha", "reference_cr", "reference_ci", "reference_omega_i", "best_status", "trusted_spectral", "trusted_modal"],
        max_rows=40,
        max_col_width=30,
    )
    add_text_pages(pdf, "FINAL23 canonical points", final_text, fontsize=7.2)

    cons_text = "Consistency check across the 3 FINAL23 files.\n\n"
    cons_text += fmt_table(consistency, list(consistency.columns), max_rows=20, max_col_width=70)
    add_text_pages(pdf, "FINAL23 consistency check", cons_text, fontsize=7.2)

    counts = []
    counts.append(f"ALL scanned points shape: {m.shape}")
    counts.append(f"KEEP/review shape: {keep.shape}")
    counts.append(f"REJECT/ignore shape: {reject.shape}")
    counts.append("")
    counts.append("[decision counts]")
    counts.append(m["decision"].value_counts(dropna=False).to_string())
    counts.append("")
    counts.append("[source_kind counts]")
    counts.append(m["source_kind"].value_counts(dropna=False).to_string())
    add_text_pages(pdf, "All scanned points - counts", "\n".join(counts), fontsize=8)

    near = m[m["decision"].isin(["canonical_FINAL23", "exact_FINAL23_duplicate", "near_FINAL23"])].copy()
    near_text = fmt_table(
        near,
        ["decision", "source_kind", "Mach", "alpha", "cr", "ci", "reference_cr", "reference_ci", "cr_abs_err", "ci_abs_err", "source_short"],
        max_rows=80,
        max_col_width=45,
    )
    add_text_pages(pdf, "Canonical / exact / near FINAL23 points", near_text, fontsize=5.9, lines_per_page=48)

    ext = m[m["decision"] == "extension_or_unmatched"].copy()
    ext_text = fmt_table(
        ext,
        ["source_kind", "Mach", "alpha", "cr", "ci", "source_short"],
        max_rows=80,
        max_col_width=55,
    )
    add_text_pages(pdf, "Extensions / unmatched points to review", ext_text, fontsize=6.1, lines_per_page=48)

    rej_text = fmt_table(
        reject,
        ["decision", "source_kind", "Mach", "alpha", "cr", "ci", "reference_cr", "reference_ci", "cr_abs_err", "ci_abs_err", "source_short"],
        max_rows=80,
        max_col_width=45,
    )
    add_text_pages(pdf, "Rejected / ignored points - first rows", rej_text, fontsize=5.8, lines_per_page=48)

readme = OUT / "README_PROF_BUNDLE.txt"
readme.write_text(
    "KH_shoot professor bundle\n"
    "========================\n\n"
    "Primary clean reference:\n"
    "- KH_shoot_FINAL23_points_complete.csv\n\n"
    "All scanned points:\n"
    "- KH_shoot_ALL_points_scanned_complete.csv\n\n"
    "Points to keep/review:\n"
    "- KH_shoot_points_keep_or_review.csv\n\n"
    "Points rejected/ignored:\n"
    "- KH_shoot_points_reject_or_ignore.csv\n\n"
    "PDF summary:\n"
    "- KH_shoot_review_summary_PROF.pdf\n\n"
    "Notes:\n"
    "- The PDF is intentionally text-only and short. The complete data are in the CSV files.\n"
    "- FINAL23 is built from the validated reference files with reference_cr/reference_ci/reference_omega_i.\n"
)

manifest = OUT / "MANIFEST.txt"
manifest.write_text("\n".join(str(p.relative_to(OUT)) for p in sorted(OUT.rglob("*")) if p.is_file()) + "\n")

print("[OK] wrote", PDF)
print("[OK] wrote", final_out)
print("[OK] wrote", all_out)
print("[OK] wrote", keep_out)
print("[OK] wrote", reject_out)
print("[OK] wrote", consistency_out)
print("[OK] wrote", skip_out)
print("[OK] wrote", readme)
print("[OK] wrote", manifest)
print("")
print("[SUMMARY]")
print("FINAL23:", final23.shape)
print("ALL:", m.shape)
print("KEEP:", keep.shape)
print("REJECT:", reject.shape)
print("")
print(m["decision"].value_counts(dropna=False).to_string())
