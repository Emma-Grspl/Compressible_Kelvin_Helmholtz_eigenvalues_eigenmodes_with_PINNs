#!/usr/bin/env python3
from pathlib import Path
import textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path("assets/classic_supersonic/KH_shoot_collect")
REPORTS = ROOT / "reports"
OUTDIR = ROOT / "review_bundle"
OUTDIR.mkdir(parents=True, exist_ok=True)

PDF = OUTDIR / "KH_shoot_review_summary.pdf"

FILES = {
    "FINAL23": REPORTS / "KH_shoot_FINAL23_curated_reference.csv",
    "candidates_all": REPORTS / "KH_shoot_candidates_vs_FINAL23_all.csv",
    "keep_or_review": REPORTS / "KH_shoot_candidates_vs_FINAL23_keep_or_review.csv",
    "reject": REPORTS / "KH_shoot_candidates_vs_FINAL23_reject.csv",
    "strict_best": REPORTS / "KH_shoot_strict_best_supersonic_points.csv",
    "errors": REPORTS / "KH_shoot_log_errors.txt",
    "final23_log": REPORTS / "KH_shoot_FINAL23_curated_reference.log",
    "candidate_log": REPORTS / "KH_shoot_candidates_vs_FINAL23.log",
}

def add_text_page(pdf, title, lines, fontsize=9):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.02, 0.96, title, fontsize=16, weight="bold", va="top")
    y = 0.90
    for line in lines:
        wrapped = textwrap.wrap(str(line), width=145) or [""]
        for w in wrapped:
            ax.text(0.02, y, w, fontsize=fontsize, family="monospace", va="top")
            y -= 0.035
            if y < 0.04:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(11.69, 8.27))
                ax = fig.add_subplot(111)
                ax.axis("off")
                ax.text(0.02, 0.96, title + " continued", fontsize=16, weight="bold", va="top")
                y = 0.90
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def add_df_pages(pdf, title, df, cols=None, max_rows_per_page=24, fontsize=6):
    if df is None or df.empty:
        add_text_page(pdf, title, ["No data."])
        return

    if cols is not None:
        cols = [c for c in cols if c in df.columns]
        df = df[cols].copy()

    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in "fc":
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            df[c] = df[c].astype(str).map(lambda x: x[:90])

    n = len(df)
    for start in range(0, n, max_rows_per_page):
        part = df.iloc[start:start + max_rows_per_page]
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title(f"{title} rows {start+1}-{min(start+max_rows_per_page, n)} / {n}", fontsize=14, weight="bold", pad=12)

        table = ax.table(
            cellText=part.values,
            colLabels=part.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1, 1.25)

        for _, cell in table.get_celld().items():
            cell.set_linewidth(0.25)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

def read_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)

def read_lines(path, max_lines=120):
    if not path.exists():
        return [f"MISSING: {path}"]
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) > max_lines:
        return lines[:max_lines] + [f"... truncated: showing {max_lines} of {len(lines)} lines ..."]
    return lines

with PdfPages(PDF) as pdf:
    status = []
    for name, path in FILES.items():
        if path.exists():
            status.append(f"[OK] {name:16s} {path}  size={path.stat().st_size/1024:.1f} KiB")
        else:
            status.append(f"[MISSING] {name:16s} {path}")

    add_text_page(pdf, "KH_shoot review bundle - file status", status)

    final23 = read_csv(FILES["FINAL23"])
    if final23 is not None:
        cols = [
            "Mach", "alpha", "reference_cr", "reference_ci", "reference_omega_i",
            "best_status", "trusted_spectral", "trusted_modal",
            "valid_spectral_candidate", "valid_modal_candidate", "source_group"
        ]
        add_df_pages(pdf, "FINAL23 curated reference", final23, cols=cols, max_rows_per_page=23, fontsize=7)

    allcand = read_csv(FILES["candidates_all"])
    if allcand is not None:
        summary = []
        if "decision" in allcand.columns:
            summary += ["Decision counts:"]
            summary += allcand["decision"].value_counts(dropna=False).to_string().splitlines()
        if "source_kind" in allcand.columns:
            summary += ["", "Source kind counts:"]
            summary += allcand["source_kind"].value_counts(dropna=False).head(30).to_string().splitlines()
        add_text_page(pdf, "Candidate comparison summary", summary)

        cols = [
            "decision", "source_kind", "Mach", "alpha", "cr", "ci",
            "reference_cr", "reference_ci", "cr_abs_err", "ci_abs_err", "ci_rel_err", "source"
        ]
        exact_near = allcand[allcand.get("decision", "").isin(["canonical_FINAL23", "exact_FINAL23", "near_FINAL23"])] if "decision" in allcand.columns else allcand.head(0)
        add_df_pages(pdf, "Canonical / exact / near FINAL23 matches", exact_near, cols=cols, max_rows_per_page=20, fontsize=5)

        ext = allcand[allcand.get("decision", "").isin(["extension_or_unmatched"])] if "decision" in allcand.columns else allcand.head(0)
        add_df_pages(pdf, "Extensions / unmatched points to review", ext.head(80), cols=cols, max_rows_per_page=20, fontsize=5)

        rej = allcand[allcand.get("decision", "").isin(["reject_off_FINAL23_branch", "ignore_polluted_aggregate"])] if "decision" in allcand.columns else allcand.head(0)
        add_df_pages(pdf, "Rejected / polluted examples", rej.head(80), cols=cols, max_rows_per_page=20, fontsize=5)

    keep = read_csv(FILES["keep_or_review"])
    if keep is not None:
        cols = [
            "decision", "source_kind", "Mach", "alpha", "cr", "ci",
            "reference_cr", "reference_ci", "cr_abs_err", "ci_abs_err", "source"
        ]
        add_df_pages(pdf, "Keep or review CSV", keep.head(120), cols=cols, max_rows_per_page=20, fontsize=5)

    add_text_page(pdf, "FINAL23 curation log", read_lines(FILES["final23_log"], max_lines=140), fontsize=7)
    add_text_page(pdf, "Candidate comparison log", read_lines(FILES["candidate_log"], max_lines=140), fontsize=7)
    add_text_page(pdf, "KH_shoot log errors excerpt", read_lines(FILES["errors"], max_lines=140), fontsize=7)

print("[OK] wrote", PDF)
