#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

OUT = Path("assets/classic_supersonic/KH_shoot_prof_bundle")
OUT.mkdir(parents=True, exist_ok=True)

PDF = OUT / "KH_shoot_review_summary_PROF.pdf"

FINAL = OUT / "KH_shoot_FINAL23_points_complete.csv"
ALL = OUT / "KH_shoot_ALL_points_scanned_complete.csv"
KEEP = OUT / "KH_shoot_points_keep_or_review.csv"
REJECT = OUT / "KH_shoot_points_reject_or_ignore.csv"
CONS = OUT / "KH_shoot_FINAL23_consistency_check.csv"
SKIP = OUT / "KH_shoot_scan_skipped_files.csv"

def read_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)

def fmt_df(df, cols=None, max_rows=40, width=180):
    if df is None:
        return "MISSING FILE"
    if df.empty:
        return "EMPTY DATAFRAME"
    d = df.copy()
    if cols is not None:
        cols = [c for c in cols if c in d.columns]
        d = d[cols]
    d = d.head(max_rows).copy()
    for c in d.columns:
        if d[c].dtype.kind in "fc":
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            d[c] = d[c].astype(str).map(lambda x: x[:55])
    return d.to_string(index=False, max_cols=20, line_width=width)

def add_page(pdf, title, text, fontsize=7.2):
    lines = []
    for line in str(text).splitlines():
        if len(line) <= 170:
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, width=170))
    if not lines:
        lines = [""]

    per_page = 43
    for start in range(0, len(lines), per_page):
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(111)
        ax.axis("off")
        suffix = "" if start == 0 else f" continued {start//per_page + 1}"
        ax.text(0.02, 0.97, title + suffix, fontsize=15, weight="bold", va="top")
        y = 0.91
        for line in lines[start:start+per_page]:
            ax.text(0.02, y, line, fontsize=fontsize, family="monospace", va="top")
            y -= 0.021
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

final = read_csv(FINAL)
allpts = read_csv(ALL)
keep = read_csv(KEEP)
reject = read_csv(REJECT)
cons = read_csv(CONS)
skip = read_csv(SKIP)

with PdfPages(PDF) as pdf:
    status = []
    for p in [FINAL, ALL, KEEP, REJECT, CONS, SKIP]:
        if p.exists():
            status.append(f"[OK] {p.name:45s} {p.stat().st_size/1024:.1f} KiB")
        else:
            status.append(f"[MISSING] {p.name}")

    add_page(
        pdf,
        "KH_shoot professor bundle - file status",
        "\n".join(status),
        fontsize=8,
    )

    if final is not None:
        txt = f"FINAL23 shape: {final.shape}\n\n"
        txt += fmt_df(
            final,
            ["Mach", "alpha", "cr", "ci", "omega_i", "best_status", "trusted_spectral", "trusted_modal"],
            max_rows=30,
        )
        add_page(pdf, "FINAL23 complete reference points", txt, fontsize=7.2)

    if cons is not None:
        txt = f"Consistency shape: {cons.shape}\n\n"
        txt += fmt_df(cons, max_rows=20)
        add_page(pdf, "FINAL23 consistency check", txt, fontsize=7.2)

    if allpts is not None:
        txt = f"All scanned points shape: {allpts.shape}\n\n"
        if "decision" in allpts.columns:
            txt += "Decision counts:\n"
            txt += allpts["decision"].value_counts(dropna=False).to_string()
            txt += "\n\n"
        if "source_kind" in allpts.columns:
            txt += "Source kind counts:\n"
            txt += allpts["source_kind"].value_counts(dropna=False).to_string()
        add_page(pdf, "All scanned points - summary counts", txt, fontsize=8)

    if keep is not None:
        txt = f"Keep/review shape: {keep.shape}\n\n"
        txt += fmt_df(
            keep,
            ["decision", "source_kind", "Mach", "alpha", "cr", "ci", "reference_cr", "reference_ci", "cr_abs_err", "ci_abs_err", "source_short"],
            max_rows=60,
        )
        add_page(pdf, "Points to keep or review - first 60 rows", txt, fontsize=5.8)

    if reject is not None:
        txt = f"Reject/ignore shape: {reject.shape}\n\n"
        txt += fmt_df(
            reject,
            ["decision", "source_kind", "Mach", "alpha", "cr", "ci", "reference_cr", "reference_ci", "cr_abs_err", "ci_abs_err", "source_short"],
            max_rows=60,
        )
        add_page(pdf, "Points rejected or ignored - first 60 rows", txt, fontsize=5.8)

README = OUT / "README_PROF_BUNDLE.txt"
README.write_text(
    "KH_shoot professor bundle\n"
    "========================\n\n"
    "Main files:\n"
    "- KH_shoot_review_summary_PROF.pdf\n"
    "- KH_shoot_FINAL23_points_complete.csv\n"
    "- KH_shoot_ALL_points_scanned_complete.csv\n"
    "- KH_shoot_points_keep_or_review.csv\n"
    "- KH_shoot_points_reject_or_ignore.csv\n"
    "- KH_shoot_FINAL23_consistency_check.csv\n\n"
    "Use KH_shoot_FINAL23_points_complete.csv as the clean validated 23-point reference.\n"
    "Use KH_shoot_points_keep_or_review.csv for possible extensions or duplicates to inspect.\n"
    "Use KH_shoot_points_reject_or_ignore.csv for off-branch / bad / ignored points.\n"
)

MANIFEST = OUT / "MANIFEST.txt"
MANIFEST.write_text("\n".join(sorted(p.name for p in OUT.iterdir() if p.is_file())) + "\n")

print("[OK] wrote", PDF)
print("[OK] wrote", README)
print("[OK] wrote", MANIFEST)
