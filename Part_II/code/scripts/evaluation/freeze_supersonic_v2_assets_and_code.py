#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


RAW = Path("assets/classic_supersonic/final_sparse_PINN_reference_v2_CONFIRMED")
POL = Path("assets/classic_supersonic/final_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1")

OUT = Path("assets/classic_supersonic/supersonic_sparse_PINN_reference_v2_FROZEN")
ASSETS = OUT / "assets"
CODE = OUT / "code_snapshot"
REPORTS = OUT / "reports"
DATA = OUT / "data"

for d in [OUT, ASSETS, CODE, REPORTS, DATA]:
    d.mkdir(parents=True, exist_ok=True)

RAW_SPEC = RAW / "supersonic_sparse_PINN_reference_v2_CONFIRMED_spectral.csv"
RAW_FIELDS = RAW / "supersonic_sparse_PINN_reference_v2_CONFIRMED_modal_fields.csv"

POL_SPEC = POL / "supersonic_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1_spectral.csv"
POL_FIELDS = POL / "supersonic_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1_modal_fields.csv"

OVERLAY = RAW / "blumen_ci_overlay_sparse_PINN_reference_v2.png"

SQUARE_PDF = REPORTS / "supersonic_sparse_PINN_reference_v2_FROZEN_modes_square.pdf"
TAIL_SQUARE_PDF = REPORTS / "supersonic_sparse_PINN_reference_v2_FROZEN_tail_polished_square_review.pdf"

SPEC_OUT = DATA / "supersonic_sparse_PINN_reference_v2_FROZEN_spectral.csv"
FIELDS_RAW_OUT = DATA / "supersonic_sparse_PINN_reference_v2_FROZEN_modal_fields_raw_confirmed.csv"
FIELDS_POL_OUT = DATA / "supersonic_sparse_PINN_reference_v2_FROZEN_modal_fields_tail_polished_v1.csv"

MANIFEST = OUT / "manifest.json"
SHA = OUT / "SHA256SUMS.txt"
README = OUT / "README.md"


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "Mach" not in df.columns and "M" in df.columns:
        df = df.rename(columns={"M": "Mach"})
    df["Mach"] = pd.to_numeric(df["Mach"], errors="coerce")
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    return df


def zfield(df: pd.DataFrame, name: str) -> np.ndarray:
    return (
        pd.to_numeric(df[f"{name}_real"], errors="coerce").to_numpy(float)
        + 1j * pd.to_numeric(df[f"{name}_imag"], errors="coerce").to_numpy(float)
    )


def scale(z: np.ndarray) -> float:
    s = float(np.nanmax(np.abs(z))) if len(z) else 1.0
    return s if np.isfinite(s) and s > 0 else 1.0


def match_point(df: pd.DataFrame, M: float, a: float) -> pd.DataFrame:
    return df[
        np.isclose(df["Mach"].astype(float), M, atol=1e-10)
        & np.isclose(df["alpha"].astype(float), a, atol=1e-10)
    ].copy()


def xlim_active(y: np.ndarray, amp: np.ndarray, frac: float = 0.01, min_half: float = 80.0):
    peak = float(np.nanmax(amp)) if len(amp) else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return float(np.nanmin(y)), float(np.nanmax(y))
    mask = amp >= frac * peak
    if not np.any(mask):
        return float(np.nanmin(y)), float(np.nanmax(y))
    half = max(float(np.nanmax(np.abs(y[mask]))), min_half)
    return -half, half


def xlim_left_tail(y: np.ndarray, amp: np.ndarray, frac: float = 0.003, min_width: float = 120.0):
    peak = float(np.nanmax(amp)) if len(amp) else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return float(np.nanmin(y)), 0.0
    mask = (y < 0.0) & (amp >= frac * peak)
    if not np.any(mask):
        mask = y < 0.0
    if not np.any(mask):
        return float(np.nanmin(y)), float(np.nanmax(y))
    left = float(np.nanmin(y[mask]))
    right = 0.0
    if right - left < min_width:
        left = right - min_width
    return left, right


def short_status(s):
    s = str(s)
    if "boundary_flag" in s:
        return "boundary flag"
    if "tail_sensitive" in s:
        return "tail-sensitive"
    if "smallM" in s:
        return "small-M strict"
    if "legacy" in s or "modal_spectral" in s:
        return "legacy/base"
    return s[:60]


def add_summary_page(pdf: PdfPages, spec: pd.DataFrame, raw_fields: pd.DataFrame, pol_fields: pd.DataFrame | None):
    fig = plt.figure(figsize=(11.69, 8.27), dpi=160)
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = [
        "Supersonic sparse PINN reference v2 - FROZEN",
        "",
        f"Spectral points: {spec[['Mach', 'alpha']].drop_duplicates().shape[0]}",
        f"Raw modal rows: {len(raw_fields)}",
        f"Tail-polished rows: {len(pol_fields) if pol_fields is not None else 'not available'}",
        "",
        "Counts by Mach:",
    ]

    for M, n in spec.groupby("Mach").size().to_dict().items():
        lines.append(f"  M={float(M):.2f}: {int(n)}")

    lines += [
        "",
        "Validation statuses:",
    ]

    for k, v in spec["validation_status"].astype(str).value_counts(dropna=False).to_dict().items():
        lines.append(f"  {k}: {v}")

    lines += [
        "",
        "Frozen convention:",
        "  - Raw confirmed fields are the primary reference.",
        "  - Tail-polished fields are an export derivative for smoother weak left tails.",
        "  - Spectral cr/ci are unchanged by tail polishing.",
        "  - Boundary-flagged points are kept with explicit status.",
    ]

    ax.text(0.04, 0.96, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_square_modes_pdf(spec, fields, pdf_path: Path, *, title_prefix: str):
    with PdfPages(pdf_path) as pdf:
        add_summary_page(pdf, spec, fields, None)

        for _, r in spec.sort_values(["Mach", "alpha"]).iterrows():
            M = float(r["Mach"])
            a = float(r["alpha"])
            sub = match_point(fields, M, a)

            if sub.empty:
                continue

            sub = sub.sort_values("y")
            y = pd.to_numeric(sub["y"], errors="coerce").to_numpy(float)

            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), dpi=170)
            axes = axes.ravel()

            for ax, name in zip(axes, ["p", "rho", "v", "u"]):
                z = zfield(sub, name)
                amp = np.abs(z)
                sc = scale(z)

                ax.plot(y, np.real(z) / sc, linewidth=0.75, label=f"Re({name})")
                ax.plot(y, amp / sc, "--", linewidth=0.75, label=f"|{name}|")
                ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)

                ax.set_xlim(*xlim_active(y, amp, frac=0.01, min_half=80.0))
                ax.set_ylim(-1.05, 1.08)
                ax.set_box_aspect(1)
                ax.grid(True, alpha=0.25, linestyle=":")
                ax.set_title(name)
                ax.set_xlabel("y")
                ax.legend(fontsize=7)

            fig.suptitle(
                f"{title_prefix} — M={M:.2f}, alpha={a:.5f}, "
                f"cr={float(r.get('cr', np.nan)):.6g}, ci={float(r.get('ci', np.nan)):.6g}\n"
                f"{short_status(r.get('validation_status', ''))}",
                fontsize=10,
            )

            fig.tight_layout(rect=[0, 0.02, 1, 0.92])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def make_tail_polish_square_review(spec, raw_fields, pol_fields, pdf_path: Path):
    with PdfPages(pdf_path) as pdf:
        add_summary_page(pdf, spec, raw_fields, pol_fields)

        for _, r in spec.sort_values(["Mach", "alpha"]).iterrows():
            M = float(r["Mach"])
            a = float(r["alpha"])

            raw = match_point(raw_fields, M, a).sort_values("y")
            pol = match_point(pol_fields, M, a).sort_values("y")

            if raw.empty or pol.empty:
                continue

            y = pd.to_numeric(raw["y"], errors="coerce").to_numpy(float)

            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), dpi=170)
            axes = axes.ravel()

            for ax, name in zip(axes, ["p", "rho", "v", "u"]):
                zr = zfield(raw, name)
                zp = zfield(pol, name)
                sc = scale(zr)
                amp = np.abs(zr)

                ax.plot(y, np.real(zr) / sc, linewidth=0.6, label=f"raw Re({name})")
                ax.plot(y, np.real(zp) / sc, "--", linewidth=0.75, label=f"polished Re({name})")
                ax.plot(y, np.abs(zp) / sc, ":", linewidth=0.75, label=f"polished |{name}|")

                ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
                ax.set_xlim(*xlim_left_tail(y, amp, frac=0.003, min_width=120.0))
                ax.set_ylim(-1.05, 1.08)
                ax.set_box_aspect(1)
                ax.grid(True, alpha=0.25, linestyle=":")
                ax.set_title(f"{name} left tail")
                ax.set_xlabel("y")
                ax.legend(fontsize=6)

            fig.suptitle(
                f"TAIL POLISH REVIEW — M={M:.2f}, alpha={a:.5f}, "
                f"cr={float(r.get('cr', np.nan)):.6g}, ci={float(r.get('ci', np.nan)):.6g}\n"
                f"{short_status(r.get('validation_status', ''))}",
                fontsize=10,
            )

            fig.tight_layout(rect=[0, 0.02, 1, 0.92])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def copy_if_exists(src: Path, dst_dir: Path):
    if src.exists():
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return str(dst)
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def copy_code_snapshot():
    files = [
        "code/src/classical_solver/supersonic/mstab17_supersonic_solver.py",
        "code/src/classical_solver/supersonic/blumen_reference.py",

        "code/scripts/audits/audit_supersonic_shooting_visual_validation_6969b4f1bf.py",
        "code/scripts/audits/audit_scan_supersonic_M18_M19_strict_modal_validation_14e1027f3b.py",
        "code/scripts/shooting/solve_refine_M18_M19_near_valid_branch_7cb1f8ed3e.py",
        "code/scripts/audits/audit_M18_M19_refined_convergence_0e0240282c.py",
        "code/scripts/audits/audit_M18_M19_max_step_convergence_c6e9b8c068.py",
        "code/scripts/audits/audit_M18_M19_mapping_vs_box_effect.py",

        "code/scripts/data_preparation/prepare_build_supersonic_fixed_ci_shooting_anchors_45523d471e.py",
        "code/scripts/data_preparation/prepare_build_supersonic_fixed_ci_shooting_extension_final.py",
        "scripts/audit_gep_nearest_to_fixed_ci_shooting_anchors.py",
        "code/scripts/audits/audit_supersonic_ci_first_shooting_mismatch_map_1a94c44bbe.py",

        "code/scripts/data_preparation/prepare_build_final_sparse_supersonic_reference_with_M18_M19.py",
        "code/scripts/audits/audit_scan_supersonic_target_campaign_3895a18fd6.py",
        "code/scripts/data_preparation/prepare_build_final_sparse_reference_v2_with_smallM_campaign.py",
        "code/scripts/evaluation/freeze_supersonic_sparse_PINN_reference_v2.py",
        "code/scripts/evaluation/polish_supersonic_v2_left_tails.py",
        "code/scripts/evaluation/freeze_supersonic_v2_assets_and_code.py",
    ]

    copied = []

    for rel in files:
        src = Path(rel)
        if src.exists():
            dst = CODE / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)

    # slurm/config snapshots if present
    for pattern in [
        "slurm/jz_*supersonic*.slurm",
        "slurm/jz_*M18*.slurm",
        "slurm/jz_*smallM*.slurm",
        "slurm/jz_scan_smallM_sparse_campaign.slurm",
        "slurm/jz_audit_M18_M19_refined_convergence.slurm",
    ]:
        for src in Path(".").glob(pattern):
            if src.exists():
                dst = CODE / src
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied.append(str(src))

    return copied


# ------------------------------------------------------------------
# Load data.
# ------------------------------------------------------------------
if not RAW_SPEC.exists():
    raise SystemExit(f"Missing {RAW_SPEC}")
if not RAW_FIELDS.exists():
    raise SystemExit(f"Missing {RAW_FIELDS}")

spec = norm_cols(pd.read_csv(RAW_SPEC))
raw_fields = norm_cols(pd.read_csv(RAW_FIELDS, low_memory=False))

spec = spec.sort_values(["Mach", "alpha"]).drop_duplicates(["Mach", "alpha"], keep="first").reset_index(drop=True)
raw_fields["y"] = pd.to_numeric(raw_fields["y"], errors="coerce")
raw_fields = raw_fields.dropna(subset=["Mach", "alpha", "y"]).sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)

if POL_FIELDS.exists():
    pol_fields = norm_cols(pd.read_csv(POL_FIELDS, low_memory=False))
    pol_fields["y"] = pd.to_numeric(pol_fields["y"], errors="coerce")
    pol_fields = pol_fields.dropna(subset=["Mach", "alpha", "y"]).sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)
else:
    pol_fields = raw_fields.copy()

if POL_SPEC.exists():
    spec_pol = norm_cols(pd.read_csv(POL_SPEC))
else:
    spec_pol = spec.copy()

# ------------------------------------------------------------------
# Freeze canonical CSVs.
# ------------------------------------------------------------------
spec_pol.to_csv(SPEC_OUT, index=False)
raw_fields.to_csv(FIELDS_RAW_OUT, index=False)
pol_fields.to_csv(FIELDS_POL_OUT, index=False)

# ------------------------------------------------------------------
# Copy maps/reports.
# ------------------------------------------------------------------
copied_assets = []
for src in [
    OVERLAY,
    RAW / "blumen_ci_overlay_sparse_PINN_reference_v2.pdf",
    RAW / "supersonic_sparse_PINN_reference_v2_CONFIRMED_modes_overview.pdf",
    RAW / "supersonic_sparse_PINN_reference_v2_CONFIRMED_modes_core_tail.pdf",
    POL / "tail_polish_raw_vs_polished_left_tail_review.pdf",
    RAW / "summary_freeze.json",
    RAW / "README_FREEZE.md",
    RAW / "SHA256SUMS.txt",
    POL / "summary_tail_polished_v1.json",
    POL / "tail_polish_audit.csv",
]:
    dst = copy_if_exists(src, ASSETS if src.suffix.lower() in [".png", ".pdf"] else REPORTS)
    if dst:
        copied_assets.append(dst)

# ------------------------------------------------------------------
# New square PDFs.
# ------------------------------------------------------------------
make_square_modes_pdf(
    spec_pol,
    pol_fields,
    SQUARE_PDF,
    title_prefix="FROZEN v2 tail-polished export",
)

make_tail_polish_square_review(
    spec,
    raw_fields,
    pol_fields,
    TAIL_SQUARE_PDF,
)

# ------------------------------------------------------------------
# Code/config snapshot.
# ------------------------------------------------------------------
copied_code = copy_code_snapshot()

git_info = {
    "git_rev_parse_HEAD": run(["git", "rev-parse", "HEAD"]),
    "git_branch": run(["git", "branch", "--show-current"]),
    "git_status_short": run(["git", "status", "--short"]),
    "git_diff_stat": run(["git", "diff", "--stat"]),
}

(CODE / "git_info.json").write_text(json.dumps(git_info, indent=2))

summary = {
    "status": "FROZEN_ASSETS_AND_CODE_BUILT",
    "dataset": "supersonic_sparse_PINN_reference_v2_FROZEN",
    "n_spectral_points": int(spec_pol[["Mach", "alpha"]].drop_duplicates().shape[0]),
    "n_raw_modal_rows": int(len(raw_fields)),
    "n_tail_polished_modal_rows": int(len(pol_fields)),
    "point_counts_by_Mach": spec_pol.groupby("Mach").size().to_dict(),
    "validation_status_counts": spec_pol["validation_status"].astype(str).value_counts(dropna=False).to_dict(),
    "outputs": {
        "spectral": str(SPEC_OUT),
        "raw_fields": str(FIELDS_RAW_OUT),
        "tail_polished_fields": str(FIELDS_POL_OUT),
        "square_modes_pdf": str(SQUARE_PDF),
        "tail_square_review_pdf": str(TAIL_SQUARE_PDF),
        "assets_dir": str(ASSETS),
        "reports_dir": str(REPORTS),
        "code_snapshot_dir": str(CODE),
        "manifest": str(MANIFEST),
        "sha256": str(SHA),
    },
    "copied_assets": copied_assets,
    "copied_code_files": copied_code,
    "git_info": git_info,
    "scientific_note": (
        "Raw confirmed fields remain the primary frozen reference. "
        "Tail-polished fields are a documented export derivative with unchanged spectral cr/ci. "
        "Square PDFs are generated for final visual inspection."
    ),
}

MANIFEST.write_text(json.dumps(summary, indent=2))

README.write_text(
f"""# Supersonic sparse PINN reference v2 - FROZEN

This folder freezes the final sparse classical supersonic reference for PINN experiments.

## Main data

- `{SPEC_OUT.relative_to(OUT)}`: spectral table.
- `{FIELDS_RAW_OUT.relative_to(OUT)}`: raw confirmed modal fields.
- `{FIELDS_POL_OUT.relative_to(OUT)}`: tail-polished export modal fields.

## Main reports

- `{SQUARE_PDF.relative_to(OUT)}`: square-panel PDF for all final modes.
- `{TAIL_SQUARE_PDF.relative_to(OUT)}`: square-panel raw vs tail-polished left-tail review.
- `assets/blumen_ci_overlay_sparse_PINN_reference_v2.png`: final Blumen coverage map.

## Code snapshot

The `code_snapshot/` folder contains the solver and campaign/freeze scripts used to generate this reference.

## Status

Raw confirmed fields are the primary reference.  
Tail-polished fields are an export derivative for smoother weak tails; spectral values are unchanged.

Spectral points: {summary["n_spectral_points"]}  
Raw modal rows: {summary["n_raw_modal_rows"]}  
Tail-polished modal rows: {summary["n_tail_polished_modal_rows"]}
"""
)

# checksums
all_files = sorted([p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"])
with SHA.open("w") as f:
    for p in all_files:
        f.write(f"{sha256_file(p)}  {p.relative_to(OUT)}\n")

print(json.dumps(summary, indent=2))
print("\nWrote:", OUT)
