#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


SRC = Path("assets/classic_supersonic/final_sparse_PINN_reference_v2_smallM_M18M19")
OUT = Path("assets/classic_supersonic/final_sparse_PINN_reference_v2_CONFIRMED")

SPEC_IN = SRC / "supersonic_sparse_PINN_reference_v2_spectral.csv"
FIELDS_IN = SRC / "supersonic_sparse_PINN_reference_v2_modal_fields.csv"
OVERLAY_IN = SRC / "blumen_ci_overlay_sparse_PINN_reference_v2.png"
COVERAGE_IN = SRC / "coverage_by_Mach_v2.csv"
SUGGESTED_IN = SRC / "suggested_remaining_targets_v2.csv"
SUMMARY_IN = SRC / "summary.json"

OUT.mkdir(parents=True, exist_ok=True)

SPEC_OUT = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_spectral.csv"
FIELDS_OUT = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_modal_fields.csv"
POINT_AUDIT_OUT = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_point_audit.csv"

PDF_OVERVIEW = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_modes_overview.pdf"
PDF_CORE_TAIL = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_modes_core_tail.pdf"

README_OUT = OUT / "README_FREEZE.md"
SUMMARY_OUT = OUT / "summary_freeze.json"
SHA_OUT = OUT / "SHA256SUMS.txt"


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "Mach" not in df.columns and "M" in df.columns:
        df = df.rename(columns={"M": "Mach"})
    df["Mach"] = pd.to_numeric(df["Mach"], errors="coerce")
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    return df


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing any of columns: {candidates}")


FIELD_COLS = {
    "p": (["p_real", "p_re", "Re_p"], ["p_imag", "p_im", "Im_p"]),
    "rho": (["rho_real", "rho_re", "Re_rho"], ["rho_imag", "rho_im", "Im_rho"]),
    "u": (["u_real", "u_re", "Re_u"], ["u_imag", "u_im", "Im_u"]),
    "v": (["v_real", "v_re", "Re_v"], ["v_imag", "v_im", "Im_v"]),
}


def complex_field(df: pd.DataFrame, name: str) -> np.ndarray:
    re_candidates, im_candidates = FIELD_COLS[name]
    re_col = find_col(df, re_candidates)
    im_col = find_col(df, im_candidates)
    re = pd.to_numeric(df[re_col], errors="coerce").to_numpy(float)
    im = pd.to_numeric(df[im_col], errors="coerce").to_numpy(float)
    return re + 1j * im


def safe_scale(z: np.ndarray) -> float:
    s = float(np.nanmax(np.abs(z))) if len(z) else 1.0
    if not np.isfinite(s) or s <= 0:
        return 1.0
    return s


def center_xlim(y: np.ndarray, amp: np.ndarray, threshold: float = 0.05, min_half: float = 60.0):
    peak = float(np.nanmax(amp)) if len(amp) else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return float(np.nanmin(y)), float(np.nanmax(y))

    mask = amp >= threshold * peak
    if not np.any(mask):
        return float(np.nanmin(y)), float(np.nanmax(y))

    half = max(float(np.nanmax(np.abs(y[mask]))), min_half)
    return -half, half


def tail_xlim(y: np.ndarray, amp: np.ndarray, threshold: float = 0.005, min_width: float = 120.0):
    peak = float(np.nanmax(amp)) if len(amp) else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return float(np.nanmin(y)), float(np.nanmax(y))

    mask = (y < 0.0) & (amp >= threshold * peak)
    if not np.any(mask):
        return center_xlim(y, amp, threshold=0.02, min_half=80.0)

    left = float(np.nanmin(y[mask]))
    right = 0.0

    if right - left < min_width:
        left = right - min_width

    return left, right


def point_match(df: pd.DataFrame, M: float, a: float) -> pd.DataFrame:
    return df[
        np.isclose(df["Mach"].astype(float), M, atol=1e-10)
        & np.isclose(df["alpha"].astype(float), a, atol=1e-10)
    ].copy()


def short_status(s: str) -> str:
    s = str(s)
    if "smallM" in s and "boundary_flag" in s:
        return "smallM boundary-flag"
    if "smallM" in s and "tail_sensitive" in s:
        return "smallM tail-sensitive"
    if "smallM" in s:
        return "smallM strict"
    if "tail_sensitive" in s or "M18_M19" in s or "core_stable" in s:
        return "core-stable tail-sensitive"
    if "legacy" in s:
        return "legacy"
    return s[:60]


def add_summary_page(pdf: PdfPages, spec: pd.DataFrame, fields: pd.DataFrame, point_audit: pd.DataFrame):
    fig = plt.figure(figsize=(11.69, 8.27), dpi=160)
    ax = fig.add_subplot(111)
    ax.axis("off")

    counts_by_M = spec.groupby("Mach").size().to_dict()
    status_counts = spec["validation_status"].astype(str).value_counts(dropna=False).to_dict()

    lines = [
        "Supersonic sparse PINN reference v2 - CONFIRMED",
        "",
        f"Spectral points: {spec[['Mach', 'alpha']].drop_duplicates().shape[0]}",
        f"Modal rows: {len(fields)}",
        "",
        "Counts by Mach:",
    ]

    for M, n in counts_by_M.items():
        lines.append(f"  M={float(M):.2f}: {int(n)}")

    lines += [
        "",
        "Validation-status counts:",
    ]

    for k, v in status_counts.items():
        lines.append(f"  {k}: {v}")

    lines += [
        "",
        "Interpretation:",
        "  - legacy/base: previously accepted reference points.",
        "  - small-M campaign: visually accepted new sparse coverage points.",
        "  - boundary flag: point hit cr=0 or ci=0.12 in the first campaign scan; kept with explicit flag.",
        "  - M=1.8/1.9: core stable; weak oscillatory tails remain mapping-sensitive.",
        "",
        "This PDF shows Re(field)/max|field| and |field|/max|field|.",
        "The core-tail PDF shows both the central active region and the left oscillatory tail.",
    ]

    ax.text(
        0.04, 0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_overview_pdf(spec: pd.DataFrame, fields: pd.DataFrame, pdf_path: Path, point_audit: pd.DataFrame):
    with PdfPages(pdf_path) as pdf:
        add_summary_page(pdf, spec, fields, point_audit)

        for _, r in spec.sort_values(["Mach", "alpha"]).iterrows():
            M = float(r["Mach"])
            a = float(r["alpha"])

            sub = point_match(fields, M, a)
            if sub.empty:
                fig = plt.figure(figsize=(11.69, 8.27), dpi=160)
                fig.text(0.05, 0.95, f"MISSING FIELDS: M={M:.2f}, alpha={a:.5f}", va="top")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                continue

            sub = sub.sort_values("y")
            y = pd.to_numeric(sub["y"], errors="coerce").to_numpy(float)

            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), dpi=170)
            axes = axes.ravel()

            for ax, name in zip(axes, ["p", "rho", "u", "v"]):
                z = complex_field(sub, name)
                amp = np.abs(z)
                scale = safe_scale(z)

                ax.plot(y, np.real(z) / scale, linewidth=0.75, label=f"Re({name})")
                ax.plot(y, amp / scale, "--", linewidth=0.75, label=f"|{name}|")
                ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
                ax.set_xlim(*center_xlim(y, amp, threshold=0.01, min_half=80.0))
                ax.set_ylim(-1.05, 1.08)
                ax.grid(True, alpha=0.25, linestyle=":")
                ax.set_title(name)
                ax.set_xlabel("y")
                ax.legend(fontsize=7)

            fig.suptitle(
                f"M={M:.2f}, alpha={a:.5f}, "
                f"cr={float(r.get('cr', np.nan)):.6g}, ci={float(r.get('ci', np.nan)):.6g}\n"
                f"{short_status(r.get('validation_status', ''))}",
                fontsize=10,
            )
            fig.tight_layout(rect=[0, 0.02, 1, 0.92])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def make_core_tail_pdf(spec: pd.DataFrame, fields: pd.DataFrame, pdf_path: Path, point_audit: pd.DataFrame):
    with PdfPages(pdf_path) as pdf:
        add_summary_page(pdf, spec, fields, point_audit)

        for _, r in spec.sort_values(["Mach", "alpha"]).iterrows():
            M = float(r["Mach"])
            a = float(r["alpha"])

            sub = point_match(fields, M, a)
            if sub.empty:
                fig = plt.figure(figsize=(11.69, 8.27), dpi=160)
                fig.text(0.05, 0.95, f"MISSING FIELDS: M={M:.2f}, alpha={a:.5f}", va="top")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                continue

            sub = sub.sort_values("y")
            y = pd.to_numeric(sub["y"], errors="coerce").to_numpy(float)

            fig, axes = plt.subplots(4, 2, figsize=(11.69, 8.27), dpi=170)
            fields_order = ["p", "rho", "u", "v"]

            for row, name in enumerate(fields_order):
                z = complex_field(sub, name)
                amp = np.abs(z)
                scale = safe_scale(z)

                for col, region in enumerate(["core", "left tail"]):
                    ax = axes[row, col]

                    ax.plot(y, np.real(z) / scale, linewidth=0.65, label=f"Re({name})")
                    ax.plot(y, amp / scale, "--", linewidth=0.65, label=f"|{name}|")
                    ax.axhline(0.0, color="black", linewidth=0.45, alpha=0.4)
                    ax.set_ylim(-1.05, 1.08)
                    ax.grid(True, alpha=0.22, linestyle=":")

                    if region == "core":
                        ax.set_xlim(*center_xlim(y, amp, threshold=0.05, min_half=60.0))
                    else:
                        ax.set_xlim(*tail_xlim(y, amp, threshold=0.005, min_width=120.0))

                    ax.set_title(f"{name} - {region}", fontsize=8)
                    ax.set_xlabel("y", fontsize=8)

                    if row == 0 and col == 1:
                        ax.legend(fontsize=6, loc="upper left")

            fig.suptitle(
                f"CORE + TAIL CHECK - M={M:.2f}, alpha={a:.5f}, "
                f"cr={float(r.get('cr', np.nan)):.6g}, ci={float(r.get('ci', np.nan)):.6g}\n"
                f"{short_status(r.get('validation_status', ''))}",
                fontsize=9,
            )

            fig.tight_layout(rect=[0, 0.02, 1, 0.91])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if not SPEC_IN.exists():
    raise SystemExit(f"Missing {SPEC_IN}")
if not FIELDS_IN.exists():
    raise SystemExit(f"Missing {FIELDS_IN}")

spec = norm_cols(pd.read_csv(SPEC_IN))
fields = norm_cols(pd.read_csv(FIELDS_IN, low_memory=False))
fields["y"] = pd.to_numeric(fields["y"], errors="coerce")
fields = fields.dropna(subset=["Mach", "alpha", "y"])

# canonical sort
spec = spec.sort_values(["Mach", "alpha"]).drop_duplicates(["Mach", "alpha"], keep="first").reset_index(drop=True)
fields = fields.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)

# audit per point
audit_rows = []
for _, r in spec.iterrows():
    M = float(r["Mach"])
    a = float(r["alpha"])
    sub = point_match(fields, M, a)

    row = {
        "Mach": M,
        "alpha": a,
        "n_field_rows": int(len(sub)),
        "has_fields": bool(len(sub) > 0),
        "validation_status": r.get("validation_status", np.nan),
        "reference_role": r.get("reference_role", np.nan),
        "cr": r.get("cr", np.nan),
        "ci": r.get("ci", np.nan),
        "omega_i": r.get("omega_i", np.nan),
    }

    if len(sub):
        y = pd.to_numeric(sub["y"], errors="coerce").to_numpy(float)
        row["y_min"] = float(np.nanmin(y))
        row["y_max"] = float(np.nanmax(y))

        for name in ["p", "rho", "u", "v"]:
            try:
                z = complex_field(sub, name)
                row[f"max_abs_{name}"] = float(np.nanmax(np.abs(z)))
            except Exception:
                row[f"max_abs_{name}"] = np.nan

    audit_rows.append(row)

point_audit = pd.DataFrame(audit_rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)

# copy canonical data
spec.to_csv(SPEC_OUT, index=False)
fields.to_csv(FIELDS_OUT, index=False)
point_audit.to_csv(POINT_AUDIT_OUT, index=False)

# copy existing useful assets
for p in [OVERLAY_IN, COVERAGE_IN, SUGGESTED_IN, SUMMARY_IN]:
    if p.exists():
        shutil.copy2(p, OUT / p.name)

# PDFs
make_overview_pdf(spec, fields, PDF_OVERVIEW, point_audit)
make_core_tail_pdf(spec, fields, PDF_CORE_TAIL, point_audit)

summary = {
    "status": "CONFIRMED_FREEZE",
    "dataset": "supersonic_sparse_PINN_reference_v2_CONFIRMED",
    "n_spectral_points": int(spec[["Mach", "alpha"]].drop_duplicates().shape[0]),
    "n_modal_rows": int(len(fields)),
    "n_points_with_fields": int(point_audit["has_fields"].sum()),
    "n_points_missing_fields": int((~point_audit["has_fields"]).sum()),
    "point_counts_by_Mach": spec.groupby("Mach").size().to_dict(),
    "validation_status_counts": spec["validation_status"].astype(str).value_counts(dropna=False).to_dict(),
    "outputs": {
        "spectral": str(SPEC_OUT),
        "modal_fields": str(FIELDS_OUT),
        "point_audit": str(POINT_AUDIT_OUT),
        "modes_overview_pdf": str(PDF_OVERVIEW),
        "modes_core_tail_pdf": str(PDF_CORE_TAIL),
        "readme": str(README_OUT),
        "sha256": str(SHA_OUT),
    },
    "scientific_status": (
        "Frozen sparse classical supersonic reference for PINN experiments. "
        "Modes accepted visually and numerically at the core; weak oscillatory tails may still be refined later. "
        "Boundary-flagged points are kept with explicit status."
    ),
}

SUMMARY_OUT.write_text(json.dumps(summary, indent=2))

readme = f"""# Supersonic sparse PINN reference v2 - CONFIRMED

This folder freezes the current classical supersonic sparse reference.

## Files

- `{SPEC_OUT.name}`: spectral table.
- `{FIELDS_OUT.name}`: modal fields p, rho, u, v.
- `{POINT_AUDIT_OUT.name}`: per-point modal-field audit.
- `{PDF_OVERVIEW.name}`: overview PDF, one page per point.
- `{PDF_CORE_TAIL.name}`: core + left-tail PDF, one page per point.
- `blumen_ci_overlay_sparse_PINN_reference_v2.png`: final coverage map.
- `SHA256SUMS.txt`: checksums for frozen files.

## Status

The reference is frozen for PINN experiments.

The modes are visually accepted and the modal core is considered stable. 
Weak oscillatory tails are not treated as perfect physical information and may be improved later.

Boundary-flagged points hit either `cr=0` or `ci=0.12` in the first campaign scan. 
They are kept with explicit status rather than silently treated as ordinary strict points.

## Counts

- Spectral points: {summary["n_spectral_points"]}
- Modal rows: {summary["n_modal_rows"]}
- Points with fields: {summary["n_points_with_fields"]}
- Points missing fields: {summary["n_points_missing_fields"]}

## Validation status counts

{json.dumps(summary["validation_status_counts"], indent=2)}
"""
README_OUT.write_text(readme)

# checksums after all outputs are written
checksum_paths = sorted([
    p for p in OUT.iterdir()
    if p.is_file() and p.name != "SHA256SUMS.txt"
])

with SHA_OUT.open("w") as f:
    for p in checksum_paths:
        f.write(f"{sha256_file(p)}  {p.name}\n")

print(json.dumps(summary, indent=2))
print("\nWrote freeze folder:", OUT)
