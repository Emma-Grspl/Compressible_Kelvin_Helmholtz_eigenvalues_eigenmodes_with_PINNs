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


ROOT = Path("assets/classic_supersonic")
FROZEN = ROOT / "supersonic_sparse_PINN_reference_v2_FROZEN"
OUT = ROOT / "supersonic_sparse_PINN_reference_v2_FINAL_FREEZE"

CI_DIR = OUT / "ci"
MODES_DIR = OUT / "modes"
DATA_DIR = OUT / "data"
CODE_DIR = OUT / "code"

for d in [OUT, CI_DIR, MODES_DIR, DATA_DIR, CODE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BLUMEN_CI = Path("assets/classic_supersonic/csv/blumen_validation/supersonic/table_ci_datasets.csv")

SPEC_CANDIDATES = [
    FROZEN / "data/supersonic_sparse_PINN_reference_v2_FROZEN_spectral.csv",
    ROOT / "final_sparse_PINN_reference_v2_CONFIRMED/supersonic_sparse_PINN_reference_v2_CONFIRMED_spectral.csv",
]

RAW_FIELDS_CANDIDATES = [
    FROZEN / "data/supersonic_sparse_PINN_reference_v2_FROZEN_modal_fields_raw_confirmed.csv",
    ROOT / "final_sparse_PINN_reference_v2_CONFIRMED/supersonic_sparse_PINN_reference_v2_CONFIRMED_modal_fields.csv",
]

POL_FIELDS_CANDIDATES = [
    FROZEN / "data/supersonic_sparse_PINN_reference_v2_FROZEN_modal_fields_tail_polished_v1.csv",
    ROOT / "final_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1/supersonic_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1_modal_fields.csv",
]

MODES_PDF_CANDIDATES = [
    FROZEN / "reports/supersonic_sparse_PINN_reference_v2_FROZEN_modes_square_full_y.pdf",
    FROZEN / "reports/supersonic_sparse_PINN_reference_v2_FROZEN_modes_square.pdf",
    ROOT / "final_sparse_PINN_reference_v2_CONFIRMED/supersonic_sparse_PINN_reference_v2_CONFIRMED_modes_overview.pdf",
]

TAIL_PDF_CANDIDATES = [
    FROZEN / "reports/supersonic_sparse_PINN_reference_v2_FROZEN_tail_polished_square_full_y_review.pdf",
    FROZEN / "reports/supersonic_sparse_PINN_reference_v2_FROZEN_tail_polished_square_review.pdf",
    ROOT / "final_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1/tail_polish_raw_vs_polished_left_tail_review.pdf",
]


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise SystemExit("Missing all candidates:\n" + "\n".join(map(str, paths)))


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "Mach" not in df.columns and "M" in df.columns:
        df = df.rename(columns={"M": "Mach"})
    df["Mach"] = pd.to_numeric(df["Mach"], errors="coerce")
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    return df


def curve_to_xy(curve):
    if isinstance(curve, pd.DataFrame):
        cols = list(curve.columns)
        lower = {str(c).lower(): c for c in cols}
        xcol = lower.get("mach") or lower.get("m") or lower.get("x")
        ycol = lower.get("alpha") or lower.get("a") or lower.get("y")

        if xcol is None or ycol is None:
            num = []
            for c in cols:
                s = pd.to_numeric(curve[c], errors="coerce")
                if s.notna().sum() > 3:
                    num.append(c)
            if len(num) >= 2:
                xcol, ycol = num[:2]

        if xcol is None or ycol is None:
            return np.array([]), np.array([])

        x = pd.to_numeric(curve[xcol], errors="coerce").to_numpy(float)
        y = pd.to_numeric(curve[ycol], errors="coerce").to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    if isinstance(curve, dict):
        keys = list(curve.keys())
        lower = {str(k).lower(): k for k in keys}

        for nested in ["points", "data", "df", "curve"]:
            if nested in lower:
                x, y = curve_to_xy(curve[lower[nested]])
                if len(x):
                    return x, y

        xkey = lower.get("mach") or lower.get("m") or lower.get("x")
        ykey = lower.get("alpha") or lower.get("a") or lower.get("y")

        if xkey is not None and ykey is not None:
            x = pd.to_numeric(pd.Series(curve[xkey]), errors="coerce").to_numpy(float)
            y = pd.to_numeric(pd.Series(curve[ykey]), errors="coerce").to_numpy(float)
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
            m = np.isfinite(x) & np.isfinite(y)
            return x[m], y[m]

        numeric = []
        for _, v in curve.items():
            try:
                arr = pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(float)
            except Exception:
                continue
            if len(arr) > 3 and np.isfinite(arr).any():
                numeric.append(arr)

        if len(numeric) >= 2:
            n = min(len(numeric[0]), len(numeric[1]))
            x, y = numeric[0][:n], numeric[1][:n]
            m = np.isfinite(x) & np.isfinite(y)
            return x[m], y[m]

    try:
        arr = np.asarray(curve, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            x, y = arr[:, 0], arr[:, 1]
            m = np.isfinite(x) & np.isfinite(y)
            return x[m], y[m]
        if arr.ndim == 2 and arr.shape[0] >= 2:
            x, y = arr[0], arr[1]
            m = np.isfinite(x) & np.isfinite(y)
            return x[m], y[m]
    except Exception:
        pass

    return np.array([]), np.array([])


def load_blumen_ci_points(path: Path) -> pd.DataFrame:
    rows = []

    if not path.exists():
        raise SystemExit(f"Missing Blumen ci file: {path}")

    try:
        from classical_solver.supersonic.blumen_reference import load_wide_digitized_curves
        curves = load_wide_digitized_curves(path)
        it = curves.values() if isinstance(curves, dict) else curves

        for i, curve in enumerate(it):
            x, y = curve_to_xy(curve)
            for M, a in zip(x, y):
                rows.append({
                    "curve_id": i,
                    "Mach": float(M),
                    "alpha": float(a),
                    "source": str(path),
                })

        if rows:
            return pd.DataFrame(rows).sort_values(["curve_id", "Mach", "alpha"]).reset_index(drop=True)

    except Exception as exc:
        print("[WARN] load_wide_digitized_curves failed:", repr(exc))

    # fallback direct CSV
    df = pd.read_csv(path)
    x, y = curve_to_xy(df)
    for M, a in zip(x, y):
        rows.append({
            "curve_id": 0,
            "Mach": float(M),
            "alpha": float(a),
            "source": str(path),
        })

    if not rows:
        raise SystemExit(f"Could not parse Blumen ci curves from {path}")

    return pd.DataFrame(rows).sort_values(["curve_id", "Mach", "alpha"]).reset_index(drop=True)


def plot_ci_only(blumen: pd.DataFrame, out_png: Path, out_pdf: Path):
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)

    for cid, g in blumen.groupby("curve_id"):
        g = g.sort_values("Mach")
        ax.plot(g["Mach"], g["alpha"], linewidth=1.1, alpha=0.65)
        ax.scatter(g["Mach"], g["alpha"], s=8, alpha=0.65)

    ax.set_title("Blumen ci digitized isolines")
    ax.set_xlabel("Mach M")
    ax.set_ylabel(r"$\alpha$")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_xlim(0.85, 2.12)
    ax.set_ylim(0.0, max(0.48, float(blumen["alpha"].max()) + 0.03))

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_ci_overlay(blumen: pd.DataFrame, spec: pd.DataFrame, out_png: Path, out_pdf: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=200)

    for ax in axes:
        for cid, g in blumen.groupby("curve_id"):
            g = g.sort_values("Mach")
            ax.plot(g["Mach"], g["alpha"], color="0.72", linewidth=1.0, alpha=0.65)
            ax.scatter(g["Mach"], g["alpha"], color="0.72", s=5, alpha=0.55)

    machs = sorted(spec["Mach"].dropna().unique())
    cmap = plt.get_cmap("tab20", max(len(machs), 1))
    colors = {m: cmap(i) for i, m in enumerate(machs)}

    for ax in axes:
        for M in machs:
            g = spec[np.isclose(spec["Mach"], M)].copy()
            status = g["validation_status"].astype(str)

            legacy = g[status.str.contains("legacy|modal_spectral", case=False, na=False)]
            small = g[status.str.contains("smallM", case=False, na=False)]
            boundary = g[status.str.contains("boundary_flag", case=False, na=False)]
            high = g[status.str.contains("core_stable|tail_sensitive", case=False, na=False) & (g["Mach"] >= 1.75)]

            if len(legacy):
                ax.scatter(
                    legacy["Mach"], legacy["alpha"],
                    s=42, marker="D", color=colors[M],
                    edgecolors="black", linewidths=0.45, zorder=3,
                )

            if len(small):
                small_clean = small[~small.index.isin(boundary.index)]
                if len(small_clean):
                    ax.scatter(
                        small_clean["Mach"], small_clean["alpha"],
                        s=45, marker="o", color=colors[M],
                        edgecolors="black", linewidths=0.45, zorder=4,
                    )

            if len(boundary):
                ax.scatter(
                    boundary["Mach"], boundary["alpha"],
                    s=65, marker="X", color=colors[M],
                    edgecolors="black", linewidths=0.6, zorder=5,
                )

            if len(high):
                ax.scatter(
                    high["Mach"], high["alpha"],
                    s=70, marker="*", color=colors[M],
                    edgecolors="black", linewidths=0.5, zorder=4,
                )

        ax.set_xlabel("Mach M")
        ax.set_ylabel(r"$\alpha$")
        ax.grid(True, alpha=0.25, linestyle=":")

    axes[0].set_title("Blumen ci digitized isolines + frozen reference points")
    axes[1].set_title("Zoom frozen reference points")

    axes[0].set_xlim(0.85, 2.12)
    axes[0].set_ylim(0.0, max(0.48, float(max(blumen["alpha"].max(), spec["alpha"].max())) + 0.03))

    axes[1].set_xlim(float(spec["Mach"].min()) - 0.03, float(spec["Mach"].max()) + 0.03)
    axes[1].set_ylim(max(0.035, float(spec["alpha"].min()) - 0.02), float(spec["alpha"].max()) + 0.03)

    counts = "\n".join(
        f"M={M:.2f}: {len(spec[np.isclose(spec['Mach'], M)])}"
        for M in machs
    )
    axes[1].text(
        0.985, 0.03, counts,
        transform=axes[1].transAxes,
        ha="right", va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.88),
    )

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="0.5", markeredgecolor="black", label="legacy/base", markersize=7),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5", markeredgecolor="black", label="small-M campaign", markersize=7),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="0.5", markeredgecolor="black", label="boundary flag", markersize=8),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="0.5", markeredgecolor="black", label="M=1.8/1.9 core-stable", markersize=10),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def copy_code_snapshot():
    copied = []

    candidates = [
        "code/src/classical_solver/supersonic/mstab17_supersonic_solver.py",
        "code/scripts/evaluation/blumen_reference.py",

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
        "scripts/regenerate_supersonic_v2_full_y_square_pdfs.py",
        "code/scripts/evaluation/package_supersonic_sparse_reference_v2_final_freeze.py",
    ]

    for rel in candidates:
        src = Path(rel)
        if src.exists():
            dst = CODE_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)

    for pattern in [
        "slurm/jz_*supersonic*.slurm",
        "slurm/jz_*M18*.slurm",
        "slurm/jz_*smallM*.slurm",
        "slurm/jz_scan_smallM_sparse_campaign.slurm",
        "slurm/jz_audit_M18_M19_refined_convergence.slurm",
    ]:
        for src in Path(".").glob(pattern):
            if src.exists():
                dst = CODE_DIR / src
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied.append(str(src))

    # Existing code snapshot if available.
    snapshot = FROZEN / "code_snapshot"
    if snapshot.exists():
        dst_root = CODE_DIR / "previous_code_snapshot"
        if dst_root.exists():
            shutil.rmtree(dst_root)
        shutil.copytree(snapshot, dst_root)
        copied.append(str(snapshot))

    git_info = {
        "branch": run(["git", "branch", "--show-current"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
        "diff_stat": run(["git", "diff", "--stat"]),
    }
    (CODE_DIR / "git_info.json").write_text(json.dumps(git_info, indent=2))

    return copied, git_info


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    spec_src = first_existing(SPEC_CANDIDATES)
    raw_src = first_existing(RAW_FIELDS_CANDIDATES)
    pol_src = first_existing(POL_FIELDS_CANDIDATES)
    modes_pdf_src = first_existing(MODES_PDF_CANDIDATES)
    tail_pdf_src = first_existing(TAIL_PDF_CANDIDATES)

    spec = norm_cols(pd.read_csv(spec_src))
    spec = spec.sort_values(["Mach", "alpha"]).drop_duplicates(["Mach", "alpha"], keep="first").reset_index(drop=True)

    # Data canonical copies.
    spectral_out = DATA_DIR / "supersonic_sparse_PINN_reference_v2_FINAL_spectral.csv"
    raw_out = DATA_DIR / "supersonic_sparse_PINN_reference_v2_FINAL_modal_fields_raw_confirmed.csv"
    pol_out = DATA_DIR / "supersonic_sparse_PINN_reference_v2_FINAL_modal_fields_tail_polished_v1.csv"

    spec.to_csv(spectral_out, index=False)
    shutil.copy2(raw_src, raw_out)
    shutil.copy2(pol_src, pol_out)

    # Modes PDFs.
    modes_pdf_out = MODES_DIR / "supersonic_sparse_PINN_reference_v2_FINAL_modes_square_full_y.pdf"
    tail_pdf_out = MODES_DIR / "supersonic_sparse_PINN_reference_v2_FINAL_tail_polished_square_full_y_review.pdf"

    shutil.copy2(modes_pdf_src, modes_pdf_out)
    shutil.copy2(tail_pdf_src, tail_pdf_out)

    # CI assets.
    blumen = load_blumen_ci_points(BLUMEN_CI)

    blumen_out = CI_DIR / "blumen_ci_digitized_points.csv"
    ref_out = CI_DIR / "reference_points_ci_overlay.csv"

    blumen.to_csv(blumen_out, index=False)
    spec.to_csv(ref_out, index=False)

    plot_ci_only(
        blumen,
        CI_DIR / "blumen_ci_digitized_only.png",
        CI_DIR / "blumen_ci_digitized_only.pdf",
    )

    plot_ci_overlay(
        blumen,
        spec,
        CI_DIR / "blumen_ci_digitized_with_reference_points.png",
        CI_DIR / "blumen_ci_digitized_with_reference_points.pdf",
    )

    copied_code, git_info = copy_code_snapshot()

    readme = f"""# Supersonic sparse PINN reference v2 FINAL FREEZE

This folder freezes the final classical supersonic sparse reference used for PINN work.

## Structure

- `ci/`: Blumen digitized ci isolines, reference points, and overlay maps.
- `modes/`: modal PDFs.
- `data/`: spectral table and modal field CSVs.
- `code/`: solver, scripts, SLURM/config snapshots, and git metadata.

## Main files

### ci

- `ci/blumen_ci_digitized_only.png`
- `ci/blumen_ci_digitized_only.pdf`
- `ci/blumen_ci_digitized_with_reference_points.png`
- `ci/blumen_ci_digitized_with_reference_points.pdf`
- `ci/blumen_ci_digitized_points.csv`
- `ci/reference_points_ci_overlay.csv`

### modes

- `modes/{modes_pdf_out.name}`
- `modes/{tail_pdf_out.name}`

### data

- `data/{spectral_out.name}`
- `data/{raw_out.name}`
- `data/{pol_out.name}`

## Scientific status

Raw confirmed fields remain the primary reference.  
Tail-polished fields are a documented export derivative; spectral cr/ci are unchanged.  
Boundary-flagged points are kept with explicit status.  

Spectral points: {spec[['Mach', 'alpha']].drop_duplicates().shape[0]}
"""

    (OUT / "README.md").write_text(readme)

    manifest = {
        "status": "FINAL_FREEZE_BUILT",
        "dataset": "supersonic_sparse_PINN_reference_v2_FINAL_FREEZE",
        "n_spectral_points": int(spec[["Mach", "alpha"]].drop_duplicates().shape[0]),
        "point_counts_by_Mach": spec.groupby("Mach").size().to_dict(),
        "validation_status_counts": spec["validation_status"].astype(str).value_counts(dropna=False).to_dict(),
        "inputs": {
            "spectral": str(spec_src),
            "raw_fields": str(raw_src),
            "tail_polished_fields": str(pol_src),
            "modes_pdf": str(modes_pdf_src),
            "tail_pdf": str(tail_pdf_src),
            "blumen_ci": str(BLUMEN_CI),
        },
        "outputs": {
            "ci_dir": str(CI_DIR),
            "modes_dir": str(MODES_DIR),
            "data_dir": str(DATA_DIR),
            "code_dir": str(CODE_DIR),
            "spectral": str(spectral_out),
            "raw_fields": str(raw_out),
            "tail_polished_fields": str(pol_out),
            "modes_pdf": str(modes_pdf_out),
            "tail_pdf": str(tail_pdf_out),
            "ci_digitized_only_png": str(CI_DIR / "blumen_ci_digitized_only.png"),
            "ci_digitized_overlay_png": str(CI_DIR / "blumen_ci_digitized_with_reference_points.png"),
        },
        "copied_code": copied_code,
        "git_info": git_info,
    }

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    with (OUT / "SHA256SUMS.txt").open("w") as f:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS.txt":
                f.write(f"{sha256_file(p)}  {p.relative_to(OUT)}\n")

    print(json.dumps(manifest, indent=2))
    print("\nWROTE:", OUT)


if __name__ == "__main__":
    main()
