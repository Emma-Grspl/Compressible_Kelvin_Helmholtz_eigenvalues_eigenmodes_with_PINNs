#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path("assets/classic_supersonic/final_sparse_PINN_reference")
OUT.mkdir(parents=True, exist_ok=True)

SELECTED_SETTING = "ms3_y2000"

BASE_SPECTRAL_CANDIDATES = [
    Path("assets/classic_supersonic/supersonic_modal_spectral_validated.csv"),
    Path("assets/classic_supersonic/final_44pts_validated_only/supersonic_modal_spectral_validated_44pts.csv"),
]

BASE_FIELDS_CANDIDATES = [
    Path("assets/classic_supersonic/final_44pts_validated_only/supersonic_modal_fields_p_rho_u_v_44pts_validated_only.csv"),
    Path("assets/classic_supersonic/validated_modal_points/rebuilt_aggregates_latest/supersonic_reference_core_local_modal_fields_REBUILT.csv"),
]

CONV = Path("assets/classic_supersonic/scan_M18_M19_alpha_0_03_strict/refined_near_valid_branch/convergence_audit_full")
CONV_AUDIT = CONV / "convergence_audit_by_point.csv"
CORE_TAIL = CONV / "core_vs_tail_phase_convergence.csv"
FIELDS_BY_SETTING = CONV / "fields_by_setting"

BLUMEN_CI = Path("assets/classic_supersonic/csv/blumen_validation/supersonic/table_ci_datasets.csv")


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise SystemExit("Missing all candidate files:\n" + "\n".join(map(str, paths)))


def norm_cols(df):
    if "Mach" not in df.columns and "M" in df.columns:
        df = df.rename(columns={"M": "Mach"})
    df["Mach"] = pd.to_numeric(df["Mach"], errors="coerce")
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    return df


def point_key(df):
    return list(zip(df["Mach"].astype(float).round(12), df["alpha"].astype(float).round(12)))


def add_missing_cols(df, columns):
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df[columns]


def load_blumen_curves(path: Path):
    curves_xy = []

    if not path.exists():
        return curves_xy

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

        if isinstance(curve, (list, tuple)) and len(curve) >= 2:
            try:
                x = pd.to_numeric(pd.Series(curve[0]), errors="coerce").to_numpy(float)
                y = pd.to_numeric(pd.Series(curve[1]), errors="coerce").to_numpy(float)
                n = min(len(x), len(y))
                x, y = x[:n], y[:n]
                m = np.isfinite(x) & np.isfinite(y)
                return x[m], y[m]
            except Exception:
                pass

        try:
            arr = np.asarray(curve, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                x, y = arr[:, 0], arr[:, 1]
                m = np.isfinite(x) & np.isfinite(y)
                return x[m], y[m]
        except Exception:
            pass

        return np.array([]), np.array([])

    try:
        from classical_solver.supersonic.blumen_reference import load_wide_digitized_curves
        curves = load_wide_digitized_curves(path)
        it = curves.values() if isinstance(curves, dict) else curves
        for c in it:
            x, y = curve_to_xy(c)
            if len(x):
                order = np.argsort(x)
                curves_xy.append((x[order], y[order]))
        if curves_xy:
            return curves_xy
    except Exception as exc:
        print("[WARN] Blumen loader failed:", repr(exc))

    # Fallback pandas direct.
    df = pd.read_csv(path)
    x, y = curve_to_xy(df)
    if len(x):
        curves_xy.append((x, y))

    return curves_xy


# ------------------------------------------------------------------
# 1. Load base 44 spectral and fields.
# ------------------------------------------------------------------
base_spectral_path = first_existing(BASE_SPECTRAL_CANDIDATES)
base_fields_path = first_existing(BASE_FIELDS_CANDIDATES)

spectral44 = norm_cols(pd.read_csv(base_spectral_path))
spectral44 = spectral44.dropna(subset=["Mach", "alpha"]).copy()

# Force strict 44 legacy only: no M >= 1.8 in base.
spectral44 = spectral44[spectral44["Mach"].astype(float) < 1.8].copy()

spectral44["validation_status"] = spectral44.get("validation_status", "legacy_44_validated_reference")
spectral44["source"] = spectral44.get("source", "legacy_classic_supersonic_44pts")
spectral44["reference_role"] = "base_legacy_44"

target44 = set(point_key(spectral44))

fields44 = norm_cols(pd.read_csv(base_fields_path, low_memory=False))
fields44 = fields44.dropna(subset=["Mach", "alpha", "y"]).copy()
fields44["_key"] = point_key(fields44)
fields44 = fields44[fields44["_key"].isin(target44)].drop(columns=["_key"]).copy()

# Deduplicate base fields by Mach/alpha/y.
fields44["_Mach_key"] = fields44["Mach"].astype(float).round(12)
fields44["_alpha_key"] = fields44["alpha"].astype(float).round(12)
fields44["_y_key"] = pd.to_numeric(fields44["y"], errors="coerce").round(10)

fields44 = (
    fields44
    .sort_values(["_Mach_key", "_alpha_key", "_y_key"])
    .drop_duplicates(["_Mach_key", "_alpha_key", "_y_key"], keep="first")
    .drop(columns=["_Mach_key", "_alpha_key", "_y_key"])
    .sort_values(["Mach", "alpha", "y"])
    .reset_index(drop=True)
)

fields44["validation_status"] = fields44.get("validation_status", "legacy_44_validated_reference")
fields44["source"] = fields44.get("source", "legacy_classic_supersonic_44pts")
fields44["reference_role"] = "base_legacy_44"


# ------------------------------------------------------------------
# 2. Load M=1.8/1.9 convergence-audited branch.
# ------------------------------------------------------------------
if not CONV_AUDIT.exists():
    raise SystemExit(f"Missing convergence audit: {CONV_AUDIT}")

audit = pd.read_csv(CONV_AUDIT)
audit = norm_cols(audit)

ok = audit[audit["convergence_status"].eq("converged_requires_visual_confirmation")].copy()
ok = ok.sort_values(["Mach", "alpha"]).reset_index(drop=True)

if len(ok) == 0:
    raise SystemExit("No converged M18/M19 candidates found.")

core = pd.read_csv(CORE_TAIL) if CORE_TAIL.exists() else pd.DataFrame()
if len(core):
    core = norm_cols(core)

spec_ext_rows = []
fields_ext_parts = []

for _, r in ok.iterrows():
    M = float(r["Mach"])
    a = float(r["alpha"])

    fpath = FIELDS_BY_SETTING / f"M{M:.2f}_alpha{a:.5f}_{SELECTED_SETTING}_fields.csv"
    if not fpath.exists():
        raise SystemExit(f"Missing field file: {fpath}")

    f = norm_cols(pd.read_csv(fpath, low_memory=False))
    f = f.sort_values("y").reset_index(drop=True)

    # Audit metrics.
    for c in [
        "cr_range", "ci_range", "max_stage1",
        "max_edge_frac", "max_center_jump", "max_modal_rel_l2",
        "convergence_status",
    ]:
        f[c] = r[c] if c in r else np.nan

    # Core/tail metrics per point.
    if len(core):
        subc = core[np.isclose(core["Mach"], M) & np.isclose(core["alpha"], a)]
        max_core_5 = float(pd.to_numeric(subc["rel_l2_core_amp_ge_0.05"], errors="coerce").max()) if len(subc) else np.nan
        max_tail = float(pd.to_numeric(subc["rel_l2_tail_amp_0p001_0p02"], errors="coerce").max()) if len(subc) else np.nan
    else:
        max_core_5 = np.nan
        max_tail = np.nan

    f["max_core_rel_l2_amp_ge_5pct"] = max_core_5
    f["max_tail_rel_l2_amp_0p1pct_to_2pct"] = max_tail
    f["source"] = f"shooting_convergence_audit_{SELECTED_SETTING}"
    f["validation_status"] = "validated_core_stable_tail_sensitive"
    f["reference_role"] = "M18_M19_extension_core_stable"

    fields_ext_parts.append(f)

    first = f.iloc[0]
    spec_ext_rows.append({
        "Mach": M,
        "alpha": a,
        "cr": float(first["cr"]),
        "ci": float(first["ci"]),
        "omega_i": float(first["omega_i"]) if "omega_i" in first else float(a * first["ci"]),
        "source": f"shooting_convergence_audit_{SELECTED_SETTING}",
        "validation_status": "validated_core_stable_tail_sensitive",
        "reference_role": "M18_M19_extension_core_stable",
        "selected_setting": SELECTED_SETTING,
        "cr_range": float(r["cr_range"]) if "cr_range" in r else np.nan,
        "ci_range": float(r["ci_range"]) if "ci_range" in r else np.nan,
        "max_stage1": float(r["max_stage1"]) if "max_stage1" in r else np.nan,
        "max_edge_frac": float(r["max_edge_frac"]) if "max_edge_frac" in r else np.nan,
        "max_center_jump": float(r["max_center_jump"]) if "max_center_jump" in r else np.nan,
        "max_modal_rel_l2": float(r["max_modal_rel_l2"]) if "max_modal_rel_l2" in r else np.nan,
        "max_core_rel_l2_amp_ge_5pct": max_core_5,
        "max_tail_rel_l2_amp_0p1pct_to_2pct": max_tail,
        "note": "Core/mid-field stable; weak oscillatory tails are mapping-sensitive and should be improved later.",
    })

spec_ext = pd.DataFrame(spec_ext_rows)
fields_ext = pd.concat(fields_ext_parts, ignore_index=True)


# ------------------------------------------------------------------
# 3. Combine spectral and fields.
# ------------------------------------------------------------------
overlap = set(point_key(spectral44)).intersection(set(point_key(spec_ext)))
if overlap:
    raise SystemExit(f"Unexpected overlap between base and extension: {sorted(overlap)[:10]}")

all_spec_cols = list(dict.fromkeys(list(spectral44.columns) + list(spec_ext.columns)))
spectral_final = pd.concat(
    [add_missing_cols(spectral44, all_spec_cols), add_missing_cols(spec_ext, all_spec_cols)],
    ignore_index=True,
)
spectral_final = spectral_final.sort_values(["Mach", "alpha"]).reset_index(drop=True)

all_field_cols = list(dict.fromkeys(list(fields44.columns) + list(fields_ext.columns)))
fields_final = pd.concat(
    [add_missing_cols(fields44, all_field_cols), add_missing_cols(fields_ext, all_field_cols)],
    ignore_index=True,
)
fields_final = fields_final.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)

# Deduplicate final fields by Mach/alpha/y.
fields_final["_Mach_key"] = fields_final["Mach"].astype(float).round(12)
fields_final["_alpha_key"] = fields_final["alpha"].astype(float).round(12)
fields_final["_y_key"] = pd.to_numeric(fields_final["y"], errors="coerce").round(10)

fields_final = (
    fields_final
    .drop_duplicates(["_Mach_key", "_alpha_key", "_y_key"], keep="first")
    .drop(columns=["_Mach_key", "_alpha_key", "_y_key"])
    .sort_values(["Mach", "alpha", "y"])
    .reset_index(drop=True)
)

spectral_out = OUT / "supersonic_sparse_PINN_reference_spectral.csv"
fields_out = OUT / "supersonic_sparse_PINN_reference_modal_fields.csv"

spectral_final.to_csv(spectral_out, index=False)
fields_final.to_csv(fields_out, index=False)


# ------------------------------------------------------------------
# 4. Coverage audit and suggested next targets.
# ------------------------------------------------------------------
pts = spectral_final[["Mach", "alpha", "validation_status", "reference_role"]].drop_duplicates().copy()
pts = pts.sort_values(["Mach", "alpha"]).reset_index(drop=True)

coverage_rows = []
suggested_rows = []

for M, g in pts.groupby("Mach"):
    alphas = np.sort(g["alpha"].astype(float).unique())
    gaps = np.diff(alphas) if len(alphas) > 1 else np.array([])
    max_gap = float(gaps.max()) if len(gaps) else np.nan

    coverage_rows.append({
        "Mach": float(M),
        "n_points": int(len(alphas)),
        "alpha_min": float(alphas.min()),
        "alpha_max": float(alphas.max()),
        "max_alpha_gap": max_gap,
        "needs_low_alpha_review": bool((M <= 1.30) and (alphas.min() > 0.035)),
        "needs_gap_review": bool(np.isfinite(max_gap) and max_gap > 0.05),
    })

    # Low-alpha candidates for small Mach.
    if M <= 1.30 and alphas.min() > 0.035:
        low_targets = np.arange(0.02, alphas.min() - 0.005, 0.02)
        for a in low_targets:
            if a > 0 and a < alphas.min() - 1e-9:
                suggested_rows.append({
                    "Mach": float(M),
                    "alpha": float(round(a, 5)),
                    "priority": "high",
                    "reason": "low_alpha_undercovered_for_small_Mach",
                })

    # Internal large gaps.
    for a0, a1 in zip(alphas[:-1], alphas[1:]):
        gap = a1 - a0
        if gap > 0.05:
            suggested_rows.append({
                "Mach": float(M),
                "alpha": float(round(0.5 * (a0 + a1), 5)),
                "priority": "medium",
                "reason": f"large_alpha_gap_{gap:.4f}",
            })

coverage = pd.DataFrame(coverage_rows).sort_values("Mach")
suggested = pd.DataFrame(suggested_rows).drop_duplicates() if suggested_rows else pd.DataFrame(
    columns=["Mach", "alpha", "priority", "reason"]
)

coverage_out = OUT / "coverage_by_Mach.csv"
suggested_out = OUT / "suggested_next_targets.csv"

coverage.to_csv(coverage_out, index=False)
suggested.to_csv(suggested_out, index=False)


# ------------------------------------------------------------------
# 5. Blumen overlay.
# ------------------------------------------------------------------
curves = load_blumen_curves(BLUMEN_CI)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=180)

for ax in axes:
    for x, y in curves:
        ax.plot(x, y, color="0.65", linewidth=1.0, alpha=0.65)

base = spectral_final[spectral_final["reference_role"].eq("base_legacy_44")]
ext = spectral_final[spectral_final["reference_role"].eq("M18_M19_extension_core_stable")]

machs = sorted(spectral_final["Mach"].dropna().unique())
cmap = plt.get_cmap("tab20", max(len(machs), 1))
colors = {m: cmap(i) for i, m in enumerate(machs)}

for ax in axes:
    for m in machs:
        b = base[np.isclose(base["Mach"], m)]
        e = ext[np.isclose(ext["Mach"], m)]

        if len(b):
            ax.scatter(
                b["Mach"], b["alpha"],
                s=55, marker="D",
                color=colors[m],
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
                label=f"M={m:.2f} legacy" if ax is axes[0] else None,
            )

        if len(e):
            ax.scatter(
                e["Mach"], e["alpha"],
                s=95, marker="*",
                color=colors[m],
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
                label=f"M={m:.2f} M18/M19" if ax is axes[0] else None,
            )

    ax.set_xlabel("Mach M")
    ax.set_ylabel(r"$\alpha$")
    ax.grid(True, alpha=0.25, linestyle=":")

axes[0].set_title("Blumen ci digitise + sparse PINN reference")
axes[1].set_title("Zoom points reference")

if curves:
    all_x = np.concatenate([x for x, _ in curves])
    all_y = np.concatenate([y for _, y in curves])
    axes[0].set_xlim(float(np.nanmin(all_x)) - 0.03, float(np.nanmax(all_x)) + 0.03)
    axes[0].set_ylim(0.0, max(float(np.nanmax(all_y)), float(spectral_final["alpha"].max())) + 0.02)
else:
    axes[0].set_xlim(float(spectral_final["Mach"].min()) - 0.05, float(spectral_final["Mach"].max()) + 0.05)
    axes[0].set_ylim(0.0, float(spectral_final["alpha"].max()) + 0.02)

axes[1].set_xlim(float(spectral_final["Mach"].min()) - 0.05, float(spectral_final["Mach"].max()) + 0.05)
axes[1].set_ylim(max(0.0, float(spectral_final["alpha"].min()) - 0.02), float(spectral_final["alpha"].max()) + 0.02)

counts = "\n".join(
    f"M={m:.2f}: {len(spectral_final[np.isclose(spectral_final['Mach'], m)])}"
    for m in machs
)
axes[1].text(
    0.98, 0.03, counts,
    transform=axes[1].transAxes,
    ha="right", va="bottom",
    fontsize=8,
    bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85),
)

handles, labels = axes[0].get_legend_handles_labels()
if handles:
    axes[0].legend(handles[:16], labels[:16], fontsize=6, loc="upper right")

fig.tight_layout()

overlay_png = OUT / "blumen_ci_overlay_sparse_PINN_reference.png"
overlay_pdf = OUT / "blumen_ci_overlay_sparse_PINN_reference.pdf"

fig.savefig(overlay_png, bbox_inches="tight")
fig.savefig(overlay_pdf, bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------------
# 6. Summary.
# ------------------------------------------------------------------
summary = {
    "status": "sparse_PINN_reference_built",
    "spectral_file": str(spectral_out),
    "modal_fields_file": str(fields_out),
    "overlay_png": str(overlay_png),
    "overlay_pdf": str(overlay_pdf),
    "coverage_file": str(coverage_out),
    "suggested_next_targets_file": str(suggested_out),
    "base_spectral_source": str(base_spectral_path),
    "base_fields_source": str(base_fields_path),
    "M18_M19_selected_setting": SELECTED_SETTING,
    "n_base_spectral_points": int(len(spectral44[["Mach", "alpha"]].drop_duplicates())),
    "n_extension_spectral_points": int(len(spec_ext)),
    "n_total_spectral_points": int(len(spectral_final[["Mach", "alpha"]].drop_duplicates())),
    "n_total_modal_rows": int(len(fields_final)),
    "point_counts_by_Mach": spectral_final.groupby("Mach").size().to_dict(),
    "validation_status_counts": spectral_final["validation_status"].value_counts(dropna=False).to_dict(),
    "important_note": (
        "M=1.8/1.9 modes are accepted as core-stable/tail-sensitive reference candidates. "
        "Weak oscillatory tails remain mapping-sensitive and should be improved later."
    ),
}

summary_out = OUT / "summary.json"
summary_out.write_text(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2))
