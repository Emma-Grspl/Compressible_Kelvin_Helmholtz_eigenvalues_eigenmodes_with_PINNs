#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path("assets/classic_supersonic/final_sparse_PINN_reference")
CAMPAIGN = Path("assets/classic_supersonic/campaign_smallM_low_high_alpha_scan")
OUT = Path("assets/classic_supersonic/final_sparse_PINN_reference_v2_smallM_M18M19")
OUT.mkdir(parents=True, exist_ok=True)

BASE_SPEC = BASE / "supersonic_sparse_PINN_reference_spectral.csv"
BASE_FIELDS = BASE / "supersonic_sparse_PINN_reference_modal_fields.csv"

CAMPAIGN_SPEC = CAMPAIGN / "campaign_near_valid_best_per_target.csv"
CAMPAIGN_FIELDS = CAMPAIGN / "campaign_smallM_modal_fields_reconstructed.csv"

BLUMEN_CI = Path("assets/classic_supersonic/csv/blumen_validation/supersonic/table_ci_datasets.csv")


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "Mach" not in df.columns and "M" in df.columns:
        df = df.rename(columns={"M": "Mach"})
    df["Mach"] = pd.to_numeric(df["Mach"], errors="coerce")
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    return df


def add_missing_cols(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df[columns]


def key_df(df: pd.DataFrame):
    return list(zip(df["Mach"].astype(float).round(12), df["alpha"].astype(float).round(12)))


def load_blumen_curves(path: Path):
    curves = []
    if not path.exists():
        return curves

    try:
        from classical_solver.supersonic.blumen_reference import load_wide_digitized_curves
        raw = load_wide_digitized_curves(path)
        it = raw.values() if isinstance(raw, dict) else raw

        for c in it:
            if isinstance(c, dict):
                keys = {str(k).lower(): k for k in c.keys()}

                if ("mach" in keys or "m" in keys) and "alpha" in keys:
                    xkey = keys.get("mach", keys.get("m"))
                    ykey = keys["alpha"]
                    x = pd.to_numeric(pd.Series(c[xkey]), errors="coerce").to_numpy(float)
                    y = pd.to_numeric(pd.Series(c[ykey]), errors="coerce").to_numpy(float)
                else:
                    vals = []
                    for v in c.values():
                        try:
                            arr = pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(float)
                            if len(arr) > 3:
                                vals.append(arr)
                        except Exception:
                            pass
                    if len(vals) < 2:
                        continue
                    n = min(len(vals[0]), len(vals[1]))
                    x, y = vals[0][:n], vals[1][:n]

            elif isinstance(c, pd.DataFrame):
                cols = list(c.columns)
                lower = {str(col).lower(): col for col in cols}
                xcol = lower.get("mach") or lower.get("m")
                ycol = lower.get("alpha")
                if xcol is None or ycol is None:
                    continue
                x = pd.to_numeric(c[xcol], errors="coerce").to_numpy(float)
                y = pd.to_numeric(c[ycol], errors="coerce").to_numpy(float)

            else:
                arr = np.asarray(c, dtype=float)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                x, y = arr[:, 0], arr[:, 1]

            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum():
                order = np.argsort(x[m])
                curves.append((x[m][order], y[m][order]))

        return curves

    except Exception as exc:
        print("[WARN] Could not load Blumen curves:", repr(exc))
        return curves


# -------------------------------------------------------------------
# Load base reference.
# -------------------------------------------------------------------
if not BASE_SPEC.exists():
    raise SystemExit(f"Missing {BASE_SPEC}")
if not BASE_FIELDS.exists():
    raise SystemExit(f"Missing {BASE_FIELDS}")
if not CAMPAIGN_SPEC.exists():
    raise SystemExit(f"Missing {CAMPAIGN_SPEC}")
if not CAMPAIGN_FIELDS.exists():
    raise SystemExit(f"Missing {CAMPAIGN_FIELDS}")

base_spec = norm_cols(pd.read_csv(BASE_SPEC))
base_fields = norm_cols(pd.read_csv(BASE_FIELDS, low_memory=False))

campaign = norm_cols(pd.read_csv(CAMPAIGN_SPEC))
campaign_fields = norm_cols(pd.read_csv(CAMPAIGN_FIELDS, low_memory=False))

# Best per campaign point only.
campaign = (
    campaign.sort_values(["Mach", "alpha", "stage1_mismatch"], na_position="last")
    .groupby(["Mach", "alpha"], as_index=False)
    .first()
    .sort_values(["Mach", "alpha"])
    .reset_index(drop=True)
)

# Keep only campaign points not already in base.
base_keys = set(key_df(base_spec))
campaign["_key"] = key_df(campaign)
new_campaign = campaign[~campaign["_key"].isin(base_keys)].drop(columns=["_key"]).copy()

new_keys = set(key_df(new_campaign))
campaign_fields["_key"] = key_df(campaign_fields)
campaign_fields = campaign_fields[campaign_fields["_key"].isin(new_keys)].drop(columns=["_key"]).copy()

# Status labels.
new_campaign["touches_ci_upper_0p12"] = np.isclose(pd.to_numeric(new_campaign["ci"], errors="coerce"), 0.12, atol=1e-10)
new_campaign["touches_cr_lower_0"] = np.isclose(pd.to_numeric(new_campaign["cr"], errors="coerce"), 0.0, atol=1e-10)
new_campaign["boundary_flag"] = new_campaign["touches_ci_upper_0p12"] | new_campaign["touches_cr_lower_0"]

def spectral_status(row):
    strict = row.get("campaign_target_status") == "strict_auto_validated"
    near = row.get("campaign_target_status") == "near_valid_except_adjacent_jump"
    boundary = bool(row.get("boundary_flag", False))

    if strict and boundary:
        return "validated_visual_smallM_strict_boundary_flag"
    if near and boundary:
        return "validated_visual_smallM_tail_sensitive_boundary_flag"
    if strict:
        return "validated_visual_smallM_strict"
    if near:
        return "validated_visual_smallM_tail_sensitive"
    return "smallM_rejected_not_used"

new_campaign["validation_status"] = new_campaign.apply(spectral_status, axis=1)
new_campaign["reference_role"] = "smallM_low_high_alpha_campaign"
new_campaign["source"] = "smallM_low_high_alpha_campaign_mapping5_visual_review"
new_campaign["selected_setting"] = "mapping_scale_5_y1600"
new_campaign["note"] = (
    "Small-M campaign point visually accepted. "
    "Tail oscillations may be imperfect, consistent with legacy modes. "
    "Boundary-flagged points hit ci=0.12 or cr=0 in initial scan and should be rescanned later."
)

# Fields metadata.
field_meta = new_campaign[
    [
        "Mach", "alpha", "campaign_target_status", "validation_status",
        "reference_role", "source", "selected_setting",
        "touches_ci_upper_0p12", "touches_cr_lower_0", "boundary_flag",
    ]
].copy()

campaign_fields = campaign_fields.merge(
    field_meta,
    on=["Mach", "alpha"],
    how="left",
    suffixes=("", "_meta"),
)

for col in ["validation_status", "reference_role", "source", "selected_setting"]:
    meta_col = f"{col}_meta"
    if meta_col in campaign_fields.columns:
        campaign_fields[col] = campaign_fields[meta_col].combine_first(campaign_fields.get(col))
        campaign_fields = campaign_fields.drop(columns=[meta_col])

campaign_fields["note"] = (
    "Small-M visual campaign modal field. "
    "Accepted for sparse PINN reference; tails may require later refinement."
)

# -------------------------------------------------------------------
# Combine spectral.
# -------------------------------------------------------------------
all_spec_cols = list(dict.fromkeys(list(base_spec.columns) + list(new_campaign.columns)))
spec_final = pd.concat(
    [
        add_missing_cols(base_spec.copy(), all_spec_cols),
        add_missing_cols(new_campaign.copy(), all_spec_cols),
    ],
    ignore_index=True,
)

spec_final = (
    spec_final
    .sort_values(["Mach", "alpha"])
    .drop_duplicates(["Mach", "alpha"], keep="first")
    .reset_index(drop=True)
)

# -------------------------------------------------------------------
# Combine fields.
# -------------------------------------------------------------------
all_field_cols = list(dict.fromkeys(list(base_fields.columns) + list(campaign_fields.columns)))
fields_final = pd.concat(
    [
        add_missing_cols(base_fields.copy(), all_field_cols),
        add_missing_cols(campaign_fields.copy(), all_field_cols),
    ],
    ignore_index=True,
)

fields_final["y"] = pd.to_numeric(fields_final["y"], errors="coerce")
fields_final = fields_final.dropna(subset=["Mach", "alpha", "y"])

fields_final["_Mach_key"] = fields_final["Mach"].astype(float).round(12)
fields_final["_alpha_key"] = fields_final["alpha"].astype(float).round(12)
fields_final["_y_key"] = fields_final["y"].astype(float).round(10)

fields_final = (
    fields_final
    .sort_values(["_Mach_key", "_alpha_key", "_y_key"])
    .drop_duplicates(["_Mach_key", "_alpha_key", "_y_key"], keep="first")
    .drop(columns=["_Mach_key", "_alpha_key", "_y_key"])
    .sort_values(["Mach", "alpha", "y"])
    .reset_index(drop=True)
)

# -------------------------------------------------------------------
# Coverage audit.
# -------------------------------------------------------------------
coverage_rows = []

for M, g in spec_final.groupby("Mach"):
    alphas = np.sort(g["alpha"].astype(float).unique())
    gaps = np.diff(alphas) if len(alphas) > 1 else np.array([])

    coverage_rows.append({
        "Mach": float(M),
        "n_points": int(len(alphas)),
        "alpha_min": float(alphas.min()),
        "alpha_max": float(alphas.max()),
        "max_alpha_gap": float(gaps.max()) if len(gaps) else np.nan,
        "n_strict": int(g["validation_status"].astype(str).str.contains("strict", na=False).sum()),
        "n_tail_sensitive": int(g["validation_status"].astype(str).str.contains("tail_sensitive", na=False).sum()),
        "n_boundary_flag": int(g["validation_status"].astype(str).str.contains("boundary_flag", na=False).sum()),
    })

coverage = pd.DataFrame(coverage_rows).sort_values("Mach").reset_index(drop=True)

# -------------------------------------------------------------------
# Suggested remaining gaps.
# -------------------------------------------------------------------
suggested = []

for _, r in coverage.iterrows():
    M = float(r["Mach"])
    amin = float(r["alpha_min"])
    amax = float(r["alpha_max"])
    maxgap = float(r["max_alpha_gap"]) if pd.notna(r["max_alpha_gap"]) else np.nan

    if M <= 1.3 and amax < 0.325:
        suggested.append({
            "Mach": M,
            "alpha": 0.325,
            "priority": "medium",
            "reason": "extend_high_alpha_if_needed",
        })

    if pd.notna(maxgap) and maxgap > 0.06:
        alphas = np.sort(spec_final[np.isclose(spec_final["Mach"], M)]["alpha"].astype(float).unique())
        for a0, a1 in zip(alphas[:-1], alphas[1:]):
            if a1 - a0 > 0.06:
                suggested.append({
                    "Mach": M,
                    "alpha": float(round(0.5 * (a0 + a1), 5)),
                    "priority": "medium",
                    "reason": f"large_gap_{a1-a0:.3f}",
                })

suggested_df = pd.DataFrame(suggested).drop_duplicates() if suggested else pd.DataFrame(
    columns=["Mach", "alpha", "priority", "reason"]
)

# -------------------------------------------------------------------
# Plot updated Blumen overlay.
# -------------------------------------------------------------------
curves = load_blumen_curves(BLUMEN_CI)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=180)

for ax in axes:
    for x, y in curves:
        ax.plot(x, y, color="0.72", linewidth=1.0, alpha=0.65)

status_markers = {
    "legacy": "D",
    "M18_M19": "*",
    "smallM": "o",
}

machs = sorted(spec_final["Mach"].dropna().unique())
cmap = plt.get_cmap("tab20", max(len(machs), 1))
colors = {m: cmap(i) for i, m in enumerate(machs)}

for ax in axes:
    for M in machs:
        g = spec_final[np.isclose(spec_final["Mach"], M)].copy()

        legacy = g[g["reference_role"].astype(str).str.contains("base_legacy", na=False)]
        highM = g[g["reference_role"].astype(str).str.contains("M18_M19", na=False)]
        smallM = g[g["reference_role"].astype(str).str.contains("smallM", na=False)]

        if len(legacy):
            ax.scatter(
                legacy["Mach"], legacy["alpha"],
                s=50, marker="D", color=colors[M],
                edgecolors="black", linewidths=0.45,
                zorder=3,
            )

        if len(highM):
            ax.scatter(
                highM["Mach"], highM["alpha"],
                s=85, marker="*", color=colors[M],
                edgecolors="black", linewidths=0.5,
                zorder=4,
            )

        if len(smallM):
            boundary = smallM["validation_status"].astype(str).str.contains("boundary_flag", na=False)
            ax.scatter(
                smallM.loc[~boundary, "Mach"], smallM.loc[~boundary, "alpha"],
                s=55, marker="o", color=colors[M],
                edgecolors="black", linewidths=0.45,
                zorder=5,
            )
            ax.scatter(
                smallM.loc[boundary, "Mach"], smallM.loc[boundary, "alpha"],
                s=75, marker="X", color=colors[M],
                edgecolors="black", linewidths=0.6,
                zorder=6,
            )

    ax.set_xlabel("Mach M")
    ax.set_ylabel(r"$\alpha$")
    ax.grid(True, alpha=0.25, linestyle=":")

axes[0].set_title("Blumen ci digitise + sparse PINN reference v2")
axes[1].set_title("Zoom points reference v2")

axes[0].set_xlim(0.85, 2.12)
axes[0].set_ylim(0.0, max(0.48, float(spec_final["alpha"].max()) + 0.03))

axes[1].set_xlim(float(spec_final["Mach"].min()) - 0.03, float(spec_final["Mach"].max()) + 0.03)
axes[1].set_ylim(0.035, float(spec_final["alpha"].max()) + 0.03)

counts = "\n".join(
    f"M={M:.2f}: {len(spec_final[np.isclose(spec_final['Mach'], M)])}"
    for M in machs
)
axes[1].text(
    0.98, 0.03, counts,
    transform=axes[1].transAxes,
    ha="right", va="bottom",
    fontsize=8,
    bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.88),
)

# Custom legend.
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="D", color="w", markerfacecolor="0.5", markeredgecolor="black", label="legacy/base", markersize=7),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="0.5", markeredgecolor="black", label="M=1.8/1.9 core-stable", markersize=10),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5", markeredgecolor="black", label="small-M campaign", markersize=7),
    Line2D([0], [0], marker="X", color="w", markerfacecolor="0.5", markeredgecolor="black", label="small-M boundary flag", markersize=8),
]
axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right")

fig.tight_layout()

# -------------------------------------------------------------------
# Write outputs.
# -------------------------------------------------------------------
spec_out = OUT / "supersonic_sparse_PINN_reference_v2_spectral.csv"
fields_out = OUT / "supersonic_sparse_PINN_reference_v2_modal_fields.csv"
coverage_out = OUT / "coverage_by_Mach_v2.csv"
suggested_out = OUT / "suggested_remaining_targets_v2.csv"
overlay_png = OUT / "blumen_ci_overlay_sparse_PINN_reference_v2.png"
overlay_pdf = OUT / "blumen_ci_overlay_sparse_PINN_reference_v2.pdf"

spec_final.to_csv(spec_out, index=False)
fields_final.to_csv(fields_out, index=False)
coverage.to_csv(coverage_out, index=False)
suggested_df.to_csv(suggested_out, index=False)
fig.savefig(overlay_png, bbox_inches="tight")
fig.savefig(overlay_pdf, bbox_inches="tight")
plt.close(fig)

summary = {
    "status": "final_sparse_PINN_reference_v2_built",
    "n_total_spectral_points": int(spec_final[["Mach", "alpha"]].drop_duplicates().shape[0]),
    "n_base_points_before_campaign": int(base_spec[["Mach", "alpha"]].drop_duplicates().shape[0]),
    "n_smallM_campaign_points_added": int(new_campaign[["Mach", "alpha"]].drop_duplicates().shape[0]),
    "n_total_modal_rows": int(len(fields_final)),
    "point_counts_by_Mach": spec_final.groupby("Mach").size().to_dict(),
    "validation_status_counts": spec_final["validation_status"].value_counts(dropna=False).to_dict(),
    "boundary_flag_points": int(new_campaign["boundary_flag"].sum()),
    "outputs": {
        "spectral": str(spec_out),
        "modal_fields": str(fields_out),
        "coverage": str(coverage_out),
        "suggested_remaining_targets": str(suggested_out),
        "overlay_png": str(overlay_png),
        "overlay_pdf": str(overlay_pdf),
    },
    "important_note": (
        "Small-M campaign modes were visually accepted. Boundary-flagged points hit ci=0.12 or cr=0 in the initial scan "
        "and should be rescanned later, but are kept with explicit flags for sparse PINN reference experiments."
    ),
}

(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
