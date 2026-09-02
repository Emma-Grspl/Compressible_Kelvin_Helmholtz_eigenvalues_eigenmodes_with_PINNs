#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("assets/pinn_subsonic")
INPUT = (
    ROOT
    / "local_atlas_v1/gep_core_atlas_ci_seeded_v2/"
    "near_neutral_ci_omega_audit.csv"
)

BRANCH_INPUT = (
    ROOT
    / "joint_ci_mode_full_gep_atlas_v2/"
    "near_neutral_refinement/near_neutral_all_54.csv"
)

OUT = ROOT / "paper_results_v1"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )


def closest_values(
    available: np.ndarray,
    targets: list[float],
) -> list[float]:
    selected: list[float] = []

    for target in targets:
        value = float(available[np.argmin(np.abs(available - target))])
        if value not in selected:
            selected.append(value)

    return selected


df = pd.read_csv(INPUT)
df = df[as_bool(df["success"])].copy()
df = df[df["eta"] >= 0.75].copy()

mach_values = np.sort(df["Mach"].unique())
selected_mach = closest_values(
    mach_values,
    [0.1, 0.5, 0.9],
)

fig, axes = plt.subplots(
    1,
    len(selected_mach),
    figsize=(4.2 * len(selected_mach), 3.8),
    sharey=True,
)

if len(selected_mach) == 1:
    axes = [axes]

for axis, mach in zip(axes, selected_mach):
    local = (
        df[np.isclose(df["Mach"], mach)]
        .sort_values("eta")
    )

    axis.plot(
        local["eta"],
        local["ci_classic"],
        linewidth=2.2,
        label="Classical",
    )
    axis.plot(
        local["eta"],
        local["ci_seed"],
        linewidth=1.5,
        label="Direct PINN",
    )
    axis.plot(
        local["eta"],
        local["gep_ci"],
        linewidth=1.5,
        label="PINN-seeded GEP",
    )

    axis.set_xlabel(r"$\eta$")
    axis.set_title(fr"$M={mach:.2f}$")
    axis.grid(alpha=0.25)

axes[0].set_ylabel(r"$c_i$")
axes[-1].legend(fontsize=8)

fig.tight_layout()

for extension in ("pdf", "png"):
    fig.savefig(
        FIG_DIR / f"Fig_near_neutral_growth_rate_cuts.{extension}",
        dpi=300,
        bbox_inches="tight",
    )

plt.close(fig)


# Error scaling versus distance to neutral boundary
df["distance_to_neutral"] = 1.0 - df["eta"]
df["direct_abs_error"] = np.abs(df["ci_seed"] - df["ci_classic"])
df["gep_abs_error"] = np.abs(df["gep_ci"] - df["ci_classic"])

bins = np.geomspace(
    max(df["distance_to_neutral"].min(), 1.0e-4),
    df["distance_to_neutral"].max(),
    10,
)

df["distance_bin"] = pd.cut(
    df["distance_to_neutral"],
    bins=np.unique(bins),
    include_lowest=True,
)

rows = []

for interval, group in df.groupby(
    "distance_bin",
    observed=True,
):
    if group.empty:
        continue

    rows.append(
        {
            "distance_bin": str(interval),
            "distance_center": float(
                np.sqrt(
                    group["distance_to_neutral"].min()
                    * group["distance_to_neutral"].max()
                )
            ),
            "n": len(group),
            "direct_median": group["direct_abs_error"].median(),
            "direct_p90": group["direct_abs_error"].quantile(0.90),
            "gep_median": group["gep_abs_error"].median(),
            "gep_p90": group["gep_abs_error"].quantile(0.90),
        }
    )

binned = pd.DataFrame(rows)
binned.to_csv(
    TABLE_DIR / "Table_near_neutral_error_scaling.csv",
    index=False,
)

fig, axis = plt.subplots(figsize=(6.4, 4.3))

axis.plot(
    binned["distance_center"],
    binned["direct_median"],
    marker="o",
    label="Direct PINN: median",
)
axis.plot(
    binned["distance_center"],
    binned["direct_p90"],
    marker="o",
    linestyle="--",
    label="Direct PINN: 90th percentile",
)
axis.plot(
    binned["distance_center"],
    binned["gep_median"],
    marker="s",
    label="PINN-seeded GEP: median",
)
axis.plot(
    binned["distance_center"],
    binned["gep_p90"],
    marker="s",
    linestyle="--",
    label="PINN-seeded GEP: 90th percentile",
)

axis.set_xscale("log")
axis.set_yscale("log")
axis.set_xlabel(r"Distance to the neutral boundary, $1-\eta$")
axis.set_ylabel(r"$|c_i-c_i^{\mathrm{class}}|$")
axis.set_title("Spectral error near the neutral boundary")
axis.grid(alpha=0.25)
axis.legend(fontsize=8)

fig.tight_layout()

for extension in ("pdf", "png"):
    fig.savefig(
        FIG_DIR / f"Fig_near_neutral_error_scaling.{extension}",
        dpi=300,
        bbox_inches="tight",
    )

plt.close(fig)


# Branch-robustness summary
branch = pd.read_csv(BRANCH_INPUT)

boolean_columns = [
    "technical_success",
    "reference_success",
    "central_branch_found",
    "central_mode_supported_by_pinn",
    "previous_pinn_match_is_central_max",
    "pinn_matched_is_most_unstable",
]

summary_rows = []

for column in boolean_columns:
    if column not in branch:
        continue

    values = as_bool(branch[column])

    summary_rows.append(
        {
            "metric": column,
            "n": int(values.size),
            "value": float(values.mean()),
            "interpretation": "success_fraction",
        }
    )

numeric_metrics = [
    "ci_pinn_abs_err_classic",
    "pinn_matched_ci_abs_err_classic",
    "central_ci_abs_err_classic",
    "pinn_matched_p_overlap_classic",
    "central_p_overlap_pinn",
]

for column in numeric_metrics:
    if column not in branch:
        continue

    values = pd.to_numeric(branch[column], errors="coerce").dropna()

    summary_rows.extend(
        [
            {
                "metric": f"{column}_median",
                "n": int(values.size),
                "value": float(values.median()),
                "interpretation": "median",
            },
            {
                "metric": f"{column}_p95",
                "n": int(values.size),
                "value": float(values.quantile(0.95)),
                "interpretation": "95th_percentile",
            },
        ]
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    TABLE_DIR / "Table_near_neutral_branch_robustness.csv",
    index=False,
)

print("Selected Mach cuts:", selected_mach)
print("\nBranch summary:")
print(summary.to_string(index=False))
