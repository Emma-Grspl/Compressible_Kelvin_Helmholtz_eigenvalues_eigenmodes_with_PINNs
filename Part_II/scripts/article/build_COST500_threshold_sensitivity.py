from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import Normalize


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PHASE3 = ROOT / "results" / "threshold_sensitivity"

COMMON_CSV = PHASE3 / "threshold_scale_sensitivity.csv"
GRID_CSV = PHASE3 / "threshold_2D_sensitivity.csv"
QUERY_CSV = PHASE3 / "per_query_threshold_margin.csv"

OUT_DIR = ROOT / "assets" / "complementary" / "cost_thresholds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Fig_COST500_threshold_sensitivity.png"
OUT_PDF = OUT_DIR / "Fig_COST500_threshold_sensitivity.pdf"

OUT_SUMMARY = OUT_DIR / "COST500_threshold_sensitivity_summary.csv"


# =============================================================================
# Load
# =============================================================================

common = pd.read_csv(COMMON_CSV)
grid = pd.read_csv(GRID_CSV)
queries = pd.read_csv(QUERY_CSV)


# =============================================================================
# Validate
# =============================================================================

required_common = [
    "multiplier",
    "stage1_threshold",
    "stage2_threshold",
    "n_total",
    "n_stage1_success",
    "stage1_success_rate",
    "n_stage2_success",
    "stage2_success_rate",
    "n_technical_success",
    "technical_success_rate",
    "n_fail_stage1_only",
    "n_fail_stage2_only",
    "n_fail_both",
]

required_grid = [
    "stage1_multiplier",
    "stage2_multiplier",
    "stage1_threshold",
    "stage2_threshold",
    "n_technical_success",
    "technical_success_rate",
]

required_queries = [
    "benchmark_id",
    "shoot_stage1_mismatch",
    "shoot_stage2_mismatch",
    "stage1_fraction_of_nominal",
    "stage2_fraction_of_nominal",
    "critical_common_multiplier",
]

for name, df, required in [
    ("common", common, required_common),
    ("grid", grid, required_grid),
    ("queries", queries, required_queries),
]:
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{name}: missing columns {missing}"
        )


if len(queries) != 500:
    raise RuntimeError(
        f"Expected COST500 with 500 queries, found {len(queries)}."
    )


multipliers = np.sort(
    common["multiplier"].to_numpy(dtype=float)
)

stage1_multipliers = np.sort(
    grid["stage1_multiplier"].unique().astype(float)
)

stage2_multipliers = np.sort(
    grid["stage2_multiplier"].unique().astype(float)
)

if len(stage1_multipliers) * len(stage2_multipliers) != len(grid):
    raise RuntimeError(
        "The 2D sensitivity table is not a complete multiplier grid."
    )


# =============================================================================
# Sort tables
# =============================================================================

common = common.sort_values(
    "multiplier"
).reset_index(drop=True)

grid = grid.sort_values(
    ["stage1_multiplier", "stage2_multiplier"]
).reset_index(drop=True)


# =============================================================================
# Numerical audit
# =============================================================================

print("=" * 88)
print("COST500 TECHNICAL-CONVERGENCE THRESHOLD SENSITIVITY")
print("=" * 88)

print(f"Queries                    : {len(queries)}")
print(f"Common multipliers         : {list(common['multiplier'])}")
print(
    "Independent grid          : "
    f"{len(stage1_multipliers)} x {len(stage2_multipliers)}"
)

print()
print("COMMON THRESHOLD SCALING")
print(
    common[
        [
            "multiplier",
            "stage1_threshold",
            "stage2_threshold",
            "n_stage1_success",
            "n_stage2_success",
            "n_technical_success",
            "n_fail_stage1_only",
            "n_fail_stage2_only",
            "n_fail_both",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6e}",
    )
)

print()
print("2D TECHNICAL SUCCESS GRID")

pivot_counts = (
    grid.pivot(
        index="stage1_multiplier",
        columns="stage2_multiplier",
        values="n_technical_success",
    )
    .sort_index()
    .sort_index(axis=1)
)

print(pivot_counts.to_string())


# =============================================================================
# Save compact source table
# =============================================================================

summary_rows = []

for _, row in common.iterrows():
    summary_rows.append(
        {
            "mode": "common_multiplier",
            "stage1_multiplier": row["multiplier"],
            "stage2_multiplier": row["multiplier"],
            "stage1_threshold": row["stage1_threshold"],
            "stage2_threshold": row["stage2_threshold"],
            "n_total": row["n_total"],
            "n_stage1_success": row["n_stage1_success"],
            "n_stage2_success": row["n_stage2_success"],
            "n_technical_success": row["n_technical_success"],
            "technical_success_rate": row["technical_success_rate"],
        }
    )

for _, row in grid.iterrows():
    summary_rows.append(
        {
            "mode": "independent_multipliers",
            "stage1_multiplier": row["stage1_multiplier"],
            "stage2_multiplier": row["stage2_multiplier"],
            "stage1_threshold": row["stage1_threshold"],
            "stage2_threshold": row["stage2_threshold"],
            "n_total": 500,
            "n_stage1_success": np.nan,
            "n_stage2_success": np.nan,
            "n_technical_success": row["n_technical_success"],
            "technical_success_rate": row["technical_success_rate"],
        }
    )

pd.DataFrame(summary_rows).to_csv(
    OUT_SUMMARY,
    index=False,
)


# =============================================================================
# Publication style
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9.4,
        "axes.labelsize": 9.8,
        "axes.titlesize": 10.3,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.4,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# =============================================================================
# Colors
# =============================================================================

COLOR_STAGE1 = "#0072B2"
COLOR_STAGE2 = "#009E73"
COLOR_TECH = "#111111"
COLOR_NOMINAL = "#D55E00"


# =============================================================================
# Figure
# =============================================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(10.2, 4.25),
    gridspec_kw={
        "width_ratios": [1.0, 1.05],
    },
)


# =============================================================================
# Panel (a) — common scaling
# =============================================================================

x = common["multiplier"].to_numpy(dtype=float)

stage1_rate = common[
    "stage1_success_rate"
].to_numpy(dtype=float)

stage2_rate = common[
    "stage2_success_rate"
].to_numpy(dtype=float)

technical_rate = common[
    "technical_success_rate"
].to_numpy(dtype=float)


ax1.plot(
    x,
    stage1_rate,
    marker="o",
    markersize=5.5,
    color=COLOR_STAGE1,
    label="Stage 1",
)

ax1.plot(
    x,
    stage2_rate,
    marker="s",
    markersize=5.2,
    color=COLOR_STAGE2,
    label="Stage 2",
)

ax1.plot(
    x,
    technical_rate,
    marker="D",
    markersize=5.0,
    color=COLOR_TECH,
    label="Technical success",
)


# Nominal threshold scale
if np.any(np.isclose(x, 1.0)):
    ax1.axvline(
        1.0,
        linestyle="--",
        linewidth=1.1,
        color=COLOR_NOMINAL,
        alpha=0.8,
    )

    ax1.text(
        1.0,
        0.905,
        "nominal",
        rotation=90,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=COLOR_NOMINAL,
    )


ax1.set_xscale(
    "log",
    base=2,
)

ax1.set_xticks(x)
ax1.set_xticklabels(
    [f"{v:g}×" for v in x]
)

ax1.set_ylim(
    min(0.90, technical_rate.min() - 0.015),
    1.005,
)

ax1.yaxis.set_major_formatter(
    PercentFormatter(xmax=1.0)
)

ax1.set_xlabel(
    "Common threshold multiplier"
)

ax1.set_ylabel(
    "Successful COST500 queries"
)

ax1.set_title(
    "(a) Common scaling of both convergence thresholds",
    loc="left",
    pad=8,
)

ax1.grid(
    axis="y",
    linestyle=":",
    linewidth=0.6,
    alpha=0.50,
)

ax1.set_axisbelow(True)

ax1.legend(
    frameon=False,
    loc="lower right",
)


# Exact technical-success count labels
for xi, rate, count in zip(
    x,
    technical_rate,
    common["n_technical_success"].astype(int),
):
    ax1.annotate(
        f"{count}/500",
        xy=(xi, rate),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=7.8,
        color=COLOR_TECH,
    )


# =============================================================================
# Panel (b) — independent 2D sensitivity
# =============================================================================

rate_matrix = (
    grid.pivot(
        index="stage1_multiplier",
        columns="stage2_multiplier",
        values="technical_success_rate",
    )
    .reindex(
        index=stage1_multipliers,
        columns=stage2_multipliers,
    )
)

count_matrix = (
    grid.pivot(
        index="stage1_multiplier",
        columns="stage2_multiplier",
        values="n_technical_success",
    )
    .reindex(
        index=stage1_multipliers,
        columns=stage2_multipliers,
    )
)


# Dynamic normalization keeps subtle differences visible without
# implying a large physical range.
vmin = float(
    np.floor(
        rate_matrix.to_numpy().min() * 100
    ) / 100
)

vmax = 1.0

im = ax2.imshow(
    rate_matrix.to_numpy(),
    origin="lower",
    aspect="auto",
    cmap="Blues",
    norm=Normalize(
        vmin=vmin,
        vmax=vmax,
    ),
)


ax2.set_xticks(
    np.arange(len(stage2_multipliers))
)

ax2.set_xticklabels(
    [f"{v:g}×" for v in stage2_multipliers]
)

ax2.set_yticks(
    np.arange(len(stage1_multipliers))
)

ax2.set_yticklabels(
    [f"{v:g}×" for v in stage1_multipliers]
)

ax2.set_xlabel(
    "Stage 2 threshold multiplier"
)

ax2.set_ylabel(
    "Stage 1 threshold multiplier"
)

ax2.set_title(
    "(b) Independent Stage 1 / Stage 2 sensitivity",
    loc="left",
    pad=8,
)


# Annotate every cell with exact success count.
for i in range(len(stage1_multipliers)):
    for j in range(len(stage2_multipliers)):
        count = int(
            count_matrix.iloc[i, j]
        )

        rate = float(
            rate_matrix.iloc[i, j]
        )

        # High success -> dark cell, use white annotation.
        text_color = (
            "white"
            if rate > (vmin + vmax) / 2
            else "black"
        )

        ax2.text(
            j,
            i,
            f"{count}/500",
            ha="center",
            va="center",
            fontsize=8.4,
            color=text_color,
        )


# Mark nominal cell with a clear outline.
nominal_i = np.where(
    np.isclose(stage1_multipliers, 1.0)
)[0]

nominal_j = np.where(
    np.isclose(stage2_multipliers, 1.0)
)[0]

if len(nominal_i) == 1 and len(nominal_j) == 1:
    i = nominal_i[0]
    j = nominal_j[0]

    rect = plt.Rectangle(
        (j - 0.5, i - 0.5),
        1,
        1,
        fill=False,
        edgecolor=COLOR_NOMINAL,
        linewidth=2.0,
    )

    ax2.add_patch(rect)


# Thin cell separators
ax2.set_xticks(
    np.arange(-0.5, len(stage2_multipliers), 1),
    minor=True,
)

ax2.set_yticks(
    np.arange(-0.5, len(stage1_multipliers), 1),
    minor=True,
)

ax2.grid(
    which="minor",
    color="white",
    linewidth=1.0,
)

ax2.tick_params(
    which="minor",
    bottom=False,
    left=False,
)


# Color bar
cbar = fig.colorbar(
    im,
    ax=ax2,
    fraction=0.046,
    pad=0.035,
)

cbar.set_label(
    "Technical success rate",
    fontsize=9.0,
)

cbar.ax.yaxis.set_major_formatter(
    PercentFormatter(xmax=1.0)
)

cbar.ax.tick_params(
    labelsize=8.2,
)


# =============================================================================
# Global note
# =============================================================================

fig.text(
    0.5,
    0.025,
    (
        "Technical convergence only; COST500 does not contain "
        "target-branch labels."
    ),
    ha="center",
    va="bottom",
    fontsize=8.1,
    color="0.30",
)


# =============================================================================
# Layout
# =============================================================================

fig.subplots_adjust(
    left=0.09,
    right=0.96,
    top=0.91,
    bottom=0.17,
    wspace=0.34,
)


# =============================================================================
# Save
# =============================================================================

fig.savefig(
    OUT_PNG,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print()
print(f"Saved PNG     : {OUT_PNG}")
print(f"Saved PDF     : {OUT_PDF}")
print(f"Saved summary : {OUT_SUMMARY}")
