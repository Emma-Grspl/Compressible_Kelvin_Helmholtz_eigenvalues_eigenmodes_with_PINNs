from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

ROUTING_DIR = (
    ROOT
    / "results"
    / "routing_audit"
    / "results"
    / "routing_boundary_audit"
)

RESULTS_CSV = ROUTING_DIR / "routing_boundary_results.csv"
BOUNDARIES_CSV = ROUTING_DIR / "routing_boundaries.csv"
EPS_CSV = ROUTING_DIR / "routing_summary_by_eps.csv"
SUMMARY_JSON = ROUTING_DIR / "routing_summary.json"

OUT_DIR = ROOT / "assets" / "complementary" / "routing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Fig_routing_boundary_discontinuity.png"
OUT_PDF = OUT_DIR / "Fig_routing_boundary_discontinuity.pdf"

OUT_PAIR_SUMMARY = OUT_DIR / "routing_boundary_by_pair_summary.csv"
OUT_EPS_SUMMARY = OUT_DIR / "routing_boundary_eps_summary_for_plot.csv"


# =============================================================================
# Load
# =============================================================================

results = pd.read_csv(RESULTS_CSV)
boundaries = pd.read_csv(BOUNDARIES_CSV)
eps_summary = pd.read_csv(EPS_CSV)

with open(SUMMARY_JSON) as f:
    global_summary = json.load(f)


# =============================================================================
# Validate structure
# =============================================================================

if len(boundaries) != 85:
    raise RuntimeError(
        f"Expected 85 representative routing boundaries, "
        f"found {len(boundaries)}."
    )

n_pairs = boundaries["pair_id"].nunique()

if n_pairs != 17:
    raise RuntimeError(
        f"Expected 17 unique neighboring-chart interfaces, "
        f"found {n_pairs}."
    )

if len(results) != 255:
    raise RuntimeError(
        f"Expected 255 routed evaluations (=85 x 3 eps), "
        f"found {len(results)}."
    )

eps_values = np.sort(
    pd.to_numeric(
        results["eps"],
        errors="coerce",
    ).dropna().unique()
)

if len(eps_values) != 3:
    raise RuntimeError(
        f"Expected 3 epsilon values, found {eps_values}."
    )

if not np.allclose(
    eps_values,
    np.array([1e-6, 1e-5, 1e-4]),
    rtol=1e-8,
    atol=0,
):
    print(
        "WARNING: epsilon values differ from "
        f"[1e-6, 1e-5, 1e-4]: {eps_values}"
    )


# =============================================================================
# Same-point data
#
# Same-point values are duplicated once per epsilon in routing_boundary_results.
# Keep one row per physical boundary and verify consistency across eps.
# =============================================================================

samepoint_columns = [
    "samepoint_cr_abs_diff",
    "samepoint_ci_abs_diff",
    "samepoint_c_abs_diff",
]

for col in samepoint_columns:
    spread = (
        results
        .groupby("boundary_id")[col]
        .agg(lambda x: np.nanmax(x) - np.nanmin(x))
    )

    if np.nanmax(spread.to_numpy()) > 1e-12:
        raise RuntimeError(
            f"{col} is not invariant across epsilon for a boundary."
        )

samepoint = (
    results
    .sort_values(["boundary_id", "eps"])
    .drop_duplicates(subset=["boundary_id"])
    .copy()
)

if len(samepoint) != 85:
    raise RuntimeError(
        f"Expected 85 unique same-point rows, found {len(samepoint)}."
    )


# =============================================================================
# Pair-wise summary
# =============================================================================

pair_rows = []

for pair_id, group in samepoint.groupby("pair_id"):
    values = group["samepoint_c_abs_diff"].to_numpy(dtype=float)

    pair_rows.append(
        {
            "pair_id": pair_id,
            "n": len(group),
            "median_c_jump": np.median(values),
            "p95_c_jump": np.quantile(values, 0.95),
            "max_c_jump": np.max(values),
            "median_cr_jump": np.median(
                group["samepoint_cr_abs_diff"]
            ),
            "median_ci_jump": np.median(
                group["samepoint_ci_abs_diff"]
            ),
        }
    )

pair_summary = pd.DataFrame(pair_rows)

pair_summary = pair_summary.sort_values(
    "median_c_jump",
    ascending=True,
).reset_index(drop=True)

pair_summary.to_csv(
    OUT_PAIR_SUMMARY,
    index=False,
)


# =============================================================================
# Global same-point statistics
# =============================================================================

same_stats = {
    "c": {
        "median": float(
            np.median(samepoint["samepoint_c_abs_diff"])
        ),
        "p95": float(
            np.quantile(
                samepoint["samepoint_c_abs_diff"],
                0.95,
            )
        ),
        "max": float(
            np.max(samepoint["samepoint_c_abs_diff"])
        ),
    },
    "cr": {
        "median": float(
            np.median(samepoint["samepoint_cr_abs_diff"])
        ),
        "p95": float(
            np.quantile(
                samepoint["samepoint_cr_abs_diff"],
                0.95,
            )
        ),
        "max": float(
            np.max(samepoint["samepoint_cr_abs_diff"])
        ),
    },
    "ci": {
        "median": float(
            np.median(samepoint["samepoint_ci_abs_diff"])
        ),
        "p95": float(
            np.quantile(
                samepoint["samepoint_ci_abs_diff"],
                0.95,
            )
        ),
        "max": float(
            np.max(samepoint["samepoint_ci_abs_diff"])
        ),
    },
}


# =============================================================================
# Build unified epsilon summary:
# "same point" + routed eps=1e-6,1e-5,1e-4
# =============================================================================

plot_rows = [
    {
        "location": "same point",
        "eps": 0.0,
        "c_median": same_stats["c"]["median"],
        "c_p95": same_stats["c"]["p95"],
        "cr_median": same_stats["cr"]["median"],
        "cr_p95": same_stats["cr"]["p95"],
        "ci_median": same_stats["ci"]["median"],
        "ci_p95": same_stats["ci"]["p95"],
    }
]

for _, row in eps_summary.sort_values("eps").iterrows():
    plot_rows.append(
        {
            "location": f"{row['eps']:.0e}",
            "eps": float(row["eps"]),
            "c_median": float(row["routed_c_median"]),
            "c_p95": float(row["routed_c_p95"]),
            "cr_median": float(row["routed_cr_median"]),
            "cr_p95": float(row["routed_cr_p95"]),
            "ci_median": float(row["routed_ci_median"]),
            "ci_p95": float(row["routed_ci_p95"]),
        }
    )

plot_summary = pd.DataFrame(plot_rows)

plot_summary.to_csv(
    OUT_EPS_SUMMARY,
    index=False,
)


# =============================================================================
# Worst boundary
# =============================================================================

worst = samepoint.loc[
    samepoint["samepoint_c_abs_diff"].idxmax()
]

worst_pair = str(worst["pair_id"])
worst_M = float(worst["Mach_boundary"])
worst_alpha = float(worst["alpha_boundary"])
worst_jump = float(worst["samepoint_c_abs_diff"])


# =============================================================================
# Numerical audit
# =============================================================================

print("=" * 86)
print("ROUTING-BOUNDARY DISCONTINUITY AUDIT")
print("=" * 86)

print(f"Representative boundaries   : {len(samepoint)}")
print(f"Neighboring chart pairs     : {n_pairs}")
print(f"Routed evaluations          : {len(results)}")
print(f"Epsilon values              : {eps_values}")

print()
print("SAME-POINT DISCREPANCIES")

for key, label in [
    ("c", "|Delta c| "),
    ("cr", "|Delta cr|"),
    ("ci", "|Delta ci|"),
]:
    s = same_stats[key]

    print(
        f"{label:12s} "
        f"median={s['median']:.6e}  "
        f"P95={s['p95']:.6e}  "
        f"max={s['max']:.6e}"
    )

print()
print("ROUTED JUMPS")

print(
    eps_summary[
        [
            "eps",
            "route_changes",
            "routed_c_median",
            "routed_c_p95",
            "routed_c_max",
            "routed_cr_median",
            "routed_ci_median",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6e}",
    )
)

print()
print(
    "Worst same-point seam       : "
    f"{worst_pair}, "
    f"M={worst_M:.6f}, "
    f"alpha={worst_alpha:.6f}, "
    f"|Delta c|={worst_jump:.6e}"
)


# =============================================================================
# Publication style
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9.3,
        "axes.labelsize": 9.7,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
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

COLOR_C = "#000000"
COLOR_CR = "#0072B2"
COLOR_CI = "#D55E00"

COLOR_POINTS = "0.68"
COLOR_MEDIAN = "#0072B2"
COLOR_WORST = "#D55E00"


# =============================================================================
# Figure
# =============================================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(10.6, 5.15),
    gridspec_kw={
        "width_ratios": [1.18, 0.82],
    },
)


# =============================================================================
# Panel (a) — same-point jumps across 17 interfaces
# =============================================================================

y_positions = np.arange(len(pair_summary))

for y, (_, row) in zip(
    y_positions,
    pair_summary.iterrows(),
):
    pair_id = row["pair_id"]

    group = samepoint[
        samepoint["pair_id"] == pair_id
    ]

    values = group[
        "samepoint_c_abs_diff"
    ].to_numpy(dtype=float)

    # Individual representative boundary points
    jitter = np.linspace(
        -0.09,
        0.09,
        len(values),
    )

    ax1.scatter(
        values,
        np.full(len(values), y) + jitter,
        s=18,
        facecolor=COLOR_POINTS,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )

    # Pair median
    ax1.scatter(
        row["median_c_jump"],
        y,
        marker="D",
        s=32,
        facecolor=COLOR_MEDIAN,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )


# Highlight worst actual boundary point
worst_y = int(
    pair_summary.index[
        pair_summary["pair_id"] == worst_pair
    ][0]
)

ax1.scatter(
    worst_jump,
    worst_y,
    marker="*",
    s=95,
    facecolor=COLOR_WORST,
    edgecolor="white",
    linewidth=0.6,
    zorder=6,
)

ax1.set_xscale("log")

ax1.set_yticks(y_positions)
ax1.set_yticklabels(
    pair_summary["pair_id"].astype(str)
)

ax1.set_xlabel(
    r"Same-point spectral discrepancy $|\Delta c|$"
)

ax1.set_ylabel(
    "Neighboring chart interface"
)

ax1.set_title(
    "(a) Same-point discontinuities across routing interfaces",
    loc="left",
    pad=8,
)

ax1.grid(
    axis="x",
    which="major",
    linestyle=":",
    linewidth=0.6,
    alpha=0.50,
)

ax1.grid(
    axis="x",
    which="minor",
    linestyle=":",
    linewidth=0.35,
    alpha=0.22,
)

ax1.set_axisbelow(True)

# Annotate worst seam compactly
ax1.annotate(
    (
        f"largest jump\n"
        rf"$|\Delta c|={worst_jump:.2e}$"
    ),
    xy=(worst_jump, worst_y),
    xytext=(8, 13),
    textcoords="offset points",
    fontsize=8.1,
    ha="left",
    va="bottom",
    arrowprops={
        "arrowstyle": "-",
        "linewidth": 0.7,
        "color": "0.35",
    },
)


# =============================================================================
# Panel (b) — routed jump versus epsilon
# =============================================================================

x = np.arange(len(plot_summary))

xlabels = [
    "same\npoint",
    r"$10^{-6}$",
    r"$10^{-5}$",
    r"$10^{-4}$",
]

series = [
    (
        r"$|\Delta c|$",
        "c_median",
        "c_p95",
        COLOR_C,
        "o",
    ),
    (
        r"$|\Delta c_r|$",
        "cr_median",
        "cr_p95",
        COLOR_CR,
        "s",
    ),
    (
        r"$|\Delta c_i|$",
        "ci_median",
        "ci_p95",
        COLOR_CI,
        "^",
    ),
]

for label, med_col, p95_col, color, marker in series:
    med = plot_summary[med_col].to_numpy(dtype=float)
    p95 = plot_summary[p95_col].to_numpy(dtype=float)

    # Median curve
    ax2.plot(
        x,
        med,
        marker=marker,
        markersize=5.2,
        linewidth=1.6,
        color=color,
        label=label,
        zorder=4,
    )

    # P95 as light vertical range from median to P95
    ax2.vlines(
        x,
        med,
        p95,
        color=color,
        linewidth=1.0,
        alpha=0.35,
        zorder=2,
    )

    ax2.scatter(
        x,
        p95,
        marker=marker,
        s=26,
        facecolor="white",
        edgecolor=color,
        linewidth=1.0,
        zorder=3,
    )


ax2.set_yscale("log")

ax2.set_xticks(x)
ax2.set_xticklabels(xlabels)

ax2.set_xlabel(
    r"Routing offset $\varepsilon$"
)

ax2.set_ylabel(
    "Spectral jump magnitude"
)

ax2.set_title(
    r"(b) Routed jump as $\varepsilon \rightarrow 0$",
    loc="left",
    pad=8,
)

ax2.grid(
    axis="y",
    which="major",
    linestyle=":",
    linewidth=0.6,
    alpha=0.50,
)

ax2.grid(
    axis="y",
    which="minor",
    linestyle=":",
    linewidth=0.35,
    alpha=0.22,
)

ax2.set_axisbelow(True)

ax2.legend(
    frameon=False,
    loc="best",
)


# =============================================================================
# Explanatory legend for marker semantics
# =============================================================================

semantic_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=COLOR_POINTS,
        markeredgecolor="white",
        markersize=5,
        label="Representative boundary point",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        linestyle="none",
        markerfacecolor=COLOR_MEDIAN,
        markeredgecolor="white",
        markersize=5.5,
        label="Interface median",
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        linestyle="none",
        markerfacecolor=COLOR_WORST,
        markeredgecolor="white",
        markersize=9,
        label="Largest observed seam",
    ),
]

fig.legend(
    handles=semantic_handles,
    loc="lower center",
    bbox_to_anchor=(0.34, 0.005),
    ncol=3,
    frameon=False,
    columnspacing=1.6,
    handletextpad=0.55,
)


# =============================================================================
# Small note for panel (b)
# =============================================================================

fig.text(
    0.98,
    0.017,
    "Filled symbols: median; open symbols: $P_{95}$",
    ha="right",
    va="bottom",
    fontsize=8.0,
    color="0.30",
)


# =============================================================================
# Layout
# =============================================================================

fig.subplots_adjust(
    left=0.12,
    right=0.975,
    top=0.925,
    bottom=0.155,
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
print(f"Saved PNG          : {OUT_PNG}")
print(f"Saved PDF          : {OUT_PDF}")
print(f"Saved pair summary : {OUT_PAIR_SUMMARY}")
print(f"Saved eps summary  : {OUT_EPS_SUMMARY}")
