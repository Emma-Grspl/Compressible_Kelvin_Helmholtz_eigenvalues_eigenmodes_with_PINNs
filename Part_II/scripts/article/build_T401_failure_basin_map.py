from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_DIR = (
    ROOT
    / "results"
    / "failure_basin"
    / "analysis"
)

BASIN_CSV = ANALYSIS_DIR / "phase8_basin_all_points.csv"
PINN_CSV = ANALYSIS_DIR / "phase8_nearest_pinn_seed.csv"

OUT_DIR = ROOT / "assets" / "complementary" / "failure_basin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Fig_T401_failure_basin_map.png"
OUT_PDF = OUT_DIR / "Fig_T401_failure_basin_map.pdf"


# =============================================================================
# Helpers
# =============================================================================

def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "success"})
    )


def classify_roots(df: pd.DataFrame) -> pd.Series:
    """
    Return one of:
        target
        failure
        other

    Prefer the stored root classification. Fall back to the explicit
    target/failure distances if needed.
    """

    result = pd.Series("other", index=df.index, dtype=object)

    # Target classification
    if "target_recovery_recomputed" in df.columns:
        target = as_bool(df["target_recovery_recomputed"])
        result.loc[target] = "target"

    # Stored textual classifications
    for col in ["root_class", "worker_class"]:
        if col in df.columns:
            text = (
                df[col]
                .fillna("")
                .astype(str)
                .str.lower()
            )

            result.loc[
                text.str.contains("target", regex=False)
            ] = "target"

            failure_mask = (
                text.str.contains("failure", regex=False)
                | text.str.contains("historical", regex=False)
            )

            result.loc[
                failure_mask & (result != "target")
            ] = "failure"

    # Numerical fallback for the historical competing root
    if "failure_branch_error" in df.columns:
        err = pd.to_numeric(
            df["failure_branch_error"],
            errors="coerce",
        )

        result.loc[
            (err <= 1e-4) & (result != "target")
        ] = "failure"

    return result


def median_spacing(values):
    values = np.sort(np.unique(np.asarray(values, dtype=float)))

    if len(values) < 2:
        return 1.0

    return float(np.median(np.diff(values)))


# =============================================================================
# Load and validate
# =============================================================================

df = pd.read_csv(BASIN_CSV)
nearest = pd.read_csv(PINN_CSV)

required = [
    "Mach",
    "alpha",
    "cr0",
    "ci0",
    "technical_success",
    "cr_target",
    "ci_target",
    "cr_pinn",
    "ci_pinn",
    "cr_failure_reference",
    "ci_failure_reference",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

if len(df) != 775:
    raise RuntimeError(
        f"Expected 775 basin initializations, found {len(df)}."
    )

df["technical"] = as_bool(df["technical_success"])
df["basin_class"] = classify_roots(df)

cr_values = np.sort(df["cr0"].unique())
ci_values = np.sort(df["ci0"].unique())

if len(cr_values) != 25:
    raise RuntimeError(
        f"Expected 25 unique cr0 values, found {len(cr_values)}."
    )

if len(ci_values) != 31:
    raise RuntimeError(
        f"Expected 31 unique ci0 values, found {len(ci_values)}."
    )

if len(cr_values) * len(ci_values) != len(df):
    raise RuntimeError(
        "The basin data do not form the expected complete 25 x 31 grid."
    )


# =============================================================================
# Basic quantities
# =============================================================================

M = float(df["Mach"].iloc[0])
alpha = float(df["alpha"].iloc[0])

cr_target = float(df["cr_target"].iloc[0])
ci_target = float(df["ci_target"].iloc[0])

cr_failure = float(df["cr_failure_reference"].iloc[0])
ci_failure = float(df["ci_failure_reference"].iloc[0])

cr_pinn = float(df["cr_pinn"].iloc[0])
ci_pinn = float(df["ci_pinn"].iloc[0])

n = len(df)

n_technical = int(df["technical"].sum())
n_target = int((df["basin_class"] == "target").sum())
n_failure = int((df["basin_class"] == "failure").sum())
n_other = int((df["basin_class"] == "other").sum())

target_df = df[df["basin_class"] == "target"]
failure_df = df[df["basin_class"] == "failure"]

if target_df.empty:
    raise RuntimeError("No target-basin points detected.")

if failure_df.empty:
    raise RuntimeError("No historical failure-basin points detected.")


# =============================================================================
# Basin ranges
# =============================================================================

target_bounds = {
    "cr_min": target_df["cr0"].min(),
    "cr_max": target_df["cr0"].max(),
    "ci_min": target_df["ci0"].min(),
    "ci_max": target_df["ci0"].max(),
}

failure_bounds = {
    "cr_min": failure_df["cr0"].min(),
    "cr_max": failure_df["cr0"].max(),
    "ci_min": failure_df["ci0"].min(),
    "ci_max": failure_df["ci0"].max(),
}

dcr = median_spacing(cr_values)
dci = median_spacing(ci_values)


# =============================================================================
# Grid for panel (a)
# =============================================================================

class_to_int = {
    "other": 0,
    "failure": 1,
    "target": 2,
}

grid = np.full(
    (len(ci_values), len(cr_values)),
    np.nan,
)

cr_index = {v: i for i, v in enumerate(cr_values)}
ci_index = {v: i for i, v in enumerate(ci_values)}

for _, row in df.iterrows():
    i = ci_index[row["ci0"]]
    j = cr_index[row["cr0"]]
    grid[i, j] = class_to_int[row["basin_class"]]


def edges(values):
    values = np.asarray(values, dtype=float)

    mids = 0.5 * (values[:-1] + values[1:])

    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])

    return np.concatenate(
        [[first], mids, [last]]
    )


cr_edges = edges(cr_values)
ci_edges = edges(ci_values)


# =============================================================================
# Marginal recovery fractions for panel (b)
# =============================================================================

summary_ci = []

for ci0, group in df.groupby("ci0", sort=True):
    summary_ci.append(
        {
            "ci0": ci0,
            "target_fraction":
                np.mean(group["basin_class"] == "target"),
            "failure_fraction":
                np.mean(group["basin_class"] == "failure"),
            "other_fraction":
                np.mean(group["basin_class"] == "other"),
        }
    )

summary_ci = pd.DataFrame(summary_ci)


# =============================================================================
# Console audit
# =============================================================================

print("=" * 78)
print("T401 FAILURE-BASIN MAP")
print("=" * 78)

print(f"Coordinate                : M={M:.5g}, alpha={alpha:.5g}")
print(f"Grid                      : {len(cr_values)} x {len(ci_values)} = {n}")
print(f"Technical convergence     : {n_technical}/{n}")
print(f"Target basin              : {n_target}/{n} ({n_target/n:.2%})")
print(f"Failure basin             : {n_failure}/{n} ({n_failure/n:.2%})")
print(f"Other roots               : {n_other}/{n} ({n_other/n:.2%})")

print()
print(
    "Target basin bounds      : "
    f"cr0=[{target_bounds['cr_min']:.6f}, "
    f"{target_bounds['cr_max']:.6f}], "
    f"ci0=[{target_bounds['ci_min']:.6f}, "
    f"{target_bounds['ci_max']:.6f}]"
)

print(
    "Failure basin bounds     : "
    f"cr0=[{failure_bounds['cr_min']:.6f}, "
    f"{failure_bounds['cr_max']:.6f}], "
    f"ci0=[{failure_bounds['ci_min']:.6f}, "
    f"{failure_bounds['ci_max']:.6f}]"
)

print()
print(
    "Target eigenvalue        : "
    f"{cr_target:.8f} + {ci_target:.8f} i"
)

print(
    "Failure eigenvalue       : "
    f"{cr_failure:.8f} + {ci_failure:.8f} i"
)

print(
    "Historical PINN seed     : "
    f"{cr_pinn:.8f} + {ci_pinn:.8f} i"
)


# =============================================================================
# Publication style
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.7,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# =============================================================================
# Colors
# =============================================================================

COLOR_OTHER = "#D9D9D9"
COLOR_FAILURE = "#D55E00"
COLOR_TARGET = "#0072B2"
COLOR_PINN = "#000000"

cmap = ListedColormap(
    [
        COLOR_OTHER,
        COLOR_FAILURE,
        COLOR_TARGET,
    ]
)

norm = BoundaryNorm(
    [-0.5, 0.5, 1.5, 2.5],
    cmap.N,
)


# =============================================================================
# Figure
# =============================================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(10.2, 4.05),
    gridspec_kw={
        "width_ratios": [1.15, 0.85],
    },
)


# -----------------------------------------------------------------------------
# Panel (a): basin map
# -----------------------------------------------------------------------------

ax1.pcolormesh(
    cr_edges,
    ci_edges,
    grid,
    cmap=cmap,
    norm=norm,
    shading="flat",
    rasterized=True,
)

# Outline target basin
target_rect = Rectangle(
    (
        target_bounds["cr_min"] - dcr / 2,
        target_bounds["ci_min"] - dci / 2,
    ),
    (
        target_bounds["cr_max"]
        - target_bounds["cr_min"]
        + dcr
    ),
    (
        target_bounds["ci_max"]
        - target_bounds["ci_min"]
        + dci
    ),
    fill=False,
    edgecolor=COLOR_TARGET,
    linewidth=1.4,
    linestyle="--",
)

ax1.add_patch(target_rect)

# Outline competing basin
failure_rect = Rectangle(
    (
        failure_bounds["cr_min"] - dcr / 2,
        failure_bounds["ci_min"] - dci / 2,
    ),
    (
        failure_bounds["cr_max"]
        - failure_bounds["cr_min"]
        + dcr
    ),
    (
        failure_bounds["ci_max"]
        - failure_bounds["ci_min"]
        + dci
    ),
    fill=False,
    edgecolor=COLOR_FAILURE,
    linewidth=1.4,
    linestyle="--",
)

ax1.add_patch(failure_rect)

# Historical PINN seed
ax1.scatter(
    cr_pinn,
    ci_pinn,
    marker="*",
    s=105,
    c=COLOR_PINN,
    edgecolors="white",
    linewidths=0.6,
    zorder=10,
)

# Target eigenvalue
ax1.scatter(
    cr_target,
    ci_target,
    marker="o",
    s=48,
    facecolors="none",
    edgecolors=COLOR_TARGET,
    linewidths=1.6,
    zorder=9,
)

# Historical competing eigenvalue
ax1.scatter(
    cr_failure,
    ci_failure,
    marker="o",
    s=48,
    facecolors="none",
    edgecolors=COLOR_FAILURE,
    linewidths=1.6,
    zorder=9,
)

ax1.set_xlabel(
    r"Initial real component $c_{r,0}$"
)

ax1.set_ylabel(
    r"Initial imaginary component $c_{i,0}$"
)

ax1.set_title(
    "(a) Local-search basin structure",
    loc="left",
    pad=8,
)

ax1.set_xlim(
    cr_edges[0],
    cr_edges[-1],
)

ax1.set_ylim(
    ci_edges[0],
    ci_edges[-1],
)

ax1.set_axisbelow(True)


# -----------------------------------------------------------------------------
# Panel (b): fraction versus ci0
# -----------------------------------------------------------------------------

ax2.plot(
    summary_ci["ci0"],
    summary_ci["target_fraction"],
    marker="o",
    markersize=4.0,
    label="Target branch",
    color=COLOR_TARGET,
)

ax2.plot(
    summary_ci["ci0"],
    summary_ci["failure_fraction"],
    marker="s",
    markersize=3.8,
    label="Competing branch",
    color=COLOR_FAILURE,
)

ax2.axvline(
    ci_pinn,
    color=COLOR_PINN,
    linestyle="--",
    linewidth=1.2,
)

ax2.text(
    ci_pinn + 0.002,
    0.96,
    r"PINN $c_{i,0}$",
    rotation=90,
    va="top",
    ha="left",
    fontsize=8.5,
)

ax2.set_xlabel(
    r"Initial imaginary component $c_{i,0}$"
)

ax2.set_ylabel(
    "Fraction of initializations"
)

ax2.yaxis.set_major_formatter(
    PercentFormatter(xmax=1.0)
)

ax2.set_ylim(
    0,
    1.02,
)

ax2.set_title(
    r"(b) Basin occupancy versus $c_{i,0}$",
    loc="left",
    pad=8,
)

ax2.grid(
    axis="y",
    linestyle=":",
    linewidth=0.65,
    alpha=0.55,
)

ax2.set_axisbelow(True)


# =============================================================================
# Shared legend
# =============================================================================

legend_handles = [
    Patch(
        facecolor=COLOR_TARGET,
        edgecolor="none",
        label=f"Target basin ({n_target}/{n})",
    ),
    Patch(
        facecolor=COLOR_FAILURE,
        edgecolor="none",
        label=f"Competing basin ({n_failure}/{n})",
    ),
    Patch(
        facecolor=COLOR_OTHER,
        edgecolor="none",
        label=f"Other roots ({n_other}/{n})",
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        linestyle="none",
        markerfacecolor=COLOR_PINN,
        markeredgecolor="white",
        markersize=10,
        label="Historical PINN seed",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor=COLOR_TARGET,
        markeredgewidth=1.5,
        markersize=6,
        label="Target eigenvalue",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor=COLOR_FAILURE,
        markeredgewidth=1.5,
        markersize=6,
        label="Competing eigenvalue",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=3,
    frameon=False,
    columnspacing=1.7,
    handletextpad=0.6,
)


# =============================================================================
# Final layout
# =============================================================================

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    top=0.90,
    bottom=0.25,
    wspace=0.32,
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
print(f"Saved PNG : {OUT_PNG}")
print(f"Saved PDF : {OUT_PDF}")
