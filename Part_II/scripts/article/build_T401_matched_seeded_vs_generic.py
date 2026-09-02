from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_DIR = (
    ROOT
    / "results"
    / "matched_branch_localization"
    / "budget_matched"
    / "budget_matched_T401"
    / "analysis"
)

PAIRED_CSV = ANALYSIS_DIR / "phase7_paired_T401.csv"
SUMMARY_JSON = ANALYSIS_DIR / "phase7_summary.json"

OUT_DIR = ROOT / "assets" / "complementary" / "matched_branch"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Fig_T401_matched_seeded_vs_generic.png"
OUT_PDF = OUT_DIR / "Fig_T401_matched_seeded_vs_generic.pdf"


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


def ecdf(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Empty ECDF input.")

    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)

    return x, y


# =============================================================================
# Load
# =============================================================================

df = pd.read_csv(PAIRED_CSV)

with open(SUMMARY_JSON) as f:
    summary = json.load(f)

required = [
    "Mach_seeded",
    "alpha_seeded",
    "seeded_technical",
    "seeded_error",
    "seeded_recovery",
    "Mach_generic",
    "alpha_generic",
    "generic_technical",
    "generic_error",
    "generic_recovery",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

if len(df) != 401:
    raise RuntimeError(f"Expected 401 T401 rows, found {len(df)}.")

if not np.allclose(
    df["Mach_seeded"].to_numpy(),
    df["Mach_generic"].to_numpy(),
    atol=1e-12,
    rtol=0,
):
    raise RuntimeError("Seeded and generic Mach coordinates do not match.")

if not np.allclose(
    df["alpha_seeded"].to_numpy(),
    df["alpha_generic"].to_numpy(),
    atol=1e-12,
    rtol=0,
):
    raise RuntimeError("Seeded and generic alpha coordinates do not match.")


seeded_technical = as_bool(df["seeded_technical"])
generic_technical = as_bool(df["generic_technical"])

seeded_recovery = as_bool(df["seeded_recovery"])
generic_recovery = as_bool(df["generic_recovery"])

seeded_error = pd.to_numeric(
    df["seeded_error"],
    errors="coerce",
).to_numpy()

generic_error = pd.to_numeric(
    df["generic_error"],
    errors="coerce",
).to_numpy()

n = len(df)

n_seeded_technical = int(seeded_technical.sum())
n_generic_technical = int(generic_technical.sum())

n_seeded_recovery = int(seeded_recovery.sum())
n_generic_recovery = int(generic_recovery.sum())

target_tolerance = float(
    summary.get("target_tolerance", 1e-4)
)


# =============================================================================
# Numerical audit printed to terminal
# =============================================================================

print("=" * 72)
print("T401 MATCHED BRANCH-LOCALIZATION CONTROL")
print("=" * 72)

print(f"Paired coordinates         : {n}")
print(f"Seeded technical success   : {n_seeded_technical}/{n}")
print(f"Generic technical success  : {n_generic_technical}/{n}")
print(f"Seeded target recovery     : {n_seeded_recovery}/{n}")
print(f"Generic target recovery    : {n_generic_recovery}/{n}")

for label, values in [
    ("Seeded", seeded_error),
    ("Generic", generic_error),
]:
    values = values[np.isfinite(values)]

    print(
        f"{label:8s} "
        f"median={np.median(values):.6e}  "
        f"P95={np.quantile(values, 0.95):.6e}  "
        f"max={np.max(values):.6e}"
    )


# =============================================================================
# Publication style
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.7,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# =============================================================================
# Figure
# =============================================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(10.2, 3.8),
)

# -----------------------------------------------------------------------------
# Panel (a): technical convergence and target recovery
# -----------------------------------------------------------------------------

labels = [
    "Technical convergence",
    "Target-branch recovery",
]

y = np.arange(len(labels))

seeded_frac = np.array(
    [
        n_seeded_technical / n,
        n_seeded_recovery / n,
    ]
)

generic_frac = np.array(
    [
        n_generic_technical / n,
        n_generic_recovery / n,
    ]
)

height = 0.30

bars_seeded = ax1.barh(
    y - height / 2,
    seeded_frac,
    height=height,
    label="PINN-seeded",
)

bars_generic = ax1.barh(
    y + height / 2,
    generic_frac,
    height=height,
    label="Matched generic",
)

ax1.set_xlim(0, 1.08)
ax1.set_yticks(y)
ax1.set_yticklabels(labels)
ax1.invert_yaxis()

ax1.set_xlabel("Fraction of T401 coordinates")
ax1.xaxis.set_major_formatter(
    PercentFormatter(xmax=1.0)
)

ax1.set_title(
    "(a) Technical convergence versus branch recovery",
    loc="left",
    pad=8,
)

ax1.grid(
    axis="x",
    linestyle=":",
    linewidth=0.65,
    alpha=0.55,
)

ax1.set_axisbelow(True)

# Counts placed cleanly at the end of each bar
seeded_counts = [
    n_seeded_technical,
    n_seeded_recovery,
]

generic_counts = [
    n_generic_technical,
    n_generic_recovery,
]

for bars, counts in [
    (bars_seeded, seeded_counts),
    (bars_generic, generic_counts),
]:
    for bar, count in zip(bars, counts):
        value = bar.get_width()

        # Put the label inside near 100% for large bars,
        # outside for very short bars.
        if value > 0.15:
            xtext = value - 0.015
            ha = "right"
        else:
            xtext = value + 0.015
            ha = "left"

        ax1.text(
            xtext,
            bar.get_y() + bar.get_height() / 2,
            f"{count}/{n}",
            ha=ha,
            va="center",
            fontsize=9,
        )


# -----------------------------------------------------------------------------
# Panel (b): target-error ECDF
# -----------------------------------------------------------------------------

xs, ys = ecdf(seeded_error)
xg, yg = ecdf(generic_error)

positive = np.concatenate(
    [
        xs[xs > 0],
        xg[xg > 0],
        np.array([target_tolerance]),
    ]
)

xmin = max(
    np.min(positive) / 1.5,
    1e-12,
)

xs_plot = np.maximum(xs, xmin)
xg_plot = np.maximum(xg, xmin)

line_seeded, = ax2.step(
    xs_plot,
    ys,
    where="post",
    label="PINN-seeded",
)

line_generic, = ax2.step(
    xg_plot,
    yg,
    where="post",
    label="Matched generic",
)

criterion = ax2.axvline(
    target_tolerance,
    linestyle="--",
    linewidth=1.3,
    label=r"Target criterion $10^{-4}$",
)

ax2.set_xscale("log")

ax2.set_xlim(
    xmin,
    max(
        1.1,
        np.nanmax(generic_error) * 1.08,
    ),
)

ax2.set_ylim(0, 1.01)

ax2.set_xlabel(
    r"Corrected spectral error "
    r"$|c^{\mathrm{shoot}}-c^\star|$"
)

ax2.set_ylabel(
    "Empirical cumulative fraction"
)

ax2.set_title(
    "(b) Corrected target-error distribution",
    loc="left",
    pad=8,
)

ax2.grid(
    which="major",
    linestyle=":",
    linewidth=0.65,
    alpha=0.55,
)

ax2.grid(
    which="minor",
    axis="x",
    linestyle=":",
    linewidth=0.45,
    alpha=0.30,
)

ax2.set_axisbelow(True)

# Small unobtrusive annotation of the branch criterion
ax2.text(
    target_tolerance * 1.15,
    0.07,
    r"$10^{-4}$",
    rotation=90,
    va="bottom",
    ha="left",
    fontsize=8.5,
)


# =============================================================================
# Shared legend outside plotting area
# =============================================================================

handles = [
    bars_seeded,
    bars_generic,
    criterion,
]

labels_legend = [
    "PINN-seeded",
    "Matched generic",
    r"Target criterion $10^{-4}$",
]

fig.legend(
    handles,
    labels_legend,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.005),
    ncol=3,
    frameon=False,
    columnspacing=2.2,
    handlelength=2.8,
)


# =============================================================================
# Final layout
# =============================================================================

fig.subplots_adjust(
    left=0.12,
    right=0.985,
    top=0.90,
    bottom=0.23,
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
print(f"Saved PNG : {OUT_PNG}")
print(f"Saved PDF : {OUT_PDF}")
