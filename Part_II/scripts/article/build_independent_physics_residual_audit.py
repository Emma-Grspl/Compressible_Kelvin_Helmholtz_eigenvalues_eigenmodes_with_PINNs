from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PHASE1_DIR = (
    ROOT
    / "results"
    / "physics_residual_audit"
    / "independent"
)

PHASE2_DIR = (
    ROOT
    / "results"
    / "physics_residual_audit"
    / "stage_comparison"
)

FINAL_COORD_CSV = (
    PHASE1_DIR
    / "physics_residual_coordinate_summary.csv"
)

PAIRED_CSV = (
    PHASE2_DIR
    / "stage2_vs_stage3_paired_coordinates.csv"
)

OUT_DIR = (
    ROOT
    / "assets"
    / "complementary"
    / "physics_residual"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = (
    OUT_DIR
    / "Fig_independent_physics_residual_audit.png"
)

OUT_PDF = (
    OUT_DIR
    / "Fig_independent_physics_residual_audit.pdf"
)


# =============================================================================
# Helpers
# =============================================================================

def select_t401(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the sealed T401 subset robustly from the 'dataset' column.
    """
    if "dataset" not in df.columns:
        raise RuntimeError("Expected a 'dataset' column.")

    mask = (
        df["dataset"]
        .astype(str)
        .str.lower()
        .str.contains("t401", regex=False)
    )

    out = df.loc[mask].copy()

    if len(out) != 401:
        raise RuntimeError(
            f"Expected 401 T401 rows, found {len(out)}. "
            f"Dataset labels: {sorted(df['dataset'].astype(str).unique())}"
        )

    return out


def ecdf(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise RuntimeError("Empty ECDF input.")

    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)

    return x, y


def safe_positive(values, name):
    values = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{name} contains non-finite values.")

    if np.any(values <= 0):
        raise RuntimeError(
            f"{name} contains non-positive values, "
            "which are incompatible with the requested log axes."
        )

    return values


def paired_panel(
    ax,
    x,
    y,
    title,
    xlabel,
    ylabel,
    color,
):
    x = safe_positive(x, f"{title} Stage 2")
    y = safe_positive(y, f"{title} Stage 3")

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())

    lo_plot = 10 ** np.floor(np.log10(lo))
    hi_plot = 10 ** np.ceil(np.log10(hi))

    # Paired coordinates
    ax.scatter(
        x,
        y,
        s=16,
        facecolors="none",
        edgecolors=color,
        linewidths=0.65,
        alpha=0.60,
        rasterized=True,
        zorder=2,
    )

    # Equality line
    ax.plot(
        [lo_plot, hi_plot],
        [lo_plot, hi_plot],
        linestyle="--",
        linewidth=1.0,
        color="0.25",
        zorder=1,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(lo_plot, hi_plot)
    ax.set_ylim(lo_plot, hi_plot)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title(
        title,
        loc="left",
        pad=7,
    )

    ax.grid(
        which="major",
        linestyle=":",
        linewidth=0.6,
        alpha=0.50,
    )

    ax.grid(
        which="minor",
        linestyle=":",
        linewidth=0.35,
        alpha=0.22,
    )

    ax.set_axisbelow(True)

    improved = np.mean(y < x)
    degraded = np.mean(y > x)

    med2 = np.median(x)
    med3 = np.median(y)
    reduction = 1.0 - med3 / med2

    annotation = (
        f"improved: {improved:.1%}\n"
        f"median reduction: {reduction:.0%}"
    )

    ax.text(
        0.045,
        0.955,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.75",
            "linewidth": 0.6,
            "boxstyle": "round,pad=0.28",
            "alpha": 0.92,
        },
    )


# =============================================================================
# Load
# =============================================================================

final_all = pd.read_csv(FINAL_COORD_CSV)
paired_all = pd.read_csv(PAIRED_CSV)

final = select_t401(final_all)
paired = select_t401(paired_all)


# =============================================================================
# Required columns
# =============================================================================

final_required = [
    "r_kappa_normalized_rms",
    "r_q_normalized_rms",
    "r_logamp_normalized_rms",
    "bc_kappa_rmse",
    "bc_q_rmse",
]

paired_required = [
    "r_kappa_normalized_rms_stage2",
    "r_kappa_normalized_rms_stage3",
    "r_q_normalized_rms_stage2",
    "r_q_normalized_rms_stage3",
    "r_logamp_normalized_rms_stage2",
    "r_logamp_normalized_rms_stage3",
]

missing_final = [
    c for c in final_required
    if c not in final.columns
]

missing_paired = [
    c for c in paired_required
    if c not in paired.columns
]

if missing_final:
    raise RuntimeError(
        f"Missing Phase-1 columns: {missing_final}"
    )

if missing_paired:
    raise RuntimeError(
        f"Missing Phase-2 columns: {missing_paired}"
    )


# =============================================================================
# Numerical audit
# =============================================================================

metrics_final = {
    r"$R_\kappa$":
        final["r_kappa_normalized_rms"].to_numpy(),
    r"$R_q$":
        final["r_q_normalized_rms"].to_numpy(),
    r"$R_\ell$":
        final["r_logamp_normalized_rms"].to_numpy(),
}

print("=" * 82)
print("INDEPENDENT PHYSICS-CONSISTENCY AUDIT — T401")
print("=" * 82)

for name, values in metrics_final.items():
    values = safe_positive(values, name)

    print(
        f"{name:12s} "
        f"median={np.median(values):.6e}  "
        f"P95={np.quantile(values, 0.95):.6e}  "
        f"max={np.max(values):.6e}"
    )

for name, col in [
    ("BC kappa", "bc_kappa_rmse"),
    ("BC q", "bc_q_rmse"),
]:
    values = safe_positive(
        final[col].to_numpy(),
        name,
    )

    print(
        f"{name:12s} "
        f"median={np.median(values):.6e}  "
        f"P95={np.quantile(values, 0.95):.6e}  "
        f"max={np.max(values):.6e}"
    )

print()
print("PAIRED STAGE 2 -> STAGE 3")

paired_specs = [
    (
        "R_kappa",
        "r_kappa_normalized_rms_stage2",
        "r_kappa_normalized_rms_stage3",
    ),
    (
        "R_q",
        "r_q_normalized_rms_stage2",
        "r_q_normalized_rms_stage3",
    ),
    (
        "R_logamp",
        "r_logamp_normalized_rms_stage2",
        "r_logamp_normalized_rms_stage3",
    ),
]

for name, c2, c3 in paired_specs:
    x = paired[c2].to_numpy(dtype=float)
    y = paired[c3].to_numpy(dtype=float)

    print(
        f"{name:12s} "
        f"Stage2 median={np.median(x):.6e}  "
        f"Stage3 median={np.median(y):.6e}  "
        f"improved={np.mean(y < x):.3%}"
    )


# =============================================================================
# Publication style
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9.2,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
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

COLOR_KAPPA = "#0072B2"
COLOR_Q = "#D55E00"
COLOR_LOGAMP = "#009E73"


# =============================================================================
# Figure
# =============================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9.0, 7.6),
)

ax_a = axes[0, 0]
ax_b = axes[0, 1]
ax_c = axes[1, 0]
ax_d = axes[1, 1]


# -----------------------------------------------------------------------------
# Panel (a): final independent residual distributions
# -----------------------------------------------------------------------------

for label, values, color in [
    (
        r"$R_\kappa$",
        final["r_kappa_normalized_rms"].to_numpy(),
        COLOR_KAPPA,
    ),
    (
        r"$R_q$",
        final["r_q_normalized_rms"].to_numpy(),
        COLOR_Q,
    ),
    (
        r"$R_\ell$",
        final["r_logamp_normalized_rms"].to_numpy(),
        COLOR_LOGAMP,
    ),
]:
    x, y = ecdf(values)

    ax_a.step(
        x,
        y,
        where="post",
        linewidth=1.7,
        label=label,
        color=color,
    )

ax_a.set_xscale("log")
ax_a.set_xlim(
    2e-3,
    5e-2,
)

# Clean publication ticks: keep the logarithmic structure without
# overlapping minor-tick labels.
ax_a.set_xticks(
    [3e-3, 1e-2, 3e-2]
)

ax_a.set_xticklabels(
    [
        r"$3\times10^{-3}$",
        r"$10^{-2}$",
        r"$3\times10^{-2}$",
    ]
)

ax_a.xaxis.set_minor_formatter(
    NullFormatter()
)

ax_a.set_ylim(
    0,
    1.01,
)

ax_a.set_xlabel(
    "Normalized RMS residual"
)

ax_a.set_ylabel(
    "Empirical cumulative fraction"
)

ax_a.set_title(
    "(a) Independent residuals on sealed $T_{401}$",
    loc="left",
    pad=7,
)

ax_a.grid(
    which="major",
    linestyle=":",
    linewidth=0.6,
    alpha=0.50,
)

ax_a.grid(
    which="minor",
    axis="x",
    linestyle=":",
    linewidth=0.35,
    alpha=0.22,
)

ax_a.legend(
    frameon=False,
    loc="lower right",
)

ax_a.set_axisbelow(True)

# Compact information on BCs:
bc_k_med = np.median(final["bc_kappa_rmse"])
bc_q_med = np.median(final["bc_q_rmse"])

ax_a.text(
    0.045,
    0.96,
    (
        "Boundary RMS medians\n"
        rf"$\kappa$: {bc_k_med:.1e}, "
        rf"$q$: {bc_q_med:.1e}"
    ),
    transform=ax_a.transAxes,
    ha="left",
    va="top",
    fontsize=8.0,
    bbox={
        "facecolor": "white",
        "edgecolor": "0.78",
        "linewidth": 0.6,
        "boxstyle": "round,pad=0.28",
        "alpha": 0.92,
    },
)


# -----------------------------------------------------------------------------
# Panel (b): R_kappa
# -----------------------------------------------------------------------------

paired_panel(
    ax=ax_b,
    x=paired[
        "r_kappa_normalized_rms_stage2"
    ].to_numpy(),
    y=paired[
        "r_kappa_normalized_rms_stage3"
    ].to_numpy(),
    title=r"(b) Paired $R_\kappa$ residuals",
    xlabel=r"Stage 2 normalized RMS",
    ylabel=r"Stage 3 normalized RMS",
    color=COLOR_KAPPA,
)


# -----------------------------------------------------------------------------
# Panel (c): R_q
# -----------------------------------------------------------------------------

paired_panel(
    ax=ax_c,
    x=paired[
        "r_q_normalized_rms_stage2"
    ].to_numpy(),
    y=paired[
        "r_q_normalized_rms_stage3"
    ].to_numpy(),
    title=r"(c) Paired $R_q$ residuals",
    xlabel=r"Stage 2 normalized RMS",
    ylabel=r"Stage 3 normalized RMS",
    color=COLOR_Q,
)


# -----------------------------------------------------------------------------
# Panel (d): R_l
# -----------------------------------------------------------------------------

paired_panel(
    ax=ax_d,
    x=paired[
        "r_logamp_normalized_rms_stage2"
    ].to_numpy(),
    y=paired[
        "r_logamp_normalized_rms_stage3"
    ].to_numpy(),
    title=r"(d) Paired $R_\ell$ residuals",
    xlabel=r"Stage 2 normalized RMS",
    ylabel=r"Stage 3 normalized RMS",
    color=COLOR_LOGAMP,
)


# =============================================================================
# Shared explanatory marker
# =============================================================================

fig.text(
    0.985,
    0.02,
    r"Dashed line in (b--d): Stage 3 = Stage 2",
    ha="right",
    va="bottom",
    fontsize=8.0,
    color="0.30",
)


# =============================================================================
# Layout
# =============================================================================

fig.subplots_adjust(
    left=0.105,
    right=0.975,
    top=0.955,
    bottom=0.095,
    wspace=0.30,
    hspace=0.34,
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
