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

PHASE4 = ROOT / "results" / "multiseed_ablation"

OUT_DIR = ROOT / "assets" / "complementary" / "anchors_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Fig_multiseed_physics_vs_anchors_ablation.png"
OUT_PDF = OUT_DIR / "Fig_multiseed_physics_vs_anchors_ablation.pdf"

OUT_PER_SEED = OUT_DIR / "multiseed_physics_vs_anchors_per_seed.csv"
OUT_SUMMARY = OUT_DIR / "multiseed_physics_vs_anchors_summary.csv"


# =============================================================================
# Protocol
# =============================================================================

TARGET_TOL = 1.0e-4

# Predeclared multibranch validation coordinate excluded from binary
# target-branch recovery.
AMBIGUOUS_M = 1.10
AMBIGUOUS_ALPHA = 0.09


RUNS = {
    "Physics-informed": {
        1: "N76_fullc_seed1",
        2: "N76_fullc_seed2",
        3: "N76_fullc_seed3",
    },
    "Anchors-only": {
        1: "N76_anchors_only",
        2: "N76_anchors_only_seed2",
        3: "N76_anchors_only_seed3",
    },
}


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
        .isin({"true", "1", "yes", "y", "success", "ok"})
    )


def find_shooting_csv(run_dir: Path) -> Path:
    candidates = [
        run_dir / "shooting_V64" / "shooting_validation_64.csv",
        run_dir / "production_shooting" / "shooting_validation_64.csv",
    ]

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"No V64 shooting CSV found for {run_dir.name}"
    )


def load_run(condition: str, seed: int, tag: str) -> dict:
    run_dir = PHASE4 / tag

    metrics_path = (
        run_dir
        / "evaluation"
        / "joint"
        / "N76_validation_metrics_global.json"
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)

    with open(metrics_path) as f:
        direct = json.load(f)

    shoot_path = find_shooting_csv(run_dir)
    shoot = pd.read_csv(shoot_path)

    if len(shoot) != 64:
        raise RuntimeError(
            f"{tag}: expected 64 V64 rows, found {len(shoot)}"
        )

    required = [
        "Mach",
        "alpha",
        "shoot_spectral_success",
        "shoot_mode_success",
        "shoot_spectral_error",
    ]

    missing = [c for c in required if c not in shoot.columns]
    if missing:
        raise RuntimeError(
            f"{tag}: missing shooting columns {missing}"
        )

    technical = (
        as_bool(shoot["shoot_spectral_success"])
        & as_bool(shoot["shoot_mode_success"])
    )

    ambiguous = (
        np.isclose(
            shoot["Mach"].to_numpy(dtype=float),
            AMBIGUOUS_M,
            atol=1e-10,
            rtol=0,
        )
        & np.isclose(
            shoot["alpha"].to_numpy(dtype=float),
            AMBIGUOUS_ALPHA,
            atol=1e-10,
            rtol=0,
        )
    )

    if ambiguous.sum() != 1:
        raise RuntimeError(
            f"{tag}: expected exactly one predeclared ambiguous "
            f"V64 coordinate, found {ambiguous.sum()}"
        )

    nonambiguous = ~ambiguous

    shoot_error = pd.to_numeric(
        shoot["shoot_spectral_error"],
        errors="coerce",
    ).to_numpy(dtype=float)

    recovery = (
        technical.to_numpy()
        & nonambiguous
        & np.isfinite(shoot_error)
        & (shoot_error <= TARGET_TOL)
    )

    recovered_errors = shoot_error[recovery]

    n_technical = int(technical.sum())
    n_recovered = int(recovery.sum())

    if n_technical != 64:
        raise RuntimeError(
            f"{tag}: expected 64/64 technical success, "
            f"found {n_technical}/64"
        )

    if n_recovered != 63:
        raise RuntimeError(
            f"{tag}: expected 63/63 target recovery, "
            f"found {n_recovered}/63"
        )

    return {
        "condition": condition,
        "seed": seed,
        "tag": tag,

        "direct_mean": float(direct["spectral_error_mean"]),
        "direct_median": float(direct["spectral_error_median"]),
        "direct_p95": float(direct["spectral_error_p95"]),
        "direct_max": float(direct["spectral_error_max"]),

        "technical_success": n_technical,
        "target_recovery": n_recovered,

        "corrected_median": float(
            np.median(recovered_errors)
        ),
        "corrected_p95": float(
            np.quantile(recovered_errors, 0.95)
        ),
        "corrected_max": float(
            np.max(recovered_errors)
        ),
    }


# =============================================================================
# Load all six runs
# =============================================================================

records = []

for condition, seeds in RUNS.items():
    for seed, tag in seeds.items():
        records.append(
            load_run(
                condition=condition,
                seed=seed,
                tag=tag,
            )
        )

df = pd.DataFrame(records)

df = df.sort_values(
    ["condition", "seed"]
).reset_index(drop=True)

df.to_csv(
    OUT_PER_SEED,
    index=False,
)


# =============================================================================
# Aggregate summary
# =============================================================================

metric_columns = [
    "direct_mean",
    "direct_median",
    "direct_p95",
    "direct_max",
    "corrected_median",
    "corrected_p95",
    "corrected_max",
]

summary_rows = []

for condition, group in df.groupby("condition", sort=False):
    row = {
        "condition": condition,
        "n_seeds": len(group),
        "technical_success_total":
            int(group["technical_success"].sum()),
        "target_recovery_total":
            int(group["target_recovery"].sum()),
    }

    for metric in metric_columns:
        values = group[metric].to_numpy(dtype=float)

        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_sample_sd"] = float(
            np.std(values, ddof=1)
        )

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUT_SUMMARY,
    index=False,
)


# =============================================================================
# Numerical audit
# =============================================================================

print("=" * 84)
print("MULTI-SEED N76 PHYSICS-INFORMED VS ANCHORS-ONLY")
print("=" * 84)

display_cols = [
    "condition",
    "seed",
    "direct_mean",
    "direct_p95",
    "technical_success",
    "target_recovery",
    "corrected_median",
    "corrected_p95",
]

print(
    df[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6e}",
    )
)

print()
print("AGGREGATE ACROSS THREE SEEDS")
print("-" * 84)

for condition in RUNS:
    g = df[df["condition"] == condition]

    print(condition)

    print(
        "  direct mean       = "
        f"{g['direct_mean'].mean():.6e} "
        f"± {g['direct_mean'].std(ddof=1):.6e}"
    )

    print(
        "  direct P95        = "
        f"{g['direct_p95'].mean():.6e} "
        f"± {g['direct_p95'].std(ddof=1):.6e}"
    )

    print(
        "  corrected median  = "
        f"{g['corrected_median'].mean():.6e} "
        f"± {g['corrected_median'].std(ddof=1):.6e}"
    )

    print(
        "  corrected P95     = "
        f"{g['corrected_p95'].mean():.6e} "
        f"± {g['corrected_p95'].std(ddof=1):.6e}"
    )

    print(
        "  technical         = "
        f"{g['technical_success'].sum()}/192"
    )

    print(
        "  target recovery   = "
        f"{g['target_recovery'].sum()}/189"
    )

    print()

physics = df[df["condition"] == "Physics-informed"]
anchors = df[df["condition"] == "Anchors-only"]

delta_direct_mean = (
    anchors["direct_mean"].mean()
    / physics["direct_mean"].mean()
    - 1.0
)

delta_direct_p95 = (
    anchors["direct_p95"].mean()
    / physics["direct_p95"].mean()
    - 1.0
)

print(
    "Anchors-only minus physics-informed:"
)
print(
    f"  mean direct error relative difference = "
    f"{delta_direct_mean:+.3%}"
)
print(
    f"  mean P95 relative difference          = "
    f"{delta_direct_p95:+.3%}"
)


# =============================================================================
# Publication style
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9.4,
        "axes.labelsize": 9.7,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# =============================================================================
# Plot configuration
# =============================================================================

CONDITIONS = [
    "Physics-informed",
    "Anchors-only",
]

X = np.array([0.0, 1.0])

COLOR_PHYSICS = "#0072B2"
COLOR_ANCHORS = "#D55E00"
COLOR_PAIR = "0.72"
COLOR_AGG = "0.10"

condition_colors = [
    COLOR_PHYSICS,
    COLOR_ANCHORS,
]


def draw_paired_metric(
    ax,
    metric,
    scale,
    ylabel,
    title,
):
    """
    Draw matched-seed paired points plus aggregate mean ± sample SD.
    """

    # Matched seed realizations
    for seed in [1, 2, 3]:
        p = df[
            (df["condition"] == "Physics-informed")
            & (df["seed"] == seed)
        ][metric].iloc[0] * scale

        a = df[
            (df["condition"] == "Anchors-only")
            & (df["seed"] == seed)
        ][metric].iloc[0] * scale

        ax.plot(
            X,
            [p, a],
            color=COLOR_PAIR,
            linewidth=1.0,
            zorder=1,
        )

        ax.scatter(
            X[0],
            p,
            s=38,
            marker="o",
            facecolor=COLOR_PHYSICS,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

        ax.scatter(
            X[1],
            a,
            s=38,
            marker="o",
            facecolor=COLOR_ANCHORS,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    # Aggregate means and sample SD
    means = []
    sds = []

    for condition in CONDITIONS:
        values = (
            df.loc[
                df["condition"] == condition,
                metric,
            ].to_numpy(dtype=float)
            * scale
        )

        means.append(values.mean())
        sds.append(values.std(ddof=1))

    ax.errorbar(
        X,
        means,
        yerr=sds,
        fmt="D",
        markersize=5.2,
        color=COLOR_AGG,
        markerfacecolor="white",
        markeredgewidth=1.2,
        capsize=3.2,
        elinewidth=1.2,
        linewidth=0,
        zorder=5,
    )

    ax.set_xlim(-0.35, 1.35)

    ax.set_xticks(X)
    ax.set_xticklabels(
        [
            "Physics-informed",
            "Anchors-only",
        ]
    )

    ax.set_ylabel(ylabel)

    ax.set_title(
        title,
        loc="left",
        pad=7,
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.50,
    )

    ax.set_axisbelow(True)

    # Reasonable y padding
    all_values = []

    for condition in CONDITIONS:
        all_values.extend(
            (
                df.loc[
                    df["condition"] == condition,
                    metric,
                ].to_numpy(dtype=float)
                * scale
            ).tolist()
        )

    lo = min(all_values)
    hi = max(all_values)
    spread = hi - lo

    if spread == 0:
        spread = max(abs(hi), 1.0) * 0.1

    ax.set_ylim(
        lo - 0.28 * spread,
        hi + 0.32 * spread,
    )


# =============================================================================
# Figure
# =============================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(8.4, 6.7),
)

draw_paired_metric(
    ax=axes[0, 0],
    metric="direct_mean",
    scale=1e3,
    ylabel=r"Mean direct error ($\times 10^{-3}$)",
    title="(a) Direct spectral mean error",
)

draw_paired_metric(
    ax=axes[0, 1],
    metric="direct_p95",
    scale=1e3,
    ylabel=r"Direct $P_{95}$ error ($\times 10^{-3}$)",
    title=r"(b) Direct spectral $P_{95}$",
)

draw_paired_metric(
    ax=axes[1, 0],
    metric="corrected_median",
    scale=1e6,
    ylabel=r"Median corrected error ($\times 10^{-6}$)",
    title="(c) Corrected median error",
)

draw_paired_metric(
    ax=axes[1, 1],
    metric="corrected_p95",
    scale=1e6,
    ylabel=r"Corrected $P_{95}$ error ($\times 10^{-6}$)",
    title=r"(d) Corrected $P_{95}$ error",
)


# =============================================================================
# Global result annotation
# =============================================================================

fig.text(
    0.5,
    0.978,
    (
        r"Technical success: $64/64$ for every realization"
        r"   $\bullet$   "
        r"Target-branch recovery: $63/63$ for every realization"
    ),
    ha="center",
    va="top",
    fontsize=9.2,
)


# =============================================================================
# Shared legend
# =============================================================================

legend_handles = [
    Line2D(
        [0, 1],
        [0, 0],
        color=COLOR_PAIR,
        linewidth=1.0,
        marker="o",
        markerfacecolor="white",
        markeredgecolor="0.45",
        markersize=5,
        label="Matched seed realization",
    ),
    Line2D(
        [0],
        [0],
        linestyle="none",
        marker="D",
        markerfacecolor="white",
        markeredgecolor=COLOR_AGG,
        markeredgewidth=1.2,
        markersize=5.5,
        label=r"Mean $\pm$ sample SD",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=2,
    frameon=False,
    columnspacing=2.4,
    handlelength=2.7,
)


# =============================================================================
# Layout
# =============================================================================

fig.subplots_adjust(
    left=0.105,
    right=0.975,
    top=0.925,
    bottom=0.115,
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
print(f"Saved PNG       : {OUT_PNG}")
print(f"Saved PDF       : {OUT_PDF}")
print(f"Saved per-seed  : {OUT_PER_SEED}")
print(f"Saved summary   : {OUT_SUMMARY}")
