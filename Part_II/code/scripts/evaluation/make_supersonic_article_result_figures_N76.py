from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(
    "assets/pinn_supersonic/"
    "atlas2d_v1_continuousM"
)

T401_PATH = (
    ROOT
    / "N76"
    / "shooting_T401"
    / "N76_T401_shooting_401.csv"
)

OUT = Path(
    "assets/p3-supersonic-results"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

BUDGETS = [24, 36, 48, 60, 76]

AMBIGUOUS_POINT = (
    1.10,
    0.09,
)

SHOOT_TOL = 1e-4


# =============================================================================
# Helpers
# =============================================================================

def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def spectral_error(
    cr_pred,
    ci_pred,
    cr_ref,
    ci_ref,
):
    return np.hypot(
        np.asarray(cr_pred, dtype=float)
        - np.asarray(cr_ref, dtype=float),
        np.asarray(ci_pred, dtype=float)
        - np.asarray(ci_ref, dtype=float),
    )


def key(m, a):
    return (
        round(float(m), 12),
        round(float(a), 12),
    )


def ecdf(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.sort(values)

    y = (
        np.arange(1, len(values) + 1)
        / len(values)
    )

    return values, y


def save_figure(fig, filename):
    path = OUT / filename

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("written:", path)


# =============================================================================
# Load and validate T401
# =============================================================================

if not T401_PATH.is_file():
    raise FileNotFoundError(
        f"Missing T401 merged file: {T401_PATH}"
    )

t401 = pd.read_csv(T401_PATH)

assert len(t401) == 401, len(t401)

assert (
    t401[["Mach", "alpha"]]
    .drop_duplicates()
    .shape[0]
    == 401
)

required_t401 = [
    "Mach",
    "alpha",
    "atlas_chart",
    "cr_reference",
    "ci_reference",
    "cr_pinn",
    "ci_pinn",
    "shoot_cr",
    "shoot_ci",
    "shoot_spectral_success",
    "shoot_mode_success",
]

missing = [
    c for c in required_t401
    if c not in t401.columns
]

if missing:
    raise RuntimeError(
        f"Missing T401 columns: {missing}"
    )


t401["pinn_error_article"] = spectral_error(
    t401["cr_pinn"],
    t401["ci_pinn"],
    t401["cr_reference"],
    t401["ci_reference"],
)

t401["shoot_error_article"] = spectral_error(
    t401["shoot_cr"],
    t401["shoot_ci"],
    t401["cr_reference"],
    t401["ci_reference"],
)

t401["technical_success"] = (
    as_bool(t401["shoot_spectral_success"])
    & as_bool(t401["shoot_mode_success"])
)

assert (
    t401["technical_success"].sum()
    == 401
)

t401["correct_branch"] = (
    t401["technical_success"]
    & (
        t401["shoot_error_article"]
        <= SHOOT_TOL
    )
)

assert t401["correct_branch"].sum() == 400


# =============================================================================
# FIGURE 1
# T401 ECDF: direct PINN vs PINN-seeded shooting
# =============================================================================

x_pinn, y_pinn = ecdf(
    t401["pinn_error_article"]
)

x_shoot, y_shoot = ecdf(
    t401["shoot_error_article"]
)

fig, ax = plt.subplots(
    figsize=(6.8, 4.8)
)

ax.plot(
    x_pinn,
    y_pinn,
    linewidth=2,
    label="Direct PINN",
)

ax.plot(
    x_shoot,
    y_shoot,
    linewidth=2,
    label="PINN-seeded shooting",
)

ax.axvline(
    1e-5,
    linestyle="--",
    linewidth=1,
    label=r"$10^{-5}$",
)

ax.axvline(
    1e-4,
    linestyle=":",
    linewidth=1,
    label=r"$10^{-4}$",
)

ax.set_xscale("log")

ax.set_xlabel(
    r"Complex spectral error "
    r"$\varepsilon_c=|c^{\mathrm{pred}}-c^\star|$"
)

ax.set_ylabel(
    "Cumulative fraction of test points"
)

ax.set_title(
    r"Final-test spectral accuracy on $T_{401}$"
)

ax.set_ylim(
    0.0,
    1.01,
)

ax.grid(
    True,
    which="both",
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

save_figure(
    fig,
    "Fig_supersonic_T401_error_ecdf.png",
)


# =============================================================================
# FIGURE 2
# Raw PINN error vs corrected shooting error
# =============================================================================

fig, ax = plt.subplots(
    figsize=(6.2, 5.4)
)

x = t401["pinn_error_article"].to_numpy(float)
y = t401["shoot_error_article"].to_numpy(float)

ax.scatter(
    x,
    y,
    s=28,
    alpha=0.75,
)

positive = np.concatenate(
    [
        x[x > 0],
        y[y > 0],
    ]
)

lo = positive.min() * 0.7
hi = positive.max() * 1.4

ax.plot(
    [lo, hi],
    [lo, hi],
    linestyle="--",
    linewidth=1.2,
    label="Equal error",
)

bad = t401[
    ~t401["correct_branch"]
]

assert len(bad) == 1

row = bad.iloc[0]

ax.scatter(
    [row["pinn_error_article"]],
    [row["shoot_error_article"]],
    s=120,
    marker="x",
    linewidths=2.2,
    label=(
        "Branch-selection failure\n"
        r"$(M,\alpha)=(1.15,0.06)$"
    ),
)

ax.annotate(
    r"Alternative admissible root",
    xy=(
        row["pinn_error_article"],
        row["shoot_error_article"],
    ),
    xytext=(12, -28),
    textcoords="offset points",
    arrowprops={
        "arrowstyle": "->",
    },
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)

ax.set_xlabel(
    r"Direct PINN error "
    r"$\varepsilon_c^{\mathrm{PINN}}$"
)

ax.set_ylabel(
    r"Corrected error "
    r"$\varepsilon_c^{\mathrm{shoot}}$"
)

ax.set_title(
    r"Pointwise effect of PINN-seeded shooting on $T_{401}$"
)

ax.grid(
    True,
    which="both",
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

save_figure(
    fig,
    "Fig_supersonic_T401_raw_vs_corrected.png",
)


# =============================================================================
# FIGURE 3
# T401 corrected error map in (M, alpha)
# =============================================================================

errors = t401["shoot_error_article"].to_numpy(float)

positive_errors = errors[
    np.isfinite(errors)
    & (errors > 0)
]

vmin = max(
    positive_errors.min(),
    1e-7,
)

vmax = positive_errors.max()

fig, ax = plt.subplots(
    figsize=(7.2, 5.2)
)

sc = ax.scatter(
    t401["Mach"],
    t401["alpha"],
    c=errors,
    s=42,
    norm=LogNorm(
        vmin=vmin,
        vmax=vmax,
    ),
)

bad = t401[
    ~t401["correct_branch"]
]

row = bad.iloc[0]

ax.scatter(
    [row["Mach"]],
    [row["alpha"]],
    marker="x",
    s=150,
    linewidths=2.5,
)

ax.annotate(
    (
        "Alternative root\n"
        r"$\varepsilon_c=9.78\times10^{-2}$"
    ),
    xy=(
        row["Mach"],
        row["alpha"],
    ),
    xytext=(28, 22),
    textcoords="offset points",
    arrowprops={
        "arrowstyle": "->",
    },
)

cbar = fig.colorbar(
    sc,
    ax=ax,
)

cbar.set_label(
    r"Corrected spectral error "
    r"$\varepsilon_c^{\mathrm{shoot}}$"
)

ax.set_xlabel(
    "Mach number $M$"
)

ax.set_ylabel(
    r"Wavenumber $\alpha$"
)

ax.set_title(
    r"PINN-seeded shooting error over $T_{401}$"
)

ax.grid(
    True,
    alpha=0.2,
)

fig.tight_layout()

save_figure(
    fig,
    "Fig_supersonic_T401_corrected_error_map.png",
)


# =============================================================================
# Build the V64 anchor-budget summary directly from production outputs
# =============================================================================

budget_frames = {}

for budget in BUDGETS:

    path = (
        ROOT
        / f"N{budget}"
        / "shooting_validation"
        / "shooting_validation_64.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing budget output: {path}"
        )

    df = pd.read_csv(path)

    assert len(df) == 64, (
        budget,
        len(df),
    )

    df["shoot_error_article"] = spectral_error(
        df["shoot_cr"],
        df["shoot_ci"],
        df["cr_reference"],
        df["ci_reference"],
    )

    df["technical_success"] = (
        as_bool(df["shoot_spectral_success"])
        & as_bool(df["shoot_mode_success"])
    )

    assert (
        df["technical_success"].sum()
        == 64
    )

    budget_frames[budget] = df


# =============================================================================
# Define the frozen N76-recoverable 63-point subset
# =============================================================================

n76 = budget_frames[76]

ambiguous_key = key(
    *AMBIGUOUS_POINT
)

n76["coord_key"] = [
    key(m, a)
    for m, a in zip(
        n76["Mach"],
        n76["alpha"],
    )
]

baseline63 = n76[
    n76["coord_key"] != ambiguous_key
].copy()

assert len(baseline63) == 63

assert (
    (
        baseline63["shoot_error_article"]
        <= SHOOT_TOL
    ).sum()
    == 63
)

baseline_keys = set(
    baseline63["coord_key"]
)


budget_rows = []

for budget in BUDGETS:

    df = budget_frames[budget].copy()

    df["coord_key"] = [
        key(m, a)
        for m, a in zip(
            df["Mach"],
            df["alpha"],
        )
    ]

    subset = df[
        df["coord_key"].isin(
            baseline_keys
        )
    ].copy()

    assert len(subset) == 63

    subset["recovered"] = (
        subset["technical_success"]
        & (
            subset["shoot_error_article"]
            <= SHOOT_TOL
        )
    )

    recovered = subset[
        subset["recovered"]
    ]

    budget_rows.append(
        {
            "budget": budget,
            "n_recovered": int(
                subset["recovered"].sum()
            ),
            "n_failed": int(
                (~subset["recovered"]).sum()
            ),
            "recovery_fraction": float(
                subset["recovered"].mean()
            ),
            "median_corrected_error_recovered":
                float(
                    np.median(
                        recovered[
                            "shoot_error_article"
                        ]
                    )
                ),
            "p95_corrected_error_recovered":
                float(
                    np.quantile(
                        recovered[
                            "shoot_error_article"
                        ],
                        0.95,
                    )
                ),
            "max_corrected_error_recovered":
                float(
                    recovered[
                        "shoot_error_article"
                    ].max()
                ),
        }
    )


budget_summary = (
    pd.DataFrame(budget_rows)
    .sort_values("budget")
    .reset_index(drop=True)
)

budget_summary.to_csv(
    OUT
    / "Tab_supersonic_anchor_budget_summary.csv",
    index=False,
)

print()
print("=" * 100)
print("ANCHOR-BUDGET SUMMARY")
print("=" * 100)
print(
    budget_summary.to_string(
        index=False
    )
)


# =============================================================================
# FIGURE 4
# Branch recovery versus anchor budget
# =============================================================================

fig, ax = plt.subplots(
    figsize=(6.6, 4.8)
)

bars = ax.bar(
    budget_summary["budget"].astype(str),
    budget_summary["n_recovered"],
)

for bar, value in zip(
    bars,
    budget_summary["n_recovered"],
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.25,
        f"{int(value)}/63",
        ha="center",
        va="bottom",
    )

ax.axhline(
    63,
    linestyle="--",
    linewidth=1.2,
    label="N76-recoverable validation subset",
)

ax.set_ylim(
    55,
    64.5,
)

ax.set_xlabel(
    "Number of unique spectral anchors"
)

ax.set_ylabel(
    "Correctly recovered validation points"
)

ax.set_title(
    "Supersonic branch recovery versus anchor budget"
)

ax.grid(
    True,
    axis="y",
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

save_figure(
    fig,
    "Fig_supersonic_budget_branch_recovery.png",
)


# =============================================================================
# FIGURE 5
# Corrected precision on points for which the target branch is recovered
# =============================================================================

fig, ax = plt.subplots(
    figsize=(6.6, 4.8)
)

ax.plot(
    budget_summary["budget"],
    budget_summary[
        "median_corrected_error_recovered"
    ],
    marker="o",
    linewidth=1.8,
    label="Median",
)

ax.plot(
    budget_summary["budget"],
    budget_summary[
        "p95_corrected_error_recovered"
    ],
    marker="s",
    linewidth=1.8,
    label="95th percentile",
)

ax.set_yscale("log")

ax.set_xticks(BUDGETS)

ax.set_xlabel(
    "Number of unique spectral anchors"
)

ax.set_ylabel(
    r"Corrected spectral error "
    r"$\varepsilon_c^{\mathrm{shoot}}$"
)

ax.set_title(
    "Local shooting precision after successful branch recovery"
)

ax.grid(
    True,
    which="both",
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

save_figure(
    fig,
    "Fig_supersonic_budget_corrected_precision.png",
)


# =============================================================================
# Publication-ready T401 summary table
# =============================================================================

corrected_good = t401[
    t401["correct_branch"]
][
    "shoot_error_article"
].to_numpy(float)

t401_summary = pd.DataFrame(
    [
        {
            "method": "Direct PINN",
            "n": 401,
            "mean_error": t401[
                "pinn_error_article"
            ].mean(),
            "median_error": t401[
                "pinn_error_article"
            ].median(),
            "p90_error": t401[
                "pinn_error_article"
            ].quantile(0.90),
            "p95_error": t401[
                "pinn_error_article"
            ].quantile(0.95),
            "max_error": t401[
                "pinn_error_article"
            ].max(),
            "n_le_1e5": int(
                (
                    t401["pinn_error_article"]
                    <= 1e-5
                ).sum()
            ),
            "n_le_1e4": int(
                (
                    t401["pinn_error_article"]
                    <= 1e-4
                ).sum()
            ),
        },
        {
            "method": "PINN-seeded shooting",
            "n": 401,
            "mean_error": t401[
                "shoot_error_article"
            ].mean(),
            "median_error": t401[
                "shoot_error_article"
            ].median(),
            "p90_error": t401[
                "shoot_error_article"
            ].quantile(0.90),
            "p95_error": t401[
                "shoot_error_article"
            ].quantile(0.95),
            "max_error": t401[
                "shoot_error_article"
            ].max(),
            "n_le_1e5": int(
                (
                    t401["shoot_error_article"]
                    <= 1e-5
                ).sum()
            ),
            "n_le_1e4": int(
                (
                    t401["shoot_error_article"]
                    <= 1e-4
                ).sum()
            ),
        },
    ]
)

t401_summary.to_csv(
    OUT
    / "Tab_supersonic_T401_spectral_summary.csv",
    index=False,
)

print()
print("=" * 100)
print("T401 ARTICLE SUMMARY")
print("=" * 100)
print(
    t401_summary.to_string(
        index=False
    )
)

print()
print("=" * 100)
print("ALL ARTICLE ASSETS GENERATED")
print("=" * 100)

for p in sorted(OUT.glob("*")):
    print(p)
