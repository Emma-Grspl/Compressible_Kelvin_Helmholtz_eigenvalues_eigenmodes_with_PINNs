#!/usr/bin/env python3

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd()

BASE = (
    ROOT
    / "assets/pinn_supersonic/"
      "atlas2d_v1_continuousM/N76"
)

ARTICLE = (
    ROOT
    / "assets/p3-supersonic-results"
)

ARTICLE.mkdir(
    parents=True,
    exist_ok=True,
)


PERF_JSON = (
    ARTICLE
    / "Tab_supersonic_computational_speedups.json"
)

CLASSICAL_CSV = (
    BASE
    / "final_benchmarks/"
      "classical_from_scratch/results/"
      "analysis/"
      "N76_CLASSICAL_COST500_merged_500.csv"
)

PINN_SHOOT_CSV = (
    BASE
    / "runtime_benchmark/analysis/"
      "N76_COST500_shooting_timing_500.csv"
)


assert PERF_JSON.is_file(), PERF_JSON
assert CLASSICAL_CSV.is_file(), CLASSICAL_CSV
assert PINN_SHOOT_CSV.is_file(), PINN_SHOOT_CSV


# =============================================================================
# Load frozen values
# =============================================================================

perf = json.loads(
    PERF_JSON.read_text()
)

classic = pd.read_csv(
    CLASSICAL_CSV
)

seeded = pd.read_csv(
    PINN_SHOOT_CSV
)

assert len(classic) == 500
assert len(seeded) == 500


def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
                "completed",
                "success",
            ]
        )
    )


# =============================================================================
# Technical-success definitions
# =============================================================================

if "_technical_success" in classic.columns:
    classic_success = as_bool(
        classic["_technical_success"]
    )

elif "classical_success" in classic.columns:
    classic_success = as_bool(
        classic["classical_success"]
    )

else:
    raise RuntimeError(
        "No classical technical-success column."
    )


if (
    "shoot_spectral_success" in seeded.columns
    and "shoot_mode_success" in seeded.columns
):
    seeded_success = (
        as_bool(
            seeded["shoot_spectral_success"]
        )
        & as_bool(
            seeded["shoot_mode_success"]
        )
    )

elif "technical_success" in seeded.columns:
    seeded_success = as_bool(
        seeded["technical_success"]
    )

else:
    raise RuntimeError(
        "No PINN-seeded technical-success columns."
    )


assert int(classic_success.sum()) == 420
assert int(seeded_success.sum()) == 500


# =============================================================================
# Computational values
# =============================================================================

labels = [
    "Direct PINN\nspectral",
    "Direct PINN\nspectral + modal",
    "Generic classical\nshooting",
    "PINN-seeded\nshooting",
]


cost1 = np.array(
    [
        perf[
            "direct_PINN_spectral_COST1_seconds"
        ],
        perf[
            "direct_PINN_full_modal_COST1_seconds"
        ],
        perf[
            "classical_COST1_seconds"
        ],
        perf[
            "hybrid_COST1_seconds"
        ],
    ],
    dtype=float,
)


cost500 = np.array(
    [
        perf[
            "direct_PINN_spectral_COST500_batch_seconds"
        ],
        perf[
            "direct_PINN_full_modal_COST500_batch_seconds"
        ],
        perf[
            "classical_COST500_serial_seconds"
        ],
        perf[
            "hybrid_COST500_serial_seconds"
        ],
    ],
    dtype=float,
)


# =============================================================================
# Figure
# =============================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        14.5,
        10.5,
    ),
)

ax1, ax2, ax3, ax4 = axes.ravel()

x = np.arange(
    len(labels)
)


# -----------------------------------------------------------------------------
# (a) COST1
# -----------------------------------------------------------------------------

bars1 = ax1.bar(
    x,
    cost1,
)

ax1.set_yscale(
    "log"
)

ax1.set_xticks(
    x
)

ax1.set_xticklabels(
    labels,
    fontsize=9,
)

ax1.set_ylabel(
    "Wall time per query [s]"
)

ax1.set_title(
    "(a) COST1: single-query latency",
    loc="left",
)

ax1.grid(
    axis="y",
    alpha=0.25,
)

for bar, value in zip(
    bars1,
    cost1,
):
    if value < 1e-3:
        text = (
            f"{value * 1e3:.3f} ms"
        )

    elif value < 1:
        text = (
            f"{value * 1e3:.1f} ms"
        )

    else:
        text = (
            f"{value:.1f} s"
        )

    ax1.text(
        bar.get_x()
        + bar.get_width() / 2,
        value * 1.35,
        text,
        ha="center",
        va="bottom",
        fontsize=8.5,
        rotation=0,
    )


# -----------------------------------------------------------------------------
# (b) COST500
# -----------------------------------------------------------------------------

bars2 = ax2.bar(
    x,
    cost500,
)

ax2.set_yscale(
    "log"
)

ax2.set_xticks(
    x
)

ax2.set_xticklabels(
    labels,
    fontsize=9,
)

ax2.set_ylabel(
    "Total time for 500 queries [s]"
)

ax2.set_title(
    "(b) COST500: batch/serial-equivalent cost",
    loc="left",
)

ax2.grid(
    axis="y",
    alpha=0.25,
)

for bar, value in zip(
    bars2,
    cost500,
):
    if value < 1:
        text = (
            f"{value * 1e3:.1f} ms"
        )

    elif value < 3600:
        text = (
            f"{value:.1f} s"
        )

    else:
        text = (
            f"{value / 3600:.1f} h"
        )

    ax2.text(
        bar.get_x()
        + bar.get_width() / 2,
        value * 1.35,
        text,
        ha="center",
        va="bottom",
        fontsize=8.5,
    )




ax2.margins(y=0.18)


# -----------------------------------------------------------------------------
# (c) Technical success
# -----------------------------------------------------------------------------

success_counts = np.array(
    [
        int(
            classic_success.sum()
        ),
        int(
            seeded_success.sum()
        ),
    ]
)

success_pct = (
    100.0
    * success_counts
    / 500.0
)

success_labels = [
    "Generic classical\nshooting",
    "PINN-seeded\nshooting",
]

bars3 = ax3.bar(
    np.arange(2),
    success_pct,
)

ax3.set_xticks(
    np.arange(2)
)

ax3.set_xticklabels(
    success_labels,
)

ax3.set_ylim(
    0,
    112,
)

ax3.set_ylabel(
    "Technical success [%]"
)

ax3.set_title(
    "(c) COST500 shooting robustness",
    loc="left",
)

ax3.grid(
    axis="y",
    alpha=0.25,
)

for bar, n, pct in zip(
    bars3,
    success_counts,
    success_pct,
):
    ax3.text(
        bar.get_x()
        + bar.get_width() / 2,
        pct + 1.5,
        f"{n}/500\n({pct:.0f}%)",
        ha="center",
        va="bottom",
        fontsize=10,
    )


# -----------------------------------------------------------------------------
# (d) Classical convergence map
# -----------------------------------------------------------------------------

required = {
    "Mach",
    "alpha",
}

if not required.issubset(
    classic.columns
):
    raise RuntimeError(
        "Mach/alpha missing from classical COST500."
    )


ok = classic_success.to_numpy(
    dtype=bool
)

fail = ~ok


ax4.scatter(
    classic.loc[
        ok,
        "Mach",
    ],
    classic.loc[
        ok,
        "alpha",
    ],
    marker="o",
    s=24,
    alpha=0.65,
    label="Generic shooting: success",
)

ax4.scatter(
    classic.loc[
        fail,
        "Mach",
    ],
    classic.loc[
        fail,
        "alpha",
    ],
    marker="x",
    s=42,
    linewidths=1.4,
    label="Generic shooting: failure",
)

ax4.set_xlabel(
    "Mach number $M$"
)

ax4.set_ylabel(
    r"Wavenumber $\alpha$"
)

ax4.set_title(
    "(d) Location of generic-shooting failures",
    loc="left",
)

ax4.grid(
    alpha=0.2,
)





# Legend for panel (d), placed outside all plotting axes.
handles4, labels4 = ax4.get_legend_handles_labels()


# =============================================================================
# Save
# =============================================================================

fig.suptitle(
    (
        "Computational cost and robustness of the "
        "supersonic spectral pipelines"
    ),
    fontsize=16,
    y=0.995,
)

fig.legend(
    handles4,
    labels4,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.048),
    ncol=2,
    frameon=False,
    fontsize=9.0,
    columnspacing=2.0,
    handletextpad=0.6,
)

fig.text(
    0.5,
    0.014,
    (
        "Timing convention: direct PINN = GPU-batched wall time; "
        "shooting methods = serial-equivalent sum of individual solves."
    ),
    ha="center",
    va="bottom",
    fontsize=9.0,
)

fig.tight_layout(
    rect=[
        0.02,
        0.115,
        0.98,
        0.95,
    ],
    h_pad=4.2,
    w_pad=2.2,
)

PNG = (
    ARTICLE
    / "Fig_supersonic_computational_cost_and_robustness.png"
)

PDF = (
    ARTICLE
    / "Fig_supersonic_computational_cost_and_robustness.pdf"
)

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF,
    bbox_inches="tight",
)

plt.close(
    fig
)


# =============================================================================
# Article table
# =============================================================================

summary = pd.DataFrame(
    {
        "pipeline": labels,
        "COST1_seconds": cost1,
        "COST500_seconds": cost500,
        "COST500_time_definition": [
            "GPU batched wall time",
            "GPU batched wall time",
            "serial-equivalent sum",
            "serial-equivalent sum",
        ],
        "technical_success_COST500": [
            np.nan,
            np.nan,
            int(
                classic_success.sum()
            ),
            int(
                seeded_success.sum()
            ),
        ],
    }
)

TABLE = (
    ARTICLE
    / "Tab_supersonic_computational_cost_and_robustness.csv"
)

summary.to_csv(
    TABLE,
    index=False,
)


print("=" * 100)
print("COMPUTATIONAL ASSET COMPLETE")
print("=" * 100)

print()
print(summary.to_string(index=False))

print()
print("classical success =", int(classic_success.sum()), "/ 500")
print("seeded success    =", int(seeded_success.sum()), "/ 500")

print()
print(PNG)
print(PDF)
print(TABLE)
