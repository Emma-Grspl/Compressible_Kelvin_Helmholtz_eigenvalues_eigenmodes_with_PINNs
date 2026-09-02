#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

OUT = (
    ROOT
    / "assets/pinn_subsonic/article/N340"
)

SHARDS = OUT / "seam_shards"


def save(fig, name):
    fig.savefig(
        OUT / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{name}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


frames = []

for path in sorted(
    SHARDS.glob("shard_*.csv")
):
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        continue

    if not df.empty:
        frames.append(df)

if not frames:
    raise RuntimeError(
        "No seam validation results."
    )

df = pd.concat(
    frames,
    ignore_index=True,
)

if "success" in df.columns:
    success = (
        df["success"]
        .fillna(False)
        .astype(bool)
    )

    df = df.loc[success].copy()

# Spectral disagreement.
if "ci_abs_ab" in df.columns:
    ci_abs = pd.to_numeric(
        df["ci_abs_ab"],
        errors="coerce",
    )

elif {
    "ci_a",
    "ci_b",
}.issubset(df.columns):
    ci_abs = np.abs(
        pd.to_numeric(
            df["ci_a"],
            errors="coerce",
        )
        - pd.to_numeric(
            df["ci_b"],
            errors="coerce",
        )
    )

else:
    raise KeyError(
        "Cannot identify seam ci columns. "
        f"Columns are: {df.columns.tolist()}"
    )

if "p_overlap_ab" not in df.columns:
    raise KeyError(
        "Missing p_overlap_ab. "
        f"Columns are: {df.columns.tolist()}"
    )

p_overlap = pd.to_numeric(
    df["p_overlap_ab"],
    errors="coerce",
)

p_defect = 1.0 - p_overlap

df["ci_abs_ab_article"] = ci_abs
df["p_overlap_defect_article"] = (
    p_defect
)

df.to_csv(
    OUT
    / "N340_seam_validation_all.csv",
    index=False,
)


def describe(name, values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    return {
        "metric": name,
        "n": len(values),
        "median": np.median(values),
        "p90": np.quantile(values, 0.90),
        "p95": np.quantile(values, 0.95),
        "p99": np.quantile(values, 0.99),
        "max": np.max(values),
    }


summary = pd.DataFrame([
    describe(
        "spectral_abs_ci_difference",
        ci_abs,
    ),
    describe(
        "pressure_overlap_defect",
        p_defect,
    ),
])

summary.to_csv(
    OUT
    / "Table_atlas_overlap_consistency_N340.csv",
    index=False,
)


def ecdf(ax, values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    values = np.maximum(
        values,
        1.0e-16,
    )

    values = np.sort(values)

    cumulative = (
        np.arange(
            1,
            len(values) + 1,
        )
        / len(values)
    )

    ax.plot(
        values,
        cumulative,
        lw=2.2,
    )

    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.grid(
        True,
        which="both",
        alpha=0.22,
    )


fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.5, 5.0),
)

ecdf(
    axes[0],
    ci_abs,
)

axes[0].set_title(
    "Spectral agreement in chart overlaps"
)

axes[0].set_xlabel(
    r"$|c_i^{(k)}-c_i^{(\ell)}|$"
)

axes[0].set_ylabel(
    "Empirical cumulative probability"
)


ecdf(
    axes[1],
    p_defect,
)

axes[1].set_title(
    "Pressure-mode agreement"
)

axes[1].set_xlabel(
    r"$1-\mathcal{O}_p^{(k,\ell)}$"
)

axes[1].set_ylabel(
    "Empirical cumulative probability"
)

fig.suptitle(
    r"N340 atlas overlap consistency"
)

fig.tight_layout()

save(
    fig,
    "Fig_atlas_overlap_consistency_N340",
)

print(summary.to_string(index=False))

print(
    "\nWrote:",
    OUT
    / "Fig_atlas_overlap_consistency_N340.png",
)
