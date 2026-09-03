from pathlib import Path
import json

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PHASE10 = (
    ROOT
    / "results"
    / "computational_cost"
    / "article_compute_cost"
)

SUMMARY_JSON = PHASE10 / "gpu_hours_summary.json"
JOBS_CSV = PHASE10 / "training_jobs_clean.csv"

OUT_DIR = ROOT / "assets" / "complementary" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "Table_T6_training_gpu_hours.csv"
OUT_TEX = OUT_DIR / "Table_T6_training_gpu_hours.tex"


# =============================================================================
# Load
# =============================================================================

with open(SUMMARY_JSON) as f:
    summary = json.load(f)

jobs = pd.read_csv(JOBS_CSV)


# =============================================================================
# Validate JSON schema
# =============================================================================

required_summary = [
    "model",
    "n_charts",
    "gpu_type",
    "gpus_per_chart",
    "prefit_steps",
    "modal_steps",
    "joint_steps",
    "total_training_steps_per_chart",
    "total_gpu_seconds",
    "total_gpu_hours",
    "mean_seconds_per_chart",
    "median_seconds_per_chart",
    "min_seconds_per_chart",
    "max_seconds_per_chart",
    "scope",
]

missing = [k for k in required_summary if k not in summary]

if missing:
    raise RuntimeError(
        f"gpu_hours_summary.json missing keys: {missing}"
    )


# =============================================================================
# Validate jobs CSV
# =============================================================================

required_jobs = [
    "chart",
    "elapsed_seconds",
    "n_gpus",
    "gpu_hours",
    "included_in_article_total",
]

missing = [c for c in required_jobs if c not in jobs.columns]

if missing:
    raise RuntimeError(
        f"training_jobs_clean.csv missing columns: {missing}"
    )


def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


included = jobs[
    as_bool(jobs["included_in_article_total"])
].copy()

n_charts = int(summary["n_charts"])

if len(included) != n_charts:
    raise RuntimeError(
        f"Expected {n_charts} included training jobs, "
        f"found {len(included)}."
    )


# =============================================================================
# Independent consistency checks
# =============================================================================

elapsed = pd.to_numeric(
    included["elapsed_seconds"],
    errors="raise",
).to_numpy(dtype=float)

gpu_hours = pd.to_numeric(
    included["gpu_hours"],
    errors="raise",
).to_numpy(dtype=float)

total_seconds_csv = float(
    np.sum(
        elapsed
        * pd.to_numeric(
            included["n_gpus"],
            errors="raise",
        ).to_numpy(dtype=float)
    )
)

total_gpu_hours_csv = float(np.sum(gpu_hours))

checks = [
    (
        "total GPU seconds",
        total_seconds_csv,
        float(summary["total_gpu_seconds"]),
        1e-6,
    ),
    (
        "total GPU hours",
        total_gpu_hours_csv,
        float(summary["total_gpu_hours"]),
        1e-9,
    ),
    (
        "mean seconds/chart",
        float(np.mean(elapsed)),
        float(summary["mean_seconds_per_chart"]),
        1e-6,
    ),
    (
        "median seconds/chart",
        float(np.median(elapsed)),
        float(summary["median_seconds_per_chart"]),
        1e-6,
    ),
    (
        "min seconds/chart",
        float(np.min(elapsed)),
        float(summary["min_seconds_per_chart"]),
        1e-6,
    ),
    (
        "max seconds/chart",
        float(np.max(elapsed)),
        float(summary["max_seconds_per_chart"]),
        1e-6,
    ),
]

for label, observed, expected, atol in checks:
    if not np.isclose(
        observed,
        expected,
        atol=atol,
        rtol=1e-8,
    ):
        raise RuntimeError(
            f"{label}: CSV gives {observed}, "
            f"JSON gives {expected}."
        )


# =============================================================================
# Check training-step accounting
# =============================================================================

prefit_steps = int(summary["prefit_steps"])
modal_steps = int(summary["modal_steps"])
joint_steps = int(summary["joint_steps"])

expected_total_steps = (
    prefit_steps
    + modal_steps
    + joint_steps
)

reported_total_steps = int(
    summary["total_training_steps_per_chart"]
)

if expected_total_steps != reported_total_steps:
    raise RuntimeError(
        "Training-step accounting mismatch: "
        f"{prefit_steps} + {modal_steps} + {joint_steps} "
        f"= {expected_total_steps}, "
        f"but JSON reports {reported_total_steps}."
    )


# =============================================================================
# Publication source table
# =============================================================================

table = pd.DataFrame(
    [
        {
            "model": str(summary["model"]),
            "gpu_type": str(summary["gpu_type"]),
            "n_charts": n_charts,
            "gpus_per_chart": int(summary["gpus_per_chart"]),
            "prefit_steps": prefit_steps,
            "modal_steps": modal_steps,
            "joint_steps": joint_steps,
            "total_steps_per_chart": reported_total_steps,
            "total_gpu_seconds": float(
                summary["total_gpu_seconds"]
            ),
            "total_gpu_hours": float(
                summary["total_gpu_hours"]
            ),
            "mean_seconds_per_chart": float(
                summary["mean_seconds_per_chart"]
            ),
            "median_seconds_per_chart": float(
                summary["median_seconds_per_chart"]
            ),
            "min_seconds_per_chart": float(
                summary["min_seconds_per_chart"]
            ),
            "max_seconds_per_chart": float(
                summary["max_seconds_per_chart"]
            ),
            "scope": str(summary["scope"]),
        }
    ]
)

table.to_csv(
    OUT_CSV,
    index=False,
)


# =============================================================================
# Console audit
# =============================================================================

print("=" * 86)
print("T6 — PRODUCTION ATLAS TRAINING COST")
print("=" * 86)

print(f"Model                    : {summary['model']}")
print(f"GPU                      : {summary['gpu_type']}")
print(f"Charts                   : {n_charts}")
print(f"GPUs / chart             : {int(summary['gpus_per_chart'])}")

print()
print("Training steps per chart")
print(f"  spectral prefit        : {prefit_steps}")
print(f"  fixed-spectrum modal   : {modal_steps}")
print(f"  joint                  : {joint_steps}")
print(f"  total                  : {reported_total_steps}")

print()
print("Measured training expenditure")
print(
    f"  cumulative GPU time    : "
    f"{float(summary['total_gpu_hours']):.6f} GPU-h"
)
print(
    f"  mean / chart           : "
    f"{float(summary['mean_seconds_per_chart']):.2f} s"
)
print(
    f"  median / chart         : "
    f"{float(summary['median_seconds_per_chart']):.2f} s"
)
print(
    f"  range / chart          : "
    f"{float(summary['min_seconds_per_chart']):.2f}"
    f" -- "
    f"{float(summary['max_seconds_per_chart']):.2f} s"
)

print()
print("Scope:")
print(summary["scope"])


# =============================================================================
# LaTeX
# =============================================================================

gpu_hours_value = float(
    summary["total_gpu_hours"]
)

mean_seconds = float(
    summary["mean_seconds_per_chart"]
)

median_seconds = float(
    summary["median_seconds_per_chart"]
)

min_seconds = float(
    summary["min_seconds_per_chart"]
)

max_seconds = float(
    summary["max_seconds_per_chart"]
)


latex = rf"""\begin{{table}}[t]
\centering
\caption{{
Measured training expenditure for the production $N=76$ FULLC neural
atlas. The reported GPU time includes the spectral-prefit,
fixed-spectrum modal, and joint-training stages for all {n_charts}
charts. It excludes post-training Riccati shooting, validation
benchmarks, threshold studies, and ablation experiments.
}}
\label{{tab:production_atlas_training_cost}}
\small
\setlength{{\tabcolsep}}{{7pt}}
\begin{{tabular}}{{lr}}
\toprule
Quantity & Value \\
\midrule
GPU hardware
& {summary["gpu_type"]} \\

Number of charts
& {n_charts} \\

GPUs per chart
& {int(summary["gpus_per_chart"])} \\

Spectral-prefit steps / chart
& {prefit_steps:,} \\

Fixed-spectrum modal steps / chart
& {modal_steps:,} \\

Joint-training steps / chart
& {joint_steps:,} \\

Total training steps / chart
& {reported_total_steps:,} \\

\midrule

Cumulative training time
& {gpu_hours_value:.3f} GPU-h \\

Mean time / chart
& {mean_seconds:.1f} s \\

Median time / chart
& {median_seconds:.1f} s \\

Min--max time / chart
& {min_seconds:.1f}--{max_seconds:.1f} s \\

\bottomrule
\end{{tabular}}
\end{{table}}
"""

OUT_TEX.write_text(latex)

print()
print(f"Saved CSV : {OUT_CSV}")
print(f"Saved TeX : {OUT_TEX}")
