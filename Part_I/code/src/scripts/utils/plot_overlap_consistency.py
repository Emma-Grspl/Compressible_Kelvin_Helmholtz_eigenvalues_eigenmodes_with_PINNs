#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path(
    "assets/pinn_subsonic/joint_ci_mode_global_validation_v1/"
    "merged/seam_dense_results.csv"
)

OUT = Path("assets/pinn_subsonic/paper_results_v1")
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    values = np.sort(values)
    probability = np.arange(1, values.size + 1) / values.size
    return values, probability


def summary(values: pd.Series, metric: str) -> dict[str, float | str]:
    array = values.to_numpy(float)
    array = array[np.isfinite(array)]

    return {
        "metric": metric,
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


df = pd.read_csv(INPUT)
df = df[df["success"].astype(str).str.lower().eq("true")].copy()

df["p_overlap_defect"] = np.maximum(
    1.0 - df["p_overlap_ab"].astype(float),
    1.0e-12,
)

metrics = pd.DataFrame(
    [
        summary(df["ci_abs_diff_ab"], "ci_abs_difference"),
        summary(df["ci_rel_diff_reg_ab"], "ci_regularized_relative_difference"),
        summary(df["p_rel_ab"], "pressure_relative_difference"),
        summary(df["p_overlap_defect"], "pressure_overlap_defect"),
    ]
)

metrics.to_csv(
    TABLE_DIR / "Table_atlas_overlap_consistency.csv",
    index=False,
)

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

x, y = ecdf(df["ci_abs_diff_ab"].to_numpy(float))
axes[0].plot(x, y, linewidth=2.0)
axes[0].set_xscale("log")
axes[0].set_xlabel(r"$|c_i^{(k)}-c_i^{(\ell)}|$")
axes[0].set_ylabel("Empirical cumulative probability")
axes[0].set_title("Spectral agreement in chart overlaps")
axes[0].grid(alpha=0.25)

x, y = ecdf(df["p_overlap_defect"].to_numpy(float))
axes[1].plot(x, y, linewidth=2.0)
axes[1].set_xscale("log")
axes[1].set_xlabel(r"$1-\mathcal{O}_p^{(k,\ell)}$")
axes[1].set_ylabel("Empirical cumulative probability")
axes[1].set_title("Pressure-mode agreement")
axes[1].grid(alpha=0.25)

fig.tight_layout()

for extension in ("pdf", "png"):
    fig.savefig(
        FIG_DIR / f"Fig_atlas_overlap_consistency.{extension}",
        dpi=300,
        bbox_inches="tight",
    )

plt.close(fig)

print(metrics.to_string(index=False))
print(f"\nSuccessful overlap points: {len(df)}")
print(f"Distinct seams: {df['seam_id'].nunique()}")
print(
    "Flagged seam outliers:",
    int(df["seam_outlier"].fillna(False).astype(bool).sum()),
)
