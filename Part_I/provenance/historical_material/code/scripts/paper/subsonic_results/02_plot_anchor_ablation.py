#!/usr/bin/env python3

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path.cwd()
OUT = ROOT / "assets/pinn_subsonic/paper_results_v1"
FIG_DIR = OUT / "figures"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"

for directory in (FIG_DIR, DATA_DIR, TABLE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


RUNS = {
    "Physics only": ROOT
    / "model_saved/"
    "kh_subsonic_fixed_mach_M05_alpha010_080_riccati_pure_physics_reference",

    "4 anchors": ROOT
    / "model_saved/"
    "kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference/"
    "hybrid_ci4_fixed",

    "8 anchors": ROOT
    / "model_saved/"
    "kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference/"
    "hybrid_ci8_fixed",

    "16 anchors": ROOT
    / "model_saved/"
    "kh_subsonic_fixed_mach_M05_alpha010_080_ci_sparse_reference/"
    "hybrid_ci16_fixed",
}

REFERENCE_CANDIDATES = [
    ROOT
    / "assets/pinn_subsonic/mach_fixed_candidates/"
    "hybrid_8pt/ci/ci_curve_vs_reference.csv",

    ROOT
    / "plot_presentation/subsonic_pinn/"
    "ci_supervision_vs_physics/ci_curve_vs_reference.csv",
]


def load_plot_module():
    path = ROOT / "scripts/plot_kh_subsonic_ci_supervision_vs_physics.py"

    spec = importlib.util.spec_from_file_location(
        "ci_supervision_plot_module",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_column(
    columns: list[str],
    required_tokens: tuple[str, ...],
    forbidden_tokens: tuple[str, ...] = (),
) -> str:
    for column in columns:
        lower = column.lower()

        if (
            all(token in lower for token in required_tokens)
            and not any(token in lower for token in forbidden_tokens)
        ):
            return column

    raise KeyError(
        f"Cannot identify column with tokens {required_tokens}; "
        f"available columns: {columns}"
    )


def read_reference() -> tuple[np.ndarray, np.ndarray]:
    path = next(
        (candidate for candidate in REFERENCE_CANDIDATES if candidate.exists()),
        None,
    )

    if path is None:
        raise FileNotFoundError(
            "No existing ci_curve_vs_reference.csv was found."
        )

    df = pd.read_csv(path)
    columns = list(df.columns)

    alpha_col = infer_column(columns, ("alpha",))

    reference_candidates = [
        column
        for column in columns
        if any(
            token in column.lower()
            for token in ("reference", "classic", "shoot")
        )
        and "error" not in column.lower()
        and "alpha" not in column.lower()
    ]

    if not reference_candidates:
        raise KeyError(
            f"No classical/reference column found in {path}; "
            f"columns={columns}"
        )

    reference_col = reference_candidates[0]

    reference = (
        df[[alpha_col, reference_col]]
        .dropna()
        .sort_values(alpha_col)
        .drop_duplicates(alpha_col)
    )

    print(f"Reference file: {path}")
    print(f"Reference columns: {alpha_col}, {reference_col}")

    return (
        reference[alpha_col].to_numpy(float),
        reference[reference_col].to_numpy(float),
    )


def read_anchor_locations(run_dir: Path) -> np.ndarray:
    config_path = run_dir / "config.csv"
    config = pd.read_csv(config_path).iloc[0]

    enabled = bool(config.get("enable_classic_ci_supervision", False))
    if not enabled:
        return np.empty(0)

    value = config.get("ci_supervision_fixed_alphas", "()")

    if pd.isna(value):
        return np.empty(0)

    parsed = ast.literal_eval(str(value))
    return np.asarray(parsed, dtype=float)


def summarize_error(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]

    return {
        "n": int(finite.size),
        "mae": float(np.mean(finite)),
        "median_abs_error": float(np.median(finite)),
        "p90_abs_error": float(np.quantile(finite, 0.90)),
        "p95_abs_error": float(np.quantile(finite, 0.95)),
        "max_abs_error": float(np.max(finite)),
    }


def main() -> None:
    module = load_plot_module()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    alpha = np.linspace(0.1, 0.8, 401)

    alpha_reference, ci_reference_raw = read_reference()
    ci_reference = np.interp(
        alpha,
        alpha_reference,
        ci_reference_raw,
    )

    result = pd.DataFrame(
        {
            "alpha": alpha,
            "ci_reference": ci_reference,
        }
    )

    metrics: list[dict[str, float | str]] = []
    anchors_by_run: dict[str, np.ndarray] = {}

    for label, run_dir in RUNS.items():
        if not (run_dir / "model_best.pt").exists():
            raise FileNotFoundError(run_dir / "model_best.pt")

        _, prediction = module.predict_ci_curve(
            run_dir,
            alpha,
            device,
        )

        prediction = np.asarray(prediction, dtype=float).reshape(-1)

        if prediction.size != alpha.size:
            raise ValueError(
                f"{label}: prediction size {prediction.size}, "
                f"expected {alpha.size}"
            )

        key = (
            label.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        absolute_error = np.abs(prediction - ci_reference)

        result[f"ci_{key}"] = prediction
        result[f"abs_error_{key}"] = absolute_error

        row = {"configuration": label}
        row.update(summarize_error(absolute_error))
        metrics.append(row)

        anchors_by_run[label] = read_anchor_locations(run_dir)

    result.to_csv(
        DATA_DIR / "single_case_anchor_ablation_pointwise.csv",
        index=False,
    )

    pd.DataFrame(metrics).to_csv(
        TABLE_DIR / "Table_single_case_anchor_ablation.csv",
        index=False,
    )

    # Figure 1: curves and pointwise absolute errors
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    axes[0].plot(
        alpha,
        ci_reference,
        linewidth=2.4,
        label="Classical reference",
    )

    for label in RUNS:
        key = label.lower().replace(" ", "_").replace("-", "_")
        axes[0].plot(
            alpha,
            result[f"ci_{key}"],
            linewidth=1.5,
            label=label,
        )

        anchors = anchors_by_run[label]
        if anchors.size:
            axes[0].scatter(
                anchors,
                np.interp(anchors, alpha, ci_reference),
                s=20,
                marker="o",
                zorder=5,
            )

    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$c_i$")
    axes[0].set_title(r"Growth-rate prediction at $M=0.5$")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    for label in RUNS:
        key = label.lower().replace(" ", "_").replace("-", "_")
        axes[1].plot(
            alpha,
            result[f"abs_error_{key}"],
            linewidth=1.5,
            label=label,
        )

    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$|c_i^{\mathrm{PINN}}-c_i^{\mathrm{ref}}|$")
    axes[1].set_title("Pointwise spectral error")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()

    for extension in ("pdf", "png"):
        fig.savefig(
            FIG_DIR / f"Fig_single_case_anchor_ablation.{extension}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)

    # Figure 2: error histograms
    all_errors = np.concatenate(
        [
            result[
                "abs_error_"
                + label.lower().replace(" ", "_").replace("-", "_")
            ].to_numpy()
            for label in RUNS
        ]
    )

    positive = all_errors[all_errors > 0]
    lower = max(float(np.min(positive)), 1.0e-8)
    upper = float(np.max(positive))
    bins = np.logspace(np.log10(lower), np.log10(upper), 32)

    fig, ax = plt.subplots(figsize=(6.3, 4.4))

    for label in RUNS:
        key = label.lower().replace(" ", "_").replace("-", "_")
        values = result[f"abs_error_{key}"].to_numpy()

        ax.hist(
            np.clip(values, lower, None),
            bins=bins,
            histtype="step",
            linewidth=1.7,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$|c_i^{\mathrm{PINN}}-c_i^{\mathrm{ref}}|$")
    ax.set_ylabel("Number of evaluation points")
    ax.set_title("Distribution of spectral errors")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    for extension in ("pdf", "png"):
        fig.savefig(
            FIG_DIR
            / f"Fig_single_case_anchor_error_histograms.{extension}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)

    print("\nMetrics:")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"\nOutputs written under: {OUT}")


if __name__ == "__main__":
    main()
