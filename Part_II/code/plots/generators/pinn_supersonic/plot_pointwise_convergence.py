#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_MACHS = {
    "subsonic": tuple(np.round(np.arange(0.1, 1.0, 0.1), 1)),
    "supersonic": tuple(np.round(np.arange(1.1, 2.0, 0.1), 1)),
}
OUTPUTS = {
    ("subsonic", "shooting_box"): (
        REPO_ROOT / "assets/classic_subsonic/article/convergence",
        "fig_subsonic_pointwise_box_convergence",
    ),
    ("subsonic", "shooting_accuracy"): (
        REPO_ROOT / "assets/classic_subsonic/article/convergence",
        "fig_subsonic_pointwise_integration_convergence",
    ),
    ("supersonic", "shooting_box"): (
        REPO_ROOT / "assets/classic_supersonic/article/convergence",
        "fig_supersonic_pointwise_box_convergence",
    ),
    ("supersonic", "shooting_accuracy"): (
        REPO_ROOT / "assets/classic_supersonic/article/convergence",
        "fig_supersonic_pointwise_integration_convergence",
    ),
}


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def validate_pointwise_frame(
    frame: pd.DataFrame,
    *,
    regime: str,
    sweep_type: str,
) -> tuple[pd.DataFrame, int]:
    required = {
        "run_id", "reference_run_id", "case_id", "regime", "sweep_type",
        "Mach", "alpha", "Ly", "accuracy_order", "accuracy_label",
        "abs_error_ci", "abs_error_omega_i", "complex_error_c",
        "mode_error_max_core", "mode_error_max_full",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pointwise convergence CSV is missing {missing}")
    selected = frame[
        frame["regime"].eq(regime) & frame["sweep_type"].eq(sweep_type)
    ].copy()
    if selected.empty:
        raise ValueError(f"No rows for {regime}/{sweep_type}")
    selected = selected.sort_values("core_threshold").drop_duplicates("run_id")
    points = selected[["case_id", "Mach", "alpha"]].drop_duplicates()
    if points.groupby("case_id")[["Mach", "alpha"]].size().ne(1).any():
        raise ValueError("A case_id maps to more than one (Mach, alpha) point")
    observed = tuple(sorted(np.round(points["Mach"].astype(float).unique(), 1)))
    if observed != EXPECTED_MACHS[regime]:
        raise ValueError(
            f"Expected exactly nine {regime} Mach values {EXPECTED_MACHS[regime]}, got {observed}"
        )
    references = selected[selected["run_id"].astype(str).eq(selected["reference_run_id"].astype(str))]
    if references.groupby("case_id").size().ne(1).any() or references["case_id"].nunique() != 9:
        raise ValueError("Each case must contain exactly one declared reference run")
    reference_points = references.set_index("run_id")[["case_id", "Mach", "alpha"]]
    for row in selected.itertuples(index=False):
        reference_id = str(row.reference_run_id)
        if reference_id not in reference_points.index:
            raise ValueError(f"Reference run {reference_id} is absent from the same sweep")
        reference = reference_points.loc[reference_id]
        if (
            str(reference["case_id"]) != str(row.case_id)
            or not np.isclose(float(reference["Mach"]), float(row.Mach))
            or not np.isclose(float(reference["alpha"]), float(row.alpha))
        ):
            raise ValueError(f"Cross-point reference detected for {row.run_id}")
    levels = selected.groupby("case_id")["run_id"].nunique()
    if levels.nunique() != 1:
        raise ValueError(f"Cases do not share a common number of levels: {levels.to_dict()}")
    return selected, int(levels.iloc[0])


def _metric(frame: pd.DataFrame, regime: str, modal: bool) -> np.ndarray:
    if modal:
        column = "mode_error_max_full"
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if regime == "subsonic":
        return np.nanmax(
            np.column_stack(
                [
                    pd.to_numeric(frame["abs_error_ci"], errors="coerce"),
                    pd.to_numeric(frame["abs_error_omega_i"], errors="coerce"),
                ]
            ),
            axis=1,
        )
    return pd.to_numeric(frame["complex_error_c"], errors="coerce").to_numpy(float)


def _x_values(frame: pd.DataFrame, sweep_type: str) -> tuple[np.ndarray, str]:
    if sweep_type == "shooting_box":
        return pd.to_numeric(frame["Ly"], errors="coerce").to_numpy(float), r"Half-box size $L_y$"
    if frame["regime"].iloc[0] == "supersonic":
        return pd.to_numeric(frame["max_step"], errors="coerce").to_numpy(float), "Maximum integration step"
    return pd.to_numeric(frame["accuracy_order"], errors="coerce").to_numpy(float), "Tolerance level"


def _title(regime: str, sweep_type: str, n_mach: int, n_levels: int) -> str:
    prefix = regime.capitalize()
    if sweep_type == "shooting_box":
        return f"{prefix} box convergence — {n_mach} Mach cases × {n_levels} box sizes"
    level_name = "tolerance levels" if regime == "subsonic" else "step sizes"
    return f"{prefix} integration convergence — {n_mach} Mach cases × {n_levels} {level_name}"


def build_pointwise_figure(frame: pd.DataFrame, *, regime: str, sweep_type: str):
    selected, n_levels = validate_pointwise_frame(frame, regime=regime, sweep_type=sweep_type)
    plotted = selected[
        ~selected["run_id"].astype(str).eq(selected["reference_run_id"].astype(str))
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 9))
    for color, (mach, group) in zip(colors, plotted.groupby("Mach", sort=True)):
        x, x_label = _x_values(group, sweep_type)
        order = np.argsort(x)
        for ax, modal in zip(axes, (False, True)):
            values = _metric(group, regime, modal)[order]
            values = np.where(values > 0.0, values, np.nan)
            ax.plot(x[order], values, marker="o", color=color, label=f"M={mach:.1f}")
            ax.set_xlabel(x_label)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.24)
    axes[0].set_title("(a) Spectral error")
    axes[1].set_title("(b) Modal error")
    axes[0].set_ylabel(
        r"max($|\Delta c_i|,|\Delta \omega_i|$)"
        if regime == "subsonic"
        else r"$|\Delta c|$"
    )
    axes[1].set_ylabel("Maximum full-domain relative modal error")
    axes[1].legend(ncols=3, fontsize=7, frameon=False)
    policy_values = selected.get("reference_policy", pd.Series(dtype=str)).dropna().unique()
    policy = str(policy_values[0]) if len(policy_values) == 1 else "most resolved run of the same case"
    fig.suptitle(_title(regime, sweep_type, 9, n_levels))
    fig.text(0.5, 0.005, f"Reference policy: {policy}; reference runs excluded.", ha="center", fontsize=8)
    return fig


def build_heatmap(frame: pd.DataFrame, *, regime: str, sweep_type: str):
    selected, n_levels = validate_pointwise_frame(frame, regime=regime, sweep_type=sweep_type)
    plotted = selected[
        ~selected["run_id"].astype(str).eq(selected["reference_run_id"].astype(str))
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    for ax, modal, label in zip(axes, (False, True), ("Spectral", "Modal")):
        rows = []
        labels = []
        for mach, group in plotted.groupby("Mach", sort=True):
            x, _ = _x_values(group, sweep_type)
            order = np.argsort(x)
            values = _metric(group, regime, modal)[order]
            rows.append(np.where(values > 0.0, np.log10(values), np.nan))
            labels.append(f"{mach:.1f}")
        image = ax.imshow(np.asarray(rows), aspect="auto", origin="lower", cmap="magma")
        ax.set_yticks(range(9), labels)
        ax.set_xlabel("Non-reference refinement level")
        ax.set_ylabel("Mach")
        ax.set_title(label + r" $\log_{10}$(error)")
        fig.colorbar(image, ax=ax, shrink=0.82)
    fig.suptitle(_title(regime, sweep_type, 9, n_levels) + " — heatmaps")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot strict nine-Mach pointwise convergence figures.")
    parser.add_argument("--errors-csv", type=Path, required=True)
    parser.add_argument("--heatmaps", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.errors_csv)
    for (regime, sweep_type), (output_dir, stem) in OUTPUTS.items():
        output_dir.mkdir(parents=True, exist_ok=True)
        figure = build_pointwise_figure(frame, regime=regime, sweep_type=sweep_type)
        figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(figure)
        if args.heatmaps:
            heatmap = build_heatmap(frame, regime=regime, sweep_type=sweep_type)
            heatmap.savefig(output_dir / f"{stem}_heatmap.pdf", bbox_inches="tight")
            heatmap.savefig(output_dir / f"{stem}_heatmap.png", dpi=300, bbox_inches="tight")
            plt.close(heatmap)


if __name__ == "__main__":
    main()
