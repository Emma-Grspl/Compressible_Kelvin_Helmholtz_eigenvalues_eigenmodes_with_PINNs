#!/usr/bin/env python3
"""
Build final summary assets for the joint subsonic PINN + dense-GEP atlas.

Inputs
------
- joint atlas training plan;
- central-branch pointwise audit produced from the complete dense GEP spectra.

Outputs
-------
- canonical pointwise CSV;
- 49-chart domain map in (Mach, alpha);
- sparse spectral-anchor map;
- classical vs PINN+GEP growth-rate isolines;
- c_i absolute-error heatmap;
- mean modal-error heatmap over p, rho, u, v;
- fixed-Mach c_i cut comparing classical, direct PINN, and PINN+GEP.

No training or GEP solve is performed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

DEFAULT_TRAINING_PLAN = (
    "assets/pinn_subsonic/joint_ci_mode_atlas_v2/training_plan.tsv"
)
DEFAULT_CENTRAL_AUDIT = (
    "assets/pinn_subsonic/joint_ci_mode_full_gep_atlas_v2/"
    "central_branch_audit/atlas_pointwise_central_branch.csv"
)
DEFAULT_OUTPUT = (
    "assets/pinn_subsonic/joint_ci_mode_final_assets_v1"
)


def pick_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise KeyError(
            "None of the expected columns were found: "
            f"{list(candidates)}\nAvailable columns:\n"
            + "\n".join(f"  - {column}" for column in frame.columns)
        )
    return None


def numeric(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def alpha_from_eta(mach: np.ndarray | float, eta: np.ndarray | float):
    mach_array = np.asarray(mach, dtype=float)
    eta_array = np.asarray(eta, dtype=float)
    return eta_array * np.sqrt(np.clip(1.0 - mach_array**2, 0.0, None))


def canonicalize_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    aliases = {
        "M": "Mach",
        "mach": "Mach",
        "Eta": "eta",
    }
    result = frame.rename(
        columns={
            old: new
            for old, new in aliases.items()
            if old in frame.columns and new not in frame.columns
        }
    ).copy()

    mach_col = pick_column(result, ["Mach"])
    eta_col = pick_column(result, ["eta"])
    alpha_col = pick_column(result, ["alpha"])

    ci_ref_col = pick_column(
        result,
        [
            "ci_classic",
            "ci_ref",
            "ci_classical",
            "shoot_ci",
            "ci_shooting",
        ],
    )
    ci_seed_col = pick_column(
        result,
        [
            "ci_pinn",
            "ci_pred",
            "ci_seed",
            "ci_pinn_direct",
            "direct_ci",
        ],
    )
    ci_final_col = pick_column(
        result,
        [
            "central_ci",
            "central_max_ci",
            "central_gep_ci",
            "pinn_matched_ci",
            "gep_ci",
            "selected_ci",
            "ci_final",
        ],
    )

    modal_candidates = {
        "p_rel_final": [
            "central_p_rel_classic",
            "pinn_matched_p_rel_classic",
            "p_rel_final",
            "p_rel_classic",
            "p_rel",
        ],
        "rho_rel_final": [
            "central_rho_rel_classic",
            "pinn_matched_rho_rel_classic",
            "rho_rel_final",
            "rho_rel_classic",
            "rho_rel",
        ],
        "u_rel_final": [
            "central_u_rel_classic",
            "pinn_matched_u_rel_classic",
            "u_rel_final",
            "u_rel_classic",
            "u_rel",
        ],
        "v_rel_final": [
            "central_v_rel_classic",
            "pinn_matched_v_rel_classic",
            "v_rel_final",
            "v_rel_classic",
            "v_rel",
        ],
        "p_overlap_final": [
            "central_p_overlap_classic",
            "pinn_matched_p_overlap_classic",
            "p_overlap_final",
            "p_overlap_classic",
            "p_overlap",
        ],
    }

    chosen: dict[str, str | None] = {
        "Mach": mach_col,
        "eta": eta_col,
        "alpha": alpha_col,
        "ci_ref": ci_ref_col,
        "ci_seed": ci_seed_col,
        "ci_final": ci_final_col,
    }

    for output, candidates in modal_candidates.items():
        chosen[output] = pick_column(result, candidates, required=False)

    canonical = pd.DataFrame(
        {
            "Mach": numeric(result, mach_col),
            "eta": numeric(result, eta_col),
            "alpha": numeric(result, alpha_col),
            "ci_ref": numeric(result, ci_ref_col),
            "ci_seed": numeric(result, ci_seed_col),
            "ci_final": numeric(result, ci_final_col),
        }
    )

    for output in modal_candidates:
        canonical[output] = numeric(result, chosen[output])

    for optional in (
        "point_id",
        "chart_id",
        "sample_group",
        "continuation_status",
        "reference_status",
    ):
        if optional in result.columns:
            canonical[optional] = result[optional].astype(str)

    canonical["omega_ref"] = canonical["alpha"] * canonical["ci_ref"]
    canonical["omega_seed"] = canonical["alpha"] * canonical["ci_seed"]
    canonical["omega_final"] = canonical["alpha"] * canonical["ci_final"]
    canonical["ci_seed_abs_err"] = (
        canonical["ci_seed"] - canonical["ci_ref"]
    ).abs()
    canonical["ci_final_abs_err"] = (
        canonical["ci_final"] - canonical["ci_ref"]
    ).abs()
    canonical["omega_final_abs_err"] = (
        canonical["omega_final"] - canonical["omega_ref"]
    ).abs()

    modal_columns = [
        "p_rel_final",
        "rho_rel_final",
        "u_rel_final",
        "v_rel_final",
    ]
    canonical["modal_error_mean"] = canonical[modal_columns].mean(
        axis=1,
        skipna=False,
    )
    canonical["modal_error_max"] = canonical[modal_columns].max(
        axis=1,
        skipna=False,
    )

    canonical = canonical.replace([np.inf, -np.inf], np.nan)
    canonical = canonical.dropna(
        subset=["Mach", "eta", "alpha", "ci_ref", "ci_seed", "ci_final"]
    )
    canonical = (
        canonical.sort_values(["Mach", "alpha"])
        .drop_duplicates(["Mach", "eta"], keep="first")
        .reset_index(drop=True)
    )
    return canonical, chosen


def plot_atlas_map(training: pd.DataFrame, figure_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 8.0))

    for index, row in training.reset_index(drop=True).iterrows():
        mach_min = float(row["mach_min"])
        mach_max = float(row["mach_max"])
        eta_min = float(row["eta_min"])
        eta_max = float(row["eta_max"])

        mach_values = np.linspace(mach_min, mach_max, 80)
        lower = alpha_from_eta(mach_values, eta_min)
        upper = alpha_from_eta(mach_values, eta_max)

        polygon_mach = np.concatenate(
            [
                mach_values,
                mach_values[::-1],
                [mach_values[0]],
            ]
        )
        polygon_alpha = np.concatenate(
            [
                lower,
                upper[::-1],
                [lower[0]],
            ]
        )

        axis.plot(
            polygon_mach,
            polygon_alpha,
            linewidth=0.75,
            alpha=0.85,
        )

        center_mach = 0.5 * (mach_min + mach_max)
        center_eta = 0.5 * (eta_min + eta_max)
        center_alpha = float(alpha_from_eta(center_mach, center_eta))
        axis.text(
            center_mach,
            center_alpha,
            str(index + 1),
            fontsize=5.2,
            ha="center",
            va="center",
        )

    neutral_mach = np.linspace(0.0, 1.0, 500)
    neutral_alpha = np.sqrt(np.clip(1.0 - neutral_mach**2, 0.0, None))
    axis.plot(
        neutral_mach,
        neutral_alpha,
        linestyle="--",
        linewidth=1.2,
        label=r"$\alpha^2+M^2=1$",
    )

    axis.set(
        xlabel=r"Mach number $M$",
        ylabel=r"Wavenumber $\alpha$",
        title="Joint PINN atlas: 49 local charts",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figure_dir / "Fig_atlas_49_charts_Mach_alpha")


def load_anchor_points(training: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in training.iterrows():
        directory = Path(str(row["output_dir"]))
        path = directory / "ci_anchor_points.csv"
        if not path.is_file():
            continue
        anchors = pd.read_csv(path).copy()
        if "Mach" not in anchors.columns:
            continue
        if "alpha" not in anchors.columns:
            if "eta" not in anchors.columns:
                continue
            anchors["alpha"] = alpha_from_eta(
                pd.to_numeric(anchors["Mach"], errors="coerce"),
                pd.to_numeric(anchors["eta"], errors="coerce"),
            )
        anchors["chart_id"] = str(row["chart_id"])
        frames.append(anchors)

    if not frames:
        raise FileNotFoundError(
            "No ci_anchor_points.csv file could be read from the training plan."
        )

    result = pd.concat(frames, ignore_index=True, sort=False)
    result["Mach"] = pd.to_numeric(result["Mach"], errors="coerce")
    result["alpha"] = pd.to_numeric(result["alpha"], errors="coerce")
    if "eta" in result:
        result["eta"] = pd.to_numeric(result["eta"], errors="coerce")
    result = result.dropna(subset=["Mach", "alpha"])
    return result


def plot_anchor_map(
    training: pd.DataFrame,
    anchors: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 8.0))

    for _, row in training.iterrows():
        mach_values = np.linspace(
            float(row["mach_min"]),
            float(row["mach_max"]),
            60,
        )
        axis.plot(
            mach_values,
            alpha_from_eta(mach_values, float(row["eta_min"])),
            linewidth=0.35,
            alpha=0.25,
        )
        axis.plot(
            mach_values,
            alpha_from_eta(mach_values, float(row["eta_max"])),
            linewidth=0.35,
            alpha=0.25,
        )

    axis.scatter(
        anchors["Mach"],
        anchors["alpha"],
        s=9,
        alpha=0.62,
        linewidths=0.0,
        label=fr"$c_i$ reference anchors ({len(anchors)} chart-anchor rows)",
    )

    unique = anchors.drop_duplicates(["Mach", "alpha"])
    axis.scatter(
        unique["Mach"],
        unique["alpha"],
        s=22,
        facecolors="none",
        linewidths=0.45,
        label=f"Unique physical locations ({len(unique)})",
    )

    neutral_mach = np.linspace(0.0, 1.0, 500)
    axis.plot(
        neutral_mach,
        np.sqrt(np.clip(1.0 - neutral_mach**2, 0.0, None)),
        linestyle="--",
        linewidth=1.1,
        label=r"$\alpha^2+M^2=1$",
    )

    axis.set(
        xlabel=r"Mach number $M$",
        ylabel=r"Wavenumber $\alpha$",
        title=r"Sparse spectral supervision: $c_i$ anchors",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figure_dir / "Fig_sparse_ci_anchor_map_Mach_alpha")


def valid_triangulation(frame: pd.DataFrame, value: str):
    selected = frame[["Mach", "alpha", value]].dropna()
    selected = (
        selected.groupby(["Mach", "alpha"], as_index=False)[value]
        .mean()
        .sort_values(["Mach", "alpha"])
    )
    if len(selected) < 3:
        raise RuntimeError(f"Not enough finite points to plot {value}.")
    triangulation = mtri.Triangulation(
        selected["Mach"].to_numpy(float),
        selected["alpha"].to_numpy(float),
    )
    return selected, triangulation


def plot_isolines(canonical: pd.DataFrame, figure_dir: Path) -> None:
    levels = [0.05, 0.10, 0.15, 0.175]
    ref, ref_tri = valid_triangulation(canonical, "omega_ref")
    final, final_tri = valid_triangulation(canonical, "omega_final")

    fig, axis = plt.subplots(figsize=(9.2, 7.3))
    ref_contours = axis.tricontour(
        ref_tri,
        ref["omega_ref"].to_numpy(float),
        levels=levels,
        linewidths=1.7,
    )
    final_contours = axis.tricontour(
        final_tri,
        final["omega_final"].to_numpy(float),
        levels=levels,
        linewidths=1.25,
        linestyles="--",
    )
    axis.clabel(ref_contours, fmt=lambda value: f"{value:g}", fontsize=8)
    axis.clabel(final_contours, fmt=lambda value: f"{value:g}", fontsize=8)

    neutral_mach = np.linspace(0.0, 1.0, 500)
    axis.plot(
        neutral_mach,
        np.sqrt(np.clip(1.0 - neutral_mach**2, 0.0, None)),
        linestyle=":",
        linewidth=1.2,
    )

    axis.legend(
        handles=[
            Line2D([0], [0], linewidth=1.7, label="Classical shooting"),
            Line2D(
                [0],
                [0],
                linewidth=1.25,
                linestyle="--",
                label="PINN + GEP",
            ),
            Line2D(
                [0],
                [0],
                linewidth=1.2,
                linestyle=":",
                label="Neutral boundary",
            ),
        ],
        frameon=False,
    )
    axis.set(
        xlabel=r"Mach number $M$",
        ylabel=r"Wavenumber $\alpha$",
        title=r"Growth-rate isolines $\omega_i=\alpha c_i$",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.18)
    fig.tight_layout()
    save_figure(fig, figure_dir / "Fig_isolines_classical_vs_PINN_GEP")


def plot_heatmap(
    canonical: pd.DataFrame,
    column: str,
    title: str,
    colorbar_label: str,
    stem: Path,
    *,
    log10: bool,
) -> None:
    selected, triangulation = valid_triangulation(canonical, column)
    values = selected[column].to_numpy(float)
    if log10:
        values = np.log10(np.clip(values, 1.0e-14, None))

    fig, axis = plt.subplots(figsize=(9.0, 7.2))
    collection = axis.tripcolor(
        triangulation,
        values,
        shading="gouraud",
    )
    colorbar = fig.colorbar(collection, ax=axis)
    colorbar.set_label(
        f"log10({colorbar_label})" if log10 else colorbar_label
    )

    neutral_mach = np.linspace(0.0, 1.0, 500)
    axis.plot(
        neutral_mach,
        np.sqrt(np.clip(1.0 - neutral_mach**2, 0.0, None)),
        linestyle="--",
        linewidth=1.0,
    )
    axis.set(
        xlabel=r"Mach number $M$",
        ylabel=r"Wavenumber $\alpha$",
        title=title,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    fig.tight_layout()
    save_figure(fig, stem)


def plot_fixed_mach_cut(
    canonical: pd.DataFrame,
    requested_mach: float,
    figure_dir: Path,
) -> float:
    available = np.sort(canonical["Mach"].dropna().unique())
    if len(available) == 0:
        raise RuntimeError("No Mach values are available.")
    selected_mach = float(available[np.argmin(np.abs(available - requested_mach))])

    tolerance = max(1.0e-10, 1.0e-8 * max(1.0, abs(selected_mach)))
    cut = canonical.loc[
        np.abs(canonical["Mach"] - selected_mach) <= tolerance
    ].sort_values("alpha")

    if len(cut) < 3:
        distances = np.abs(canonical["Mach"] - requested_mach)
        nearest = canonical.loc[distances.nsmallest(30).index]
        selected_mach = float(nearest["Mach"].mode().iloc[0])
        cut = canonical.loc[
            np.isclose(canonical["Mach"], selected_mach)
        ].sort_values("alpha")

    fig, axis = plt.subplots(figsize=(8.2, 5.6))
    axis.plot(
        cut["alpha"],
        cut["ci_ref"],
        marker="o",
        markersize=3.2,
        linewidth=1.5,
        label="Classical shooting",
    )
    axis.plot(
        cut["alpha"],
        cut["ci_seed"],
        marker="s",
        markersize=3.0,
        linewidth=1.2,
        label="Direct PINN",
    )
    axis.plot(
        cut["alpha"],
        cut["ci_final"],
        marker="^",
        markersize=3.0,
        linewidth=1.2,
        label="PINN + GEP",
    )
    axis.set(
        xlabel=r"Wavenumber $\alpha$",
        ylabel=r"$c_i$",
        title=fr"Spectral cut at $M={selected_mach:.6g}$",
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figure_dir / "Fig_ci_cut_fixed_Mach")
    return selected_mach


def metric_summary(frame: pd.DataFrame, columns: Iterable[str]) -> dict:
    output = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            output[column] = {"n": 0}
            continue
        output[column] = {
            "n": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-plan", default=DEFAULT_TRAINING_PLAN)
    parser.add_argument("--central-audit", default=DEFAULT_CENTRAL_AUDIT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--fixed-mach", type=float, default=0.50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    for directory in (data_dir, figure_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(args.training_plan, sep="\t")
    required_training = {
        "chart_id",
        "output_dir",
        "mach_min",
        "mach_max",
        "eta_min",
        "eta_max",
    }
    missing_training = sorted(required_training.difference(training.columns))
    if missing_training:
        raise KeyError(
            f"Training plan is missing columns {missing_training}"
        )

    source = pd.read_csv(args.central_audit)
    canonical, chosen_columns = canonicalize_audit(source)
    canonical_path = data_dir / "validation_pointwise_canonical.csv"
    canonical.to_csv(canonical_path, index=False)

    anchors = load_anchor_points(training)
    anchors.to_csv(
        data_dir / "spectral_supervision_anchors_explicit.csv",
        index=False,
    )
    training.to_csv(
        data_dir / "atlas_catalog_49_charts.tsv",
        sep="\t",
        index=False,
    )

    plot_atlas_map(training, figure_dir)
    plot_anchor_map(training, anchors, figure_dir)
    plot_isolines(canonical, figure_dir)
    plot_heatmap(
        canonical,
        "ci_final_abs_err",
        r"Absolute spectral error: PINN + GEP versus classical shooting",
        r"$|c_i^{PINN+GEP}-c_i^{shoot}|$",
        figure_dir / "Fig_ci_error_heatmap_PINN_GEP_vs_classical",
        log10=True,
    )

    if canonical["modal_error_mean"].notna().sum() >= 3:
        plot_heatmap(
            canonical,
            "modal_error_mean",
            r"Mean modal error over $p,\rho,u,v$",
            r"$\frac{1}{4}\sum_f \varepsilon_f$",
            figure_dir / "Fig_modal_error_mean_heatmap_PINN_GEP_vs_classical",
            log10=True,
        )

    selected_mach = plot_fixed_mach_cut(
        canonical,
        args.fixed_mach,
        figure_dir,
    )

    summary = {
        "source_training_plan": str(args.training_plan),
        "source_central_audit": str(args.central_audit),
        "n_charts": int(training["chart_id"].nunique()),
        "n_canonical_points": int(len(canonical)),
        "n_anchor_rows": int(len(anchors)),
        "n_unique_anchor_locations": int(
            len(anchors.drop_duplicates(["Mach", "alpha"]))
        ),
        "fixed_mach_requested": float(args.fixed_mach),
        "fixed_mach_selected": selected_mach,
        "resolved_source_columns": chosen_columns,
        "metrics": metric_summary(
            canonical,
            [
                "ci_seed_abs_err",
                "ci_final_abs_err",
                "omega_final_abs_err",
                "p_rel_final",
                "rho_rel_final",
                "u_rel_final",
                "v_rel_final",
                "p_overlap_final",
                "modal_error_mean",
                "modal_error_max",
            ],
        ),
    }
    (output_dir / "asset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"metric": metric, **values}
            for metric, values in summary["metrics"].items()
        ]
    ).to_csv(
        table_dir / "Table_global_PINN_GEP_metrics.csv",
        index=False,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Wrote:", output_dir)


if __name__ == "__main__":
    main()
