#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from scripts.evaluation.blumen_isoline_common import (
    POSITIVE_LEVELS,
    attach_original_digitization_order,
)


OK_STATUSES = {"converged_center", "converged_scan_point", "converged_root"}
OUTPUT_COLUMNS = [
    "blumen_row_id",
    "source_row_id",
    "digitization_order",
    "curve_id",
    "curve_key",
    "curve_label",
    "Mach",
    "alpha_blumen",
    "target_ci",
    "alpha_classical",
    "delta_alpha",
    "abs_delta_alpha",
    "classical_cr",
    "classical_ci",
    "delta_ci",
    "residual_norm",
    "status",
    "n_solver_calls",
    "bracket_alpha_left",
    "bracket_alpha_right",
    "message",
]


def numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def load_positive_targets(path: Path, raw_blumen_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = numeric(
        frame,
        (
            "task_index",
            "blumen_row_id",
            "source_row_id",
            "Mach",
            "alpha",
            "blumen_ci",
        ),
    )
    frame = frame.loc[
        np.isfinite(frame["Mach"])
        & np.isfinite(frame["alpha"])
        & np.isfinite(frame["blumen_ci"])
        & (frame["blumen_ci"] > 0.0)
    ].copy()
    positive_mask = np.zeros(len(frame), dtype=bool)
    for level in POSITIVE_LEVELS:
        positive_mask |= np.isclose(
            frame["blumen_ci"], level, rtol=0.0, atol=1e-12
        )
    frame = frame.loc[positive_mask].copy()
    order_column = "source_row_id" if "source_row_id" in frame.columns else "blumen_row_id"
    frame = frame.sort_values(order_column, kind="stable").reset_index(drop=True)
    frame["true_isoline_task_index"] = np.arange(len(frame), dtype=int)
    if len(frame) != 117:
        raise ValueError(f"Expected 117 positive Blumen targets, found {len(frame)}.")
    return attach_original_digitization_order(frame, raw_blumen_csv)


def load_results(targets: pd.DataFrame, result_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for _, target in targets.iterrows():
        task_index = int(target["true_isoline_task_index"])
        path = result_root / f"point_{task_index:03d}" / "result.csv"
        if path.is_file() and path.stat().st_size > 0:
            result = pd.read_csv(path)
            if len(result) == 1:
                row = result.iloc[0].to_dict()
                row["digitization_order"] = int(target["digitization_order"])
                rows.append(row)
                continue
        row = target.to_dict()
        row.update(
            {
                "status": "missing_result",
                "alpha_blumen": float(target["alpha"]),
                "target_ci": float(target["blumen_ci"]),
                "alpha_classical": np.nan,
                "delta_alpha": np.nan,
                "classical_cr": np.nan,
                "classical_ci": np.nan,
                "delta_ci": np.nan,
                "residual_norm": np.nan,
                "n_solver_calls": 0,
                "bracket_alpha_left": np.nan,
                "bracket_alpha_right": np.nan,
                "message": f"Missing result file: {path}",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def contiguous_segments(frame: pd.DataFrame, order_column: str) -> list[pd.DataFrame]:
    ordered = frame.sort_values(order_column, kind="stable").copy()
    segments: list[pd.DataFrame] = []
    current: list[int] = []
    for index, ok in zip(ordered.index, ordered["converged"], strict=True):
        if bool(ok):
            current.append(index)
        elif current:
            segments.append(ordered.loc[current].copy())
            current = []
    if current:
        segments.append(ordered.loc[current].copy())
    return segments


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise-csv", type=Path, required=True)
    parser.add_argument("--raw-blumen-csv", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--article-root", type=Path, required=True)
    parser.add_argument("--ci-tolerance", type=float, default=1e-6)
    parser.add_argument("--residual-tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    article_root = args.article_root.resolve()
    article_root.mkdir(parents=True, exist_ok=True)

    targets = load_positive_targets(
        args.pointwise_csv.resolve(),
        args.raw_blumen_csv.resolve(),
    )
    data = load_results(targets, args.result_root.resolve())
    data = numeric(
        data,
        (
            "true_isoline_task_index",
            "blumen_row_id",
            "source_row_id",
            "digitization_order",
            "Mach",
            "alpha",
            "alpha_blumen",
            "blumen_ci",
            "target_ci",
            "alpha_classical",
            "delta_alpha",
            "classical_cr",
            "classical_ci",
            "delta_ci",
            "residual_norm",
            "n_solver_calls",
            "bracket_alpha_left",
            "bracket_alpha_right",
        ),
    )
    if "alpha_blumen" not in data.columns:
        data["alpha_blumen"] = data["alpha"]
    if "target_ci" not in data.columns:
        data["target_ci"] = data["blumen_ci"]
    data["delta_ci"] = data["classical_ci"] - data["target_ci"]
    data["delta_alpha"] = data["alpha_classical"] - data["alpha_blumen"]
    data["abs_delta_alpha"] = data["delta_alpha"].abs()

    data["converged"] = (
        data["status"].astype(str).isin(OK_STATUSES)
        & np.isfinite(data["alpha_classical"])
        & np.isfinite(data["classical_ci"])
        & np.isfinite(data["residual_norm"])
        & (data["delta_ci"].abs() <= args.ci_tolerance)
        & (data["residual_norm"] <= args.residual_tolerance)
    )
    validation_failure = data["status"].astype(str).isin(OK_STATUSES) & ~data["converged"]
    data.loc[validation_failure, "message"] = (
        "Candidate root failed final ci/residual validation: "
        + data.loc[validation_failure, "message"].fillna("").astype(str)
    )
    data.loc[validation_failure, "status"] = "failed_final_validation"
    for column in OUTPUT_COLUMNS:
        if column not in data:
            data[column] = np.nan

    order_column = "digitization_order"
    levels = sorted(float(value) for value in data["target_ci"].dropna().unique())
    cmap = plt.get_cmap("viridis")
    level_colors = {
        level: cmap(0.08 + 0.84 * index / max(1, len(levels) - 1))
        for index, level in enumerate(levels)
    }

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16.0, 6.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    ax_overlay, ax_error = axes

    level_legend: list[Line2D] = []
    for level in levels:
        level_frame = data.loc[np.isclose(data["target_ci"], level, atol=1e-12)].copy()
        color = level_colors[level]

        # Never join different curve keys, even when they share the same ci level.
        for _, curve in level_frame.groupby("curve_key", sort=False):
            ordered_curve = curve.sort_values(order_column, kind="stable")
            ax_overlay.plot(
                ordered_curve["Mach"],
                ordered_curve["alpha_blumen"],
                color=color,
                linestyle=(0, (2.0, 2.2)),
                linewidth=1.1,
                marker="o",
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                zorder=2,
            )
            for segment in contiguous_segments(curve, order_column):
                ax_overlay.plot(
                    segment["Mach"],
                    segment["alpha_classical"],
                    color=color,
                    linewidth=2.1,
                    marker="o",
                    markersize=3.8,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    zorder=3,
                )

        failed = level_frame.loc[~level_frame["converged"]]
        if not failed.empty:
            ax_overlay.scatter(
                failed["Mach"],
                failed["alpha_blumen"],
                marker="x",
                s=58,
                color="black",
                linewidths=1.5,
                zorder=5,
            )

        level_legend.append(
            Line2D([0], [0], color=color, linewidth=2.1, label=rf"$c_i={level:g}$")
        )

    style_legend = [
        Line2D(
            [0], [0], linestyle=(0, (2.0, 2.2)), marker="o", markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=1.25, markersize=6,
            label="Blumen digitization",
        ),
        Line2D(
            [0], [0], color="black", linewidth=2.1, marker="o", markersize=4,
            label="Classical root locus",
        ),
    ]
    if (~data["converged"]).any():
        style_legend.append(
            Line2D(
                [0], [0], linestyle="none", marker="x", color="black",
                markeredgewidth=1.5, markersize=7, label="No converged alpha root",
            )
        )

    first_legend = ax_overlay.legend(
        handles=level_legend,
        title="Isoline level",
        loc="upper right",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncols=2,
    )
    ax_overlay.add_artist(first_legend)
    ax_overlay.legend(handles=style_legend, loc="lower left", frameon=True, fontsize=8)
    ax_overlay.set_xlabel(r"Mach number $M$")
    ax_overlay.set_ylabel(r"Wavenumber $\alpha$")
    ax_overlay.set_title("(a) Digitized Blumen points and classical isolines")
    ax_overlay.grid(True, alpha=0.20)

    converged = data.loc[data["converged"]].copy()
    if converged.empty:
        raise RuntimeError("No converged true-isoline roots were found.")
    positive_errors = converged.loc[converged["abs_delta_alpha"] > 0.0, "abs_delta_alpha"]
    vmin = max(float(positive_errors.min()) if len(positive_errors) else 1e-8, 1e-8)
    vmax = max(float(converged["abs_delta_alpha"].max()), vmin * 10.0)

    scatter = ax_error.scatter(
        converged["Mach"],
        converged["alpha_blumen"],
        c=converged["abs_delta_alpha"],
        cmap="inferno",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        s=35,
        edgecolors="none",
        zorder=2,
    )
    q90 = float(converged["abs_delta_alpha"].quantile(0.90))
    worst = converged.loc[converged["abs_delta_alpha"] >= q90]
    ax_error.scatter(
        worst["Mach"],
        worst["alpha_blumen"],
        s=68,
        marker="o",
        facecolors="none",
        edgecolors="cyan",
        linewidths=1.25,
        label="Top 10% errors",
        zorder=3,
    )
    failed_all = data.loc[~data["converged"]]
    if not failed_all.empty:
        ax_error.scatter(
            failed_all["Mach"],
            failed_all["alpha_blumen"],
            marker="x",
            s=58,
            color="black",
            linewidths=1.5,
            label="No converged alpha root",
            zorder=4,
        )
    colorbar = figure.colorbar(scatter, ax=ax_error, pad=0.018)
    colorbar.set_label(r"$|\alpha_{\mathrm{classical}}-\alpha_{\mathrm{Blumen}}|$")
    ax_error.set_xlabel(r"Mach number $M$")
    ax_error.set_title("(b) Geometric error at the Blumen Mach values")
    ax_error.grid(True, alpha=0.20)
    ax_error.legend(loc="best", frameon=True, fontsize=8)

    x_min = float(data["Mach"].min())
    x_max = float(data["Mach"].max())
    y_values = pd.concat([data["alpha_blumen"], data["alpha_classical"]], ignore_index=True)
    y_min = max(0.0, float(y_values.min(skipna=True)))
    y_max = float(y_values.max(skipna=True))
    x_margin = 0.025 * max(x_max - x_min, 1e-3)
    y_margin = 0.035 * max(y_max - y_min, 1e-3)
    for axis in axes:
        axis.set_xlim(x_min - x_margin, x_max + x_margin)
        axis.set_ylim(max(0.0, y_min - y_margin), y_max + y_margin)

    output_base = article_root / "Fig_supersonic_blumen_vs_classical_isolines_1x2"
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(figure)

    output_data = data[OUTPUT_COLUMNS + ["converged"]].copy()
    output_data.to_csv(
        article_root / "supersonic_blumen_true_classical_isolines.csv",
        index=False,
    )
    failures = output_data.loc[~output_data["converged"]].copy()
    failures.to_csv(
        article_root / "supersonic_blumen_true_classical_isoline_failures.csv",
        index=False,
    )

    summary_rows = []
    for level, group in data.groupby("target_ci", sort=True):
        ok = group.loc[group["converged"]]
        summary_rows.append(
            {
                "target_ci": float(level),
                "n_points": int(len(group)),
                "n_converged": int(group["converged"].sum()),
                "completion": float(group["converged"].mean()),
                "mean_signed_delta_alpha": float(ok["delta_alpha"].mean()) if len(ok) else np.nan,
                "mae_alpha": float(ok["abs_delta_alpha"].mean()) if len(ok) else np.nan,
                "rmse_alpha": (
                    float(np.sqrt(np.mean(np.square(ok["delta_alpha"]))))
                    if len(ok)
                    else np.nan
                ),
                "median_abs_delta_alpha": (
                    float(ok["abs_delta_alpha"].median()) if len(ok) else np.nan
                ),
                "max_abs_delta_alpha": float(ok["abs_delta_alpha"].max()) if len(ok) else np.nan,
                "max_residual": float(ok["residual_norm"].max()) if len(ok) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        article_root / "supersonic_blumen_true_classical_isoline_summary_by_level.csv",
        index=False,
    )

    metadata = {
        "n_targets": int(len(data)),
        "n_converged": int(data["converged"].sum()),
        "n_failed": int((~data["converged"]).sum()),
        "status": "PASS" if bool(data["converged"].all()) else "INCOMPLETE",
    }
    (
        article_root / "supersonic_blumen_true_classical_isolines_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== TRUE POSITIVE BLUMEN ISOLINE ASSETS ===")
    print(f"Converged : {metadata['n_converged']}/{metadata['n_targets']}")
    print(f"Failed    : {metadata['n_failed']}")
    print(f"Status    : {metadata['status']}")
    print()
    print(summary.to_string(index=False))
    print()
    print("PDF       :", output_base.with_suffix(".pdf"))
    print("PNG       :", output_base.with_suffix(".png"))
    print("CSV       :", article_root / "supersonic_blumen_true_classical_isolines.csv")
    print(
        "Failures  :",
        article_root / "supersonic_blumen_true_classical_isoline_failures.csv",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
