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


POINTWISE_OK = {"accepted_root", "accepted", "converged", "success"}
ISOLINE_OK = {"converged_center", "converged_scan_point", "converged_root"}


def numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def load_pointwise(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Mach", "alpha", "blumen_ci", "classical_ci", "status"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing pointwise columns: {missing}")

    frame = numeric(
        frame,
        (
            "task_index",
            "blumen_row_id",
            "source_row_id",
            "Mach",
            "alpha",
            "blumen_ci",
            "classical_cr",
            "classical_ci",
            "residual_norm",
        ),
    )
    frame = frame.loc[
        np.isfinite(frame["Mach"])
        & np.isfinite(frame["alpha"])
        & np.isfinite(frame["blumen_ci"])
        & (frame["blumen_ci"] > 0.0)
    ].copy()

    if "blumen_row_id" not in frame.columns:
        frame["blumen_row_id"] = np.arange(len(frame), dtype=int)
    if "source_row_id" not in frame.columns:
        frame["source_row_id"] = frame["blumen_row_id"]
    if "curve_key" not in frame.columns:
        if "curve_id" in frame.columns:
            frame["curve_key"] = frame["curve_id"].astype(str)
        else:
            raise KeyError(
                "The pointwise CSV has neither curve_key nor curve_id. "
                "Branch-safe line reconstruction is therefore impossible."
            )

    frame = frame.sort_values("source_row_id", kind="stable").reset_index(drop=True)
    frame["true_isoline_task_index"] = np.arange(len(frame), dtype=int)
    frame["pointwise_converged"] = (
        frame["status"].astype(str).str.strip().isin(POINTWISE_OK)
        & np.isfinite(frame["classical_ci"])
    )
    frame["delta_ci"] = frame["classical_ci"] - frame["blumen_ci"]
    frame["abs_delta_ci"] = frame["delta_ci"].abs()
    return frame


def load_isoline_results(pointwise: pd.DataFrame, result_root: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for _, target in pointwise.iterrows():
        task = int(target["true_isoline_task_index"])
        path = result_root / f"point_{task:03d}" / "result.csv"

        base = target.to_dict()
        base.update(
            {
                "alpha_blumen": float(target["alpha"]),
                "target_ci": float(target["blumen_ci"]),
                "alpha_classical": np.nan,
                "delta_alpha": np.nan,
                "classical_cr_at_isoline": np.nan,
                "classical_ci_at_isoline": np.nan,
                "isoline_residual_norm": np.nan,
                "isoline_status": "missing_result",
                "isoline_message": f"Missing result file: {path}",
            }
        )

        if path.is_file() and path.stat().st_size > 0:
            result = pd.read_csv(path)
            if len(result) == 1:
                row = result.iloc[0].to_dict()
                base.update(
                    {
                        "alpha_blumen": row.get("alpha_blumen", target["alpha"]),
                        "target_ci": row.get("target_ci", target["blumen_ci"]),
                        "alpha_classical": row.get("alpha_classical", np.nan),
                        "delta_alpha": row.get("delta_alpha", np.nan),
                        "classical_cr_at_isoline": row.get("classical_cr", np.nan),
                        "classical_ci_at_isoline": row.get("classical_ci", np.nan),
                        "isoline_residual_norm": row.get("residual_norm", np.nan),
                        "isoline_status": row.get("status", "unknown"),
                        "isoline_message": row.get("message", ""),
                        "n_spectral_solves": row.get("n_spectral_solves", np.nan),
                        "alpha_bracket_lower": row.get("alpha_bracket_lower", np.nan),
                        "alpha_bracket_upper": row.get("alpha_bracket_upper", np.nan),
                    }
                )
        rows.append(base)

    data = pd.DataFrame(rows)
    data = numeric(
        data,
        (
            "Mach",
            "alpha",
            "alpha_blumen",
            "blumen_ci",
            "target_ci",
            "classical_ci",
            "alpha_classical",
            "delta_alpha",
            "classical_cr_at_isoline",
            "classical_ci_at_isoline",
            "isoline_residual_norm",
            "n_spectral_solves",
            "alpha_bracket_lower",
            "alpha_bracket_upper",
        ),
    )
    data["isoline_converged"] = (
        data["isoline_status"].astype(str).isin(ISOLINE_OK)
        & np.isfinite(data["alpha_classical"])
    )
    data["abs_delta_alpha"] = data["delta_alpha"].abs()
    return data


def split_converged_segments(
    curve: pd.DataFrame,
    order_column: str,
    x_column: str = "Mach",
    y_column: str = "alpha_classical",
) -> list[pd.DataFrame]:
    ordered = curve.sort_values(order_column, kind="stable").copy()
    segments: list[pd.DataFrame] = []
    current_indices: list[int] = []

    for index, row in ordered.iterrows():
        ok = bool(row["isoline_converged"])
        if not ok:
            if current_indices:
                segments.append(ordered.loc[current_indices].copy())
                current_indices = []
            continue

        if current_indices:
            previous = ordered.loc[current_indices[-1]]
            jump = float(
                np.hypot(
                    float(row[x_column]) - float(previous[x_column]),
                    float(row[y_column]) - float(previous[y_column]),
                )
            )

            history = ordered.loc[current_indices]
            if len(history) >= 3:
                dx = np.diff(history[x_column].to_numpy(float))
                dy = np.diff(history[y_column].to_numpy(float))
                distances = np.hypot(dx, dy)
                positive = distances[distances > 0]
                typical = float(np.median(positive)) if len(positive) else 0.0
                threshold = max(0.08, 6.0 * typical)
            else:
                threshold = 0.12

            if jump > threshold:
                segments.append(ordered.loc[current_indices].copy())
                current_indices = []

        current_indices.append(index)

    if current_indices:
        segments.append(ordered.loc[current_indices].copy())

    return [segment for segment in segments if len(segment) >= 2]


def add_top_decile(
    axis: plt.Axes,
    frame: pd.DataFrame,
    value_column: str,
    *,
    size: float = 60.0,
) -> float:
    threshold = float(frame[value_column].quantile(0.90))
    worst = frame.loc[frame[value_column] >= threshold]
    axis.scatter(
        worst["Mach"],
        worst["alpha_blumen"],
        s=size,
        marker="o",
        facecolors="none",
        edgecolors="cyan",
        linewidths=1.25,
        label="Top 10% errors",
        zorder=4,
    )
    return threshold


def safe_log_norm(values: pd.Series) -> LogNorm:
    values = pd.to_numeric(values, errors="coerce")
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0.0]
    vmin = max(float(positive.min()) if len(positive) else 1e-10, 1e-10)
    vmax = max(float(finite.max()) if len(finite) else vmin * 10.0, vmin * 10.0)
    return LogNorm(vmin=vmin, vmax=vmax)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise-csv", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    pointwise = load_pointwise(args.pointwise_csv.resolve())
    data = load_isoline_results(pointwise, args.result_root.resolve())

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    levels = sorted(float(v) for v in data["target_ci"].dropna().unique())
    cmap = plt.get_cmap("viridis")
    level_colors = {
        level: cmap(0.08 + 0.84 * i / max(1, len(levels) - 1))
        for i, level in enumerate(levels)
    }
    order_column = "source_row_id"

    # ------------------------------------------------------------------
    # Figure 1: digitized Blumen points + true classical isolines
    # ------------------------------------------------------------------
    fig1, ax = plt.subplots(figsize=(12.5, 8.0), constrained_layout=True)

    level_handles: list[Line2D] = []
    for level in levels:
        group = data.loc[np.isclose(data["target_ci"], level, atol=1e-12)].copy()
        color = level_colors[level]

        ax.scatter(
            group["Mach"],
            group["alpha_blumen"],
            s=30,
            marker="o",
            facecolors="none",
            edgecolors=color,
            linewidths=1.25,
            zorder=3,
        )

        longest_segment: pd.DataFrame | None = None
        for _, curve in group.groupby("curve_key", sort=False):
            for segment in split_converged_segments(curve, order_column):
                ax.plot(
                    segment["Mach"],
                    segment["alpha_classical"],
                    color=color,
                    linewidth=2.2,
                    zorder=2,
                )
                if longest_segment is None or len(segment) > len(longest_segment):
                    longest_segment = segment

        if longest_segment is not None and len(longest_segment) >= 3:
            position = min(len(longest_segment) - 2, max(1, int(0.58 * len(longest_segment))))
            row = longest_segment.iloc[position]
            ax.text(
                float(row["Mach"]),
                float(row["alpha_classical"]),
                rf"$c_i={level:g}$",
                color=color,
                fontsize=9,
                ha="left",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.60, "pad": 0.5},
                zorder=5,
            )

        level_handles.append(
            Line2D([0], [0], color=color, linewidth=2.2, label=rf"$c_i={level:g}$")
        )

    missing = data.loc[~data["isoline_converged"]]
    if not missing.empty:
        ax.scatter(
            missing["Mach"],
            missing["alpha_blumen"],
            marker="x",
            s=58,
            color="black",
            linewidths=1.5,
            zorder=5,
        )

    style_handles = [
        Line2D(
            [0], [0], linestyle="none", marker="o",
            markerfacecolor="none", markeredgecolor="black",
            markeredgewidth=1.25, markersize=6,
            label="Digitized Blumen points",
        ),
        Line2D(
            [0], [0], color="black", linewidth=2.2,
            label="Classical shooting isolines",
        ),
    ]
    if not missing.empty:
        style_handles.append(
            Line2D(
                [0], [0], linestyle="none", marker="x", color="black",
                markeredgewidth=1.5, markersize=7,
                label="No converged classical isoline root",
            )
        )

    legend_levels = ax.legend(
        handles=level_handles,
        title="Target level",
        loc="upper right",
        frameon=True,
        fontsize=9,
        title_fontsize=9,
        ncols=2,
    )
    ax.add_artist(legend_levels)
    ax.legend(handles=style_handles, loc="lower left", frameon=True, fontsize=9)

    ax.set_xlabel(r"Mach number $M$")
    ax.set_ylabel(r"Wavenumber $\alpha$")
    ax.set_title("Blumen 1970: supersonic reconstruction by shooting")
    ax.grid(True, alpha=0.25)

    x_min = float(data["Mach"].min())
    x_max = float(data["Mach"].max())
    y_all = pd.concat([data["alpha_blumen"], data["alpha_classical"]], ignore_index=True)
    y_min = max(0.0, float(y_all.min(skipna=True)))
    y_max = float(y_all.max(skipna=True))
    ax.set_xlim(x_min - 0.025 * (x_max - x_min), x_max + 0.025 * (x_max - x_min))
    ax.set_ylim(max(0.0, y_min - 0.04 * (y_max - y_min)), y_max + 0.04 * (y_max - y_min))

    reconstruction_base = output_root / "Fig_supersonic_blumen_reconstruction_by_shooting"
    fig1.savefig(reconstruction_base.with_suffix(".pdf"), bbox_inches="tight")
    fig1.savefig(reconstruction_base.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: pointwise ci error + geometric alpha distance
    # ------------------------------------------------------------------
    fig2, axes = plt.subplots(
        1, 2, figsize=(15.8, 6.5),
        sharex=True, sharey=True, constrained_layout=True,
    )
    ax_ci, ax_alpha = axes

    pointwise_ok = data.loc[data["pointwise_converged"]].copy()
    if pointwise_ok.empty:
        raise RuntimeError("No accepted pointwise classical roots are available.")

    scatter_ci = ax_ci.scatter(
        pointwise_ok["Mach"],
        pointwise_ok["alpha_blumen"],
        c=pointwise_ok["abs_delta_ci"],
        cmap="inferno",
        norm=safe_log_norm(pointwise_ok["abs_delta_ci"]),
        s=34,
        edgecolors="none",
        zorder=2,
    )
    q90_ci = add_top_decile(ax_ci, pointwise_ok, "abs_delta_ci")

    pointwise_missing = data.loc[~data["pointwise_converged"]]
    if not pointwise_missing.empty:
        ax_ci.scatter(
            pointwise_missing["Mach"],
            pointwise_missing["alpha_blumen"],
            marker="x",
            s=58,
            color="black",
            linewidths=1.5,
            label="No accepted pointwise root",
            zorder=4,
        )

    cbar_ci = fig2.colorbar(scatter_ci, ax=ax_ci, pad=0.018)
    cbar_ci.set_label(
        r"$|c_i^{\mathrm{classical}}(M_B,\alpha_B)-c_i^B|$"
    )
    ax_ci.set_title(r"Error in $c_i$ at Blumen points")
    ax_ci.set_xlabel(r"Mach number $M$")
    ax_ci.set_ylabel(r"Wavenumber $\alpha$")
    ax_ci.grid(True, alpha=0.22)
    ax_ci.legend(loc="best", frameon=True, fontsize=9)

    isoline_ok = data.loc[data["isoline_converged"]].copy()
    if isoline_ok.empty:
        raise RuntimeError("No converged true-isoline roots are available.")

    scatter_alpha = ax_alpha.scatter(
        isoline_ok["Mach"],
        isoline_ok["alpha_blumen"],
        c=isoline_ok["abs_delta_alpha"],
        cmap="inferno",
        norm=safe_log_norm(isoline_ok["abs_delta_alpha"]),
        s=34,
        edgecolors="none",
        zorder=2,
    )
    q90_alpha = add_top_decile(ax_alpha, isoline_ok, "abs_delta_alpha")

    isoline_missing = data.loc[~data["isoline_converged"]]
    if not isoline_missing.empty:
        ax_alpha.scatter(
            isoline_missing["Mach"],
            isoline_missing["alpha_blumen"],
            marker="x",
            s=58,
            color="black",
            linewidths=1.5,
            label="No converged isoline root",
            zorder=4,
        )

    cbar_alpha = fig2.colorbar(scatter_alpha, ax=ax_alpha, pad=0.018)
    cbar_alpha.set_label(
        r"$|\alpha_{\mathrm{classical}}-\alpha_B|$"
    )
    ax_alpha.set_title("Point-to-solver-isoline distance")
    ax_alpha.set_xlabel(r"Mach number $M$")
    ax_alpha.grid(True, alpha=0.22)
    ax_alpha.legend(loc="best", frameon=True, fontsize=9)

    for axis in axes:
        axis.set_xlim(x_min - 0.025 * (x_max - x_min), x_max + 0.025 * (x_max - x_min))
        axis.set_ylim(max(0.0, y_min - 0.04 * (y_max - y_min)), y_max + 0.04 * (y_max - y_min))

    fig2.suptitle("Supersonic shooting error map", fontsize=16)

    error_base = output_root / "Fig_supersonic_shooting_error_map"
    fig2.savefig(error_base.with_suffix(".pdf"), bbox_inches="tight")
    fig2.savefig(error_base.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Source data and summary tables
    # ------------------------------------------------------------------
    data.to_csv(
        output_root / "supersonic_blumen_validation_points.csv",
        index=False,
    )

    missing_table = data.loc[
        (~data["pointwise_converged"]) | (~data["isoline_converged"])
    ].copy()
    missing_table.to_csv(
        output_root / "Tab_supersonic_blumen_missing_roots.csv",
        index=False,
    )

    summary_rows: list[dict] = []
    for level, group in data.groupby("target_ci", sort=True):
        point_ok = group.loc[group["pointwise_converged"]]
        iso_ok = group.loc[group["isoline_converged"]]
        summary_rows.append(
            {
                "blumen_ci": float(level),
                "n_digitized_points": int(len(group)),
                "n_accepted_pointwise_roots": int(group["pointwise_converged"].sum()),
                "pointwise_completion": float(group["pointwise_converged"].mean()),
                "mean_signed_error_ci": float(point_ok["delta_ci"].mean()) if len(point_ok) else np.nan,
                "mae_ci": float(point_ok["abs_delta_ci"].mean()) if len(point_ok) else np.nan,
                "rmse_ci": float(np.sqrt(np.mean(point_ok["delta_ci"] ** 2))) if len(point_ok) else np.nan,
                "max_abs_error_ci": float(point_ok["abs_delta_ci"].max()) if len(point_ok) else np.nan,
                "n_converged_isoline_roots": int(group["isoline_converged"].sum()),
                "isoline_completion": float(group["isoline_converged"].mean()),
                "mean_signed_delta_alpha": float(iso_ok["delta_alpha"].mean()) if len(iso_ok) else np.nan,
                "mean_distance_alpha": float(iso_ok["abs_delta_alpha"].mean()) if len(iso_ok) else np.nan,
                "rmse_distance_alpha": float(np.sqrt(np.mean(iso_ok["delta_alpha"] ** 2))) if len(iso_ok) else np.nan,
                "max_distance_alpha": float(iso_ok["abs_delta_alpha"].max()) if len(iso_ok) else np.nan,
                "median_pointwise_residual": float(point_ok["residual_norm"].median()) if len(point_ok) else np.nan,
                "max_pointwise_residual": float(point_ok["residual_norm"].max()) if len(point_ok) else np.nan,
                "median_isoline_residual": float(iso_ok["isoline_residual_norm"].median()) if len(iso_ok) else np.nan,
                "max_isoline_residual": float(iso_ok["isoline_residual_norm"].max()) if len(iso_ok) else np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_root / "Tab_supersonic_blumen_reconstruction_summary.csv",
        index=False,
    )

    metadata = {
        "n_positive_targets": int(len(data)),
        "n_pointwise_converged": int(data["pointwise_converged"].sum()),
        "n_isoline_converged": int(data["isoline_converged"].sum()),
        "pointwise_top_10_percent_threshold": q90_ci,
        "isoline_top_10_percent_threshold": q90_alpha,
    }
    (output_root / "supersonic_blumen_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== SUPERSONIC BLUMEN VALIDATION FIGURES ===")
    print(f"Positive targets           : {metadata['n_positive_targets']}")
    print(f"Pointwise roots accepted   : {metadata['n_pointwise_converged']}")
    print(f"True isoline roots accepted: {metadata['n_isoline_converged']}")
    print()
    print(summary.to_string(index=False))
    print()
    print("Reconstruction PDF :", reconstruction_base.with_suffix(".pdf"))
    print("Reconstruction PNG :", reconstruction_base.with_suffix(".png"))
    print("Error-map PDF      :", error_base.with_suffix(".pdf"))
    print("Error-map PNG      :", error_base.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
