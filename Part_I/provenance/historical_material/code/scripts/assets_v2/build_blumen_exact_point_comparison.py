#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.assets_v2.build_exact4_article_modes import (
    choose_checkpoint,
    checkpoint_dataframe,
    load_classic_full_mode,
)
from scripts.dev.train_subsonic_seedGEP_pq2d_continuous_M_alpha_etaaware import (
    CiGridIDW,
)

ATLAS_MACH_MIN = 0.02
ATLAS_MACH_MAX = 0.98
ATLAS_ETA_MIN = 0.02
ATLAS_ETA_MAX = 0.98


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the classical reference at every digitized Blumen point "
            "and the PINN only where the curated subsonic atlas is defined."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compute = subparsers.add_parser("compute")
    compute.add_argument("--blumen-csv", required=True)
    compute.add_argument("--index", required=True, type=int)
    compute.add_argument("--output-dir", required=True)

    merge = subparsers.add_parser("merge-plot")
    merge.add_argument("--blumen-csv", required=True)
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--figure-dir", required=True)
    merge.add_argument("--manifest-out", required=True)

    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def build_ci_provider(item: dict[str, Any]) -> CiGridIDW:
    checkpoint = item["checkpoint"]
    args = item["args"]
    provider = CiGridIDW(
        anchor_df=checkpoint_dataframe(checkpoint["anchor_df"]),
        eta_scale=float(args.get("ci_idw_eta_scale", 0.25)),
        mach_scale=float(args.get("ci_idw_mach_scale", 0.25)),
        power=float(args.get("ci_idw_power", 4.0)),
    ).double()
    provider.eval()
    return provider


def inside_atlas_rectangle(mach: float, eta: float) -> bool:
    return (
        ATLAS_MACH_MIN <= mach <= ATLAS_MACH_MAX
        and ATLAS_ETA_MIN <= eta <= ATLAS_ETA_MAX
    )


def compute_one(blumen_csv: Path, index: int, output_dir: Path) -> None:
    frame = pd.read_csv(blumen_csv)
    required = {
        "curve_id",
        "ci",
        "alpha",
        "Mach",
        "source_file",
        "point_index",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"{blumen_csv}: missing columns {sorted(missing)}"
        )
    if index < 0 or index >= len(frame):
        raise IndexError(f"index={index}; expected 0..{len(frame)-1}")

    row = frame.iloc[index]
    mach = float(row["Mach"])
    alpha = float(row["alpha"])
    ci_blumen = float(row["ci"])

    if not (0.0 < mach < 1.0):
        raise RuntimeError(f"Invalid subsonic Mach at row {index}: {mach}")
    if alpha <= 0.0:
        raise RuntimeError(f"Invalid alpha at row {index}: {alpha}")

    eta = alpha / math.sqrt(1.0 - mach * mach)
    in_rectangle = inside_atlas_rectangle(mach, eta)

    ci_classic: float | None = None
    classic_status = "not_run"
    classic_error = ""
    try:
        _, classic_value = load_classic_full_mode(alpha, mach)
        ci_classic = finite_float(classic_value)
        if ci_classic is None:
            classic_status = "non_finite"
        else:
            classic_status = "ok"
    except Exception as exc:
        classic_status = "failed"
        classic_error = f"{type(exc).__name__}: {exc}"

    ci_pinn: float | None = None
    chart_id = ""
    pinn_status = "outside_atlas_rectangle"
    pinn_error = ""

    if in_rectangle:
        try:
            item = choose_checkpoint(mach, eta)
            chart_id = str(item["chart_id"])
            provider = build_ci_provider(item)
            at = torch.tensor([[alpha]], dtype=torch.float64)
            mt = torch.tensor([[mach]], dtype=torch.float64)
            with torch.no_grad():
                ci_pinn = finite_float(provider(at, mt).item())
            if ci_pinn is None:
                pinn_status = "non_finite"
            else:
                pinn_status = "ok"
        except Exception as exc:
            pinn_status = "no_curated_chart_or_eval_failed"
            pinn_error = f"{type(exc).__name__}: {exc}"

    payload = {
        "row_index": int(index),
        "curve_id": float(row["curve_id"]),
        "ci_blumen": ci_blumen,
        "alpha": alpha,
        "Mach": mach,
        "eta": eta,
        "source_file": str(row["source_file"]),
        "point_index": int(row["point_index"]),
        "alpha_digitized_raw": (
            finite_float(row["alpha_digitized_raw"])
            if "alpha_digitized_raw" in frame.columns
            and pd.notna(row["alpha_digitized_raw"])
            else None
        ),
        "inside_atlas_rectangle": bool(in_rectangle),
        "chart_id": chart_id,
        "classic_status": classic_status,
        "classic_error": classic_error,
        "pinn_status": pinn_status,
        "pinn_error": pinn_error,
        "ci_classic": ci_classic,
        "ci_pinn": ci_pinn,
        "classic_minus_blumen": (
            ci_classic - ci_blumen
            if ci_classic is not None
            else None
        ),
        "pinn_minus_blumen": (
            ci_pinn - ci_blumen
            if ci_pinn is not None
            else None
        ),
        "pinn_minus_classic": (
            ci_pinn - ci_classic
            if ci_pinn is not None and ci_classic is not None
            else None
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"point_{index:04d}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "path": str(path),
                "classic_status": classic_status,
                "pinn_status": pinn_status,
                "inside_atlas_rectangle": in_rectangle,
            },
            sort_keys=True,
        )
    )


def common_ci_limits(frame: pd.DataFrame) -> tuple[float, float]:
    arrays = [
        pd.to_numeric(frame["ci_blumen"], errors="coerce").to_numpy(float),
        pd.to_numeric(frame["ci_classic"], errors="coerce").to_numpy(float),
        pd.to_numeric(frame["ci_pinn"], errors="coerce").to_numpy(float),
    ]
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise RuntimeError("No finite ci values available for plotting")
    lower = float(np.min(values))
    upper = float(np.max(values))
    padding = max(0.03 * (upper - lower), 1.0e-4)
    return lower - padding, upper + padding


def add_point_map(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value_column: str,
    *,
    title: str,
    edge_color: str,
    norm: Normalize,
    unsupported: pd.DataFrame | None = None,
):
    values = pd.to_numeric(frame[value_column], errors="coerce")
    valid = frame.loc[np.isfinite(values)].copy()
    valid[value_column] = values.loc[valid.index]

    scatter = ax.scatter(
        valid["Mach"],
        valid["alpha"],
        c=valid[value_column],
        cmap="viridis",
        norm=norm,
        s=25,
        edgecolors=edge_color,
        linewidths=0.45,
    )

    if unsupported is not None and not unsupported.empty:
        ax.scatter(
            unsupported["Mach"],
            unsupported["alpha"],
            color="lightgray",
            marker="x",
            s=17,
            linewidths=0.7,
            label="outside PINN atlas",
        )
        ax.legend(fontsize=8, loc="best")

    ax.set(
        xlabel=r"Mach number $M$",
        ylabel=r"Wavenumber $\alpha$",
        title=title,
    )
    ax.grid(True, alpha=0.2)
    return scatter


def merge_and_plot(
    blumen_csv: Path,
    output_dir: Path,
    figure_dir: Path,
    manifest_out: Path,
) -> None:
    source = pd.read_csv(blumen_csv)
    shards = sorted(output_dir.glob("point_*.json"))
    if len(shards) != len(source):
        found = {int(path.stem.split("_")[-1]) for path in shards}
        missing = [
            index
            for index in range(len(source))
            if index not in found
        ]
        raise RuntimeError(
            f"Expected {len(source)} shards, found {len(shards)}. "
            f"Missing indices: {missing[:30]}"
        )

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in shards
    ]
    frame = pd.DataFrame(records).sort_values(
        ["ci_blumen", "point_index", "Mach"]
    )

    numeric_columns = (
        "ci_blumen",
        "ci_classic",
        "ci_pinn",
        "classic_minus_blumen",
        "pinn_minus_blumen",
        "pinn_minus_classic",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_out, index=False)

    ci_min, ci_max = common_ci_limits(frame)
    norm = Normalize(vmin=ci_min, vmax=ci_max)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 10.5),
        constrained_layout=True,
    )

    blumen_scatter = add_point_map(
        axes[0, 0],
        frame,
        "ci_blumen",
        title="Blumen digitized values at every point",
        edge_color="darkorange",
        norm=norm,
    )

    classic_scatter = add_point_map(
        axes[0, 1],
        frame.loc[frame["classic_status"] == "ok"],
        "ci_classic",
        title="Classical solver at the same (M, alpha) points",
        edge_color="black",
        norm=norm,
    )

    pinn_ok = frame.loc[frame["pinn_status"] == "ok"]
    pinn_unsupported = frame.loc[frame["pinn_status"] != "ok"]
    pinn_scatter = add_point_map(
        axes[1, 0],
        pinn_ok,
        "ci_pinn",
        title="PINN at the same points inside its validated atlas",
        edge_color="green",
        norm=norm,
        unsupported=pinn_unsupported,
    )

    colorbar = figure.colorbar(
        blumen_scatter,
        ax=[axes[0, 0], axes[0, 1], axes[1, 0]],
        label=r"$c_i$",
        shrink=0.92,
    )
    colorbar.ax.tick_params(labelsize=9)

    parity_ax = axes[1, 1]
    parity_lower = ci_min
    parity_upper = ci_max
    parity_ax.plot(
        [parity_lower, parity_upper],
        [parity_lower, parity_upper],
        color="orange",
        linewidth=1.5,
        label=r"Blumen reference $y=x$",
    )

    classic_valid = frame.loc[
        (frame["classic_status"] == "ok")
        & np.isfinite(frame["ci_classic"])
    ]
    parity_ax.scatter(
        classic_valid["ci_blumen"],
        classic_valid["ci_classic"],
        color="black",
        marker="x",
        s=24,
        label="Classical solver",
    )

    pinn_valid = frame.loc[
        (frame["pinn_status"] == "ok")
        & np.isfinite(frame["ci_pinn"])
    ]
    parity_ax.scatter(
        pinn_valid["ci_blumen"],
        pinn_valid["ci_pinn"],
        facecolors="none",
        edgecolors="green",
        marker="o",
        s=24,
        linewidths=0.8,
        label="PINN (inside atlas)",
    )
    parity_ax.set(
        xlabel=r"Blumen $c_i$",
        ylabel=r"Computed $c_i$",
        title="Pointwise parity",
        xlim=(parity_lower, parity_upper),
        ylim=(parity_lower, parity_upper),
    )
    parity_ax.grid(True, alpha=0.2)
    parity_ax.legend(fontsize=8, loc="best")

    classic_abs = np.abs(
        classic_valid["classic_minus_blumen"].to_numpy(float)
    )
    pinn_abs = np.abs(
        pinn_valid["pinn_minus_blumen"].to_numpy(float)
    )

    summary_lines = [
        f"all Blumen points: {len(frame)}",
        f"classical successful: {len(classic_valid)}",
        f"PINN inside validated atlas: {len(pinn_valid)}",
        f"PINN unavailable/outside atlas: {len(frame) - len(pinn_valid)}",
    ]
    if len(classic_abs):
        summary_lines.append(
            f"classical median |delta ci|: {np.median(classic_abs):.3e}"
        )
    if len(pinn_abs):
        summary_lines.append(
            f"PINN median |delta ci|: {np.median(pinn_abs):.3e}"
        )

    parity_ax.text(
        0.03,
        0.97,
        "\n".join(summary_lines),
        transform=parity_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    figure.suptitle(
        (
            "Blumen points evaluated at identical (M, alpha): "
            "classical reference and validated PINN domain"
        ),
        fontsize=14,
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "Fig_ci_Blumen_exact_points_classical_PINN"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    status_summary = {
        "n_total": int(len(frame)),
        "classic_status": {
            str(key): int(value)
            for key, value in frame["classic_status"].value_counts().items()
        },
        "pinn_status": {
            str(key): int(value)
            for key, value in frame["pinn_status"].value_counts().items()
        },
        "inside_atlas_rectangle": int(
            frame["inside_atlas_rectangle"].astype(bool).sum()
        ),
    }
    summary_path = manifest_out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(status_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(manifest_out)
    print(summary_path)
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".png"))
    print(json.dumps(status_summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    torch.set_num_threads(
        max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    )

    if args.command == "compute":
        compute_one(
            Path(args.blumen_csv),
            args.index,
            Path(args.output_dir),
        )
    elif args.command == "merge-plot":
        merge_and_plot(
            Path(args.blumen_csv),
            Path(args.output_dir),
            Path(args.figure_dir),
            Path(args.manifest_out),
        )
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
