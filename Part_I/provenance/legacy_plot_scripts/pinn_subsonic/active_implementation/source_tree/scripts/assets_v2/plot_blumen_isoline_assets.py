#!/usr/bin/env python3
"""Build Blumen-vs-PINN/GEP assets from digitized constant-ci isolines.

Expected raw format
-------------------
Input directory contains files named, for example:
    0.00.csv
    0.05.csv
    0.10.csv

The filename stem is the constant c_i level. Each file contains one
digitized isoline with rows formatted as:
    alpha; Mach
using a decimal comma, for example:
    0,003864552420605044; 0,9999704714237203

Outputs
-------
data/blumen_ci_datasets.csv
data/Blumen_interpolated_comparison.csv
figures/Fig04a_Blumen_PINN_GEP_overlay.{pdf,png}
figures/Fig04b_Blumen_error_maps.{pdf,png}
figures/Fig04c_Blumen_Mach_cuts.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.interpolate import LinearNDInterpolator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blumen-dir",
        default="KH_RT_Blumen/subsonic",
        help="Directory containing one CSV per digitized c_i isoline.",
    )
    parser.add_argument(
        "--release-csv",
        default=(
            "assets/pinn_subsonic/local_atlas_v1/offgrid_validation_384/"
            "offgrid_validation_results_384_release.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "assets/pinn_subsonic/local_atlas_v1/"
            "publication_assets_scientific_v2"
        ),
    )
    parser.add_argument(
        "--negative-alpha-tolerance",
        type=float,
        default=2.0e-3,
        help=(
            "Digitized alpha values in [-tol,0) are snapped to zero. "
            "The raw value is preserved."
        ),
    )
    parser.add_argument(
        "--relative-error-floor",
        type=float,
        default=1.0e-3,
        help="Denominator floor for relative errors near the neutral curve.",
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def numeric_stem(path: Path) -> float | None:
    try:
        return float(path.stem)
    except ValueError:
        return None


def parse_decimal_comma(value: str) -> float:
    text = value.strip().replace(" ", "").replace(",", ".")
    return float(text)


def read_isolines(directory: Path, negative_alpha_tolerance: float) -> pd.DataFrame:
    if not directory.is_dir():
        raise FileNotFoundError(f"Blumen directory not found: {directory}")

    files = sorted(
        (p for p in directory.glob("*.csv") if numeric_stem(p) is not None),
        key=lambda p: float(p.stem),
    )
    if not files:
        raise FileNotFoundError(
            f"No numerically named isoline CSV files found in {directory}"
        )

    rows: list[dict[str, object]] = []
    for path in files:
        ci_level = float(path.stem)
        for point_index, raw_line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{point_index}: expected 'alpha; Mach', got {line!r}"
                )
            alpha_raw = parse_decimal_comma(parts[0])
            mach = parse_decimal_comma(parts[1])
            alpha = alpha_raw
            if -negative_alpha_tolerance <= alpha_raw < 0.0:
                alpha = 0.0

            rows.append(
                {
                    "curve_id": path.stem,
                    "ci": ci_level,
                    "alpha": alpha,
                    "Mach": mach,
                    "source_file": path.name,
                    "point_index": point_index,
                    "alpha_digitized_raw": alpha_raw,
                }
            )

    frame = pd.DataFrame(rows)
    numeric = ["ci", "alpha", "Mach", "alpha_digitized_raw"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["ci", "alpha", "Mach"]).reset_index(drop=True)

    if frame.empty:
        raise ValueError("No valid digitized Blumen points were parsed.")

    return frame


def first_existing_column(frame: pd.DataFrame, names: Iterable[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise KeyError(f"None of the required columns exists: {list(names)}")


def canonicalize_release(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Release CSV not found: {path}")

    raw = pd.read_csv(path)

    mach_col = first_existing_column(raw, ["Mach", "M", "mach"])
    eta_col = first_existing_column(raw, ["eta", "Eta"])
    alpha_col = first_existing_column(raw, ["alpha", "Alpha"])
    seed_col = first_existing_column(
        raw, ["ci_seed", "ci_seed_target", "pinn_ci", "ci_pinn"]
    )
    final_col = first_existing_column(
        raw, ["ci_final", "gep_ci", "ci_forward"]
    )

    out = pd.DataFrame(
        {
            "point_id": (
                raw["point_id"].astype(str)
                if "point_id" in raw.columns
                else [f"P{i:04d}" for i in range(len(raw))]
            ),
            "Mach": pd.to_numeric(raw[mach_col], errors="coerce"),
            "eta": pd.to_numeric(raw[eta_col], errors="coerce"),
            "alpha": pd.to_numeric(raw[alpha_col], errors="coerce"),
            "ci_seed": pd.to_numeric(raw[seed_col], errors="coerce"),
            "ci_final": pd.to_numeric(raw[final_col], errors="coerce"),
        }
    )

    out = out.dropna(
        subset=["Mach", "eta", "alpha", "ci_seed", "ci_final"]
    ).reset_index(drop=True)
    if len(out) < 10:
        raise ValueError(f"Too few valid release points: {len(out)}")
    return out


def deduplicate_interpolation_points(
    frame: pd.DataFrame, value_column: str
) -> pd.DataFrame:
    work = frame[["Mach", "alpha", value_column]].dropna().copy()
    return (
        work.groupby(["Mach", "alpha"], as_index=False)[value_column]
        .mean()
        .sort_values(["Mach", "alpha"])
        .reset_index(drop=True)
    )


def make_interpolator(frame: pd.DataFrame, value_column: str) -> LinearNDInterpolator:
    work = deduplicate_interpolation_points(frame, value_column)
    if len(work) < 3:
        raise ValueError(f"Too few points for {value_column} interpolation.")
    return LinearNDInterpolator(
        work[["Mach", "alpha"]].to_numpy(dtype=float),
        work[value_column].to_numpy(dtype=float),
        fill_value=np.nan,
    )


def finite_triangulation(frame: pd.DataFrame) -> tuple[mtri.Triangulation, np.ndarray]:
    work = frame[["alpha", "Mach", "ci_seed"]].dropna().copy()
    work = work.groupby(["alpha", "Mach"], as_index=False).mean(numeric_only=True)
    tri = mtri.Triangulation(
        work["alpha"].to_numpy(),
        work["Mach"].to_numpy(),
    )
    return tri, work.index.to_numpy()


def save_figure(fig: plt.Figure, base: Path, dpi: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(
    blumen: pd.DataFrame,
    release: pd.DataFrame,
    output_base: Path,
    dpi: int,
) -> None:
    levels = sorted(blumen["ci"].unique())
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(levels), max(levels) if max(levels) > min(levels) else min(levels) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharex=True, sharey=True)

    # Panel A: original digitized curves, preserving digitization order.
    ax = axes[0]
    for ci_level, group in blumen.groupby("ci", sort=True):
        group = group.sort_values(["source_file", "point_index"])
        ax.plot(
            group["alpha"],
            group["Mach"],
            color=cmap(norm(ci_level)),
            lw=1.8,
            label=fr"$c_i={ci_level:g}$",
        )
    ax.set_title("(a) Blumen digitized isolines")
    ax.legend(fontsize=8, loc="best")

    # Panels B/C: contours reconstructed from release point clouds.
    unique_release = (
        release.groupby(["Mach", "alpha"], as_index=False)
        .agg(ci_seed=("ci_seed", "mean"), ci_final=("ci_final", "mean"))
    )
    triangulation = mtri.Triangulation(
        unique_release["alpha"].to_numpy(),
        unique_release["Mach"].to_numpy(),
    )

    for ax, column, title in [
        (axes[1], "ci_seed", "(b) Direct PINN seed"),
        (axes[2], "ci_final", "(c) PINN-seeded GEP"),
    ]:
        values = unique_release[column].to_numpy()
        contour = ax.tricontour(
            triangulation,
            values,
            levels=levels,
            cmap=cmap,
            norm=norm,
            linewidths=1.8,
        )
        ax.clabel(contour, inline=True, fontsize=7, fmt="%g")

        # Overlay digitized Blumen curves as thin black references.
        for _, group in blumen.groupby("ci", sort=True):
            group = group.sort_values(["source_file", "point_index"])
            ax.plot(group["alpha"], group["Mach"], color="black", lw=0.7, alpha=0.45)
        ax.set_title(title)

    for ax in axes:
        ax.set_xlabel(r"Physical wavenumber $\alpha$")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel(r"Mach number $M$")
    fig.suptitle(
        r"Blumen, direct PINN and GEP-refined growth-rate isolines",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, output_base, dpi)


def scatter_error_map(
    ax: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    title: str,
    norm: LogNorm,
) -> None:
    sc = ax.scatter(
        frame["alpha"],
        frame["Mach"],
        c=frame[column],
        s=28,
        cmap="magma",
        norm=norm,
        edgecolors="none",
    )
    ax.set(
        xlabel=r"Physical wavenumber $\alpha$",
        ylabel=r"Mach number $M$",
        title=title,
    )
    ax.grid(alpha=0.12)
    plt.colorbar(sc, ax=ax, label=column)


def positive_lognorm(values: pd.Series, floor: float = 1.0e-10) -> LogNorm:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    positive = finite[finite > 0]
    if positive.empty:
        return LogNorm(vmin=floor, vmax=floor * 10)
    vmin = max(float(positive.min()), floor)
    vmax = max(float(positive.max()), vmin * 1.01)
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_errors(
    comparison: pd.DataFrame,
    output_base: Path,
    dpi: int,
) -> None:
    valid = comparison.dropna(subset=["ci_blumen"]).copy()
    if len(valid) < 10:
        raise ValueError(
            f"Only {len(valid)} release points lie inside the Blumen interpolation support."
        )

    abs_norm = positive_lognorm(
        pd.concat([valid["pinn_blumen_abs"], valid["gep_blumen_abs"]])
    )
    rel_norm = positive_lognorm(
        pd.concat([valid["pinn_blumen_rel"], valid["gep_blumen_rel"]]),
        floor=1.0e-8,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), sharex=True, sharey=True)
    scatter_error_map(
        axes[0, 0], valid, "pinn_blumen_abs",
        "(a) Direct PINN vs Blumen: absolute error", abs_norm
    )
    scatter_error_map(
        axes[0, 1], valid, "gep_blumen_abs",
        "(b) GEP final vs Blumen: absolute error", abs_norm
    )
    scatter_error_map(
        axes[1, 0], valid, "pinn_blumen_rel",
        "(c) Direct PINN vs Blumen: stabilized relative error", rel_norm
    )
    scatter_error_map(
        axes[1, 1], valid, "gep_blumen_rel",
        "(d) GEP final vs Blumen: stabilized relative error", rel_norm
    )
    fig.tight_layout()
    save_figure(fig, output_base, dpi)


def alpha_limit_for_mach(mach: float) -> float:
    return 0.98 * math.sqrt(max(0.0, 1.0 - mach * mach))


def plot_mach_cuts(
    blumen_interpolator: LinearNDInterpolator,
    seed_interpolator: LinearNDInterpolator,
    final_interpolator: LinearNDInterpolator,
    blumen: pd.DataFrame,
    output_base: Path,
    dpi: int,
) -> None:
    targets = [0.10, 0.30, 0.50, 0.70, 0.90, 0.97]
    alpha_max_digitized = float(blumen["alpha"].max())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), squeeze=False)
    for ax, mach in zip(axes.ravel(), targets):
        alpha_max = min(alpha_max_digitized, alpha_limit_for_mach(mach))
        alpha_grid = np.linspace(0.0, max(alpha_max, 1.0e-6), 500)
        mach_grid = np.full_like(alpha_grid, mach)

        ci_blumen = np.asarray(
            blumen_interpolator(mach_grid, alpha_grid), dtype=float
        )
        ci_seed = np.asarray(
            seed_interpolator(mach_grid, alpha_grid), dtype=float
        )
        ci_final = np.asarray(
            final_interpolator(mach_grid, alpha_grid), dtype=float
        )

        if np.isfinite(ci_blumen).any():
            ax.plot(alpha_grid, ci_blumen, color="black", lw=2.2, label="Blumen")
        if np.isfinite(ci_seed).any():
            ax.plot(alpha_grid, ci_seed, ls="--", lw=1.8, label="Direct PINN")
        if np.isfinite(ci_final).any():
            ax.plot(alpha_grid, ci_final, ls="-.", lw=1.8, label="PINN + GEP")

        ax.set(
            title=fr"$M={mach:.2f}$",
            xlabel=r"$\alpha$",
            ylabel=r"$c_i$",
        )
        ax.grid(alpha=0.2)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(r"Growth-rate cuts at fixed Mach number", y=1.01)
    fig.tight_layout()
    save_figure(fig, output_base, dpi)


def main() -> None:
    args = parse_args()

    blumen_dir = Path(args.blumen_dir).resolve()
    release_csv = Path(args.release_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    data_dir = output_dir / "data"
    figure_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    blumen = read_isolines(
        blumen_dir,
        negative_alpha_tolerance=args.negative_alpha_tolerance,
    )
    release = canonicalize_release(release_csv)

    normalized_path = data_dir / "blumen_ci_datasets.csv"
    blumen.to_csv(normalized_path, index=False)

    blumen_interp = make_interpolator(blumen, "ci")
    seed_interp = make_interpolator(release, "ci_seed")
    final_interp = make_interpolator(release, "ci_final")

    query = release[["Mach", "alpha"]].to_numpy(dtype=float)
    comparison = release.copy()
    comparison["ci_blumen"] = np.asarray(blumen_interp(query), dtype=float)
    comparison["pinn_blumen_abs"] = (
        comparison["ci_seed"] - comparison["ci_blumen"]
    ).abs()
    comparison["gep_blumen_abs"] = (
        comparison["ci_final"] - comparison["ci_blumen"]
    ).abs()

    denominator = comparison["ci_blumen"].abs().clip(
        lower=args.relative_error_floor
    )
    comparison["pinn_blumen_rel"] = (
        comparison["pinn_blumen_abs"] / denominator
    )
    comparison["gep_blumen_rel"] = (
        comparison["gep_blumen_abs"] / denominator
    )
    comparison["inside_blumen_support"] = comparison["ci_blumen"].notna()

    comparison_path = data_dir / "Blumen_interpolated_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    plot_overlay(
        blumen,
        release,
        figure_dir / "Fig04a_Blumen_PINN_GEP_overlay",
        args.dpi,
    )
    plot_errors(
        comparison,
        figure_dir / "Fig04b_Blumen_error_maps",
        args.dpi,
    )
    plot_mach_cuts(
        blumen_interp,
        seed_interp,
        final_interp,
        blumen,
        figure_dir / "Fig04c_Blumen_Mach_cuts",
        args.dpi,
    )

    valid = comparison["inside_blumen_support"]
    summary = {
        "raw_isoline_directory": str(blumen_dir),
        "release_csv": str(release_csv),
        "n_isolines": int(blumen["ci"].nunique()),
        "ci_levels": [float(v) for v in sorted(blumen["ci"].unique())],
        "n_digitized_points": int(len(blumen)),
        "n_release_points": int(len(release)),
        "n_release_points_inside_blumen_support": int(valid.sum()),
        "alpha_range_digitized": [
            float(blumen["alpha"].min()),
            float(blumen["alpha"].max()),
        ],
        "Mach_range_digitized": [
            float(blumen["Mach"].min()),
            float(blumen["Mach"].max()),
        ],
        "negative_alpha_values_snapped_to_zero": int(
            (
                (blumen["alpha_digitized_raw"] < 0)
                & (blumen["alpha"] == 0)
            ).sum()
        ),
        "relative_error_floor": float(args.relative_error_floor),
    }
    (output_dir / "blumen_build_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("===== BLUMEN ASSETS BUILT =====")
    print(json.dumps(summary, indent=2))
    print("Normalized:", normalized_path)
    print("Comparison:", comparison_path)
    print("Figures:", figure_dir)


if __name__ == "__main__":
    main()
