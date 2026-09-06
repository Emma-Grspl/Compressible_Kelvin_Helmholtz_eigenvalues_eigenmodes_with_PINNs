#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.tri import LinearTriInterpolator, Triangulation
import numpy as np
import pandas as pd


def resolve_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    label: str,
) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise RuntimeError(
        f"Missing column for {label}. Available columns: "
        f"{list(frame.columns)}"
    )


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 320) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def finite_log_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate(
        [np.asarray(array, dtype=float).ravel() for array in arrays]
    )
    positive = values[np.isfinite(values) & (values > 0.0)]

    if positive.size == 0:
        return 1.0e-14, 1.0e-13

    vmin = max(float(np.min(positive)), 1.0e-14)
    vmax = float(np.max(positive))

    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = 10.0 * vmin

    return vmin, vmax


def safe_log_values(
    values: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    output[valid] = np.clip(values[valid], vmin, vmax)
    return output


def exact_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {
        "alpha": resolve_column(frame, ["alpha", "Alpha"], "alpha"),
        "Mach": resolve_column(frame, ["Mach", "M"], "Mach"),
        "blumen": resolve_column(
            frame,
            [
                "ci_blumen",
                "ci_digitized",
                "ci_exact",
                "ci_target",
                "ci_blumen_digitized",
            ],
            "digitized Blumen ci",
        ),
        "classic": resolve_column(
            frame,
            [
                "ci_classic",
                "ci_ref",
                "ci_shooting",
                "ci_reference",
            ],
            "classical ci",
        ),
        "gep": resolve_column(
            frame,
            [
                "ci_gep",
                "ci_final",
                "pinn_matched_ci",
                "ci_pinn_gep",
            ],
            "PINN+GEP ci",
        ),
    }


def build_exact_error_figure(asset_root: Path) -> None:
    path = asset_root / "data" / "Blumen_exact_point_comparison.csv"
    frame = pd.read_csv(path)
    columns = exact_columns(frame)

    alpha = pd.to_numeric(frame[columns["alpha"]], errors="coerce").to_numpy()
    mach = pd.to_numeric(frame[columns["Mach"]], errors="coerce").to_numpy()
    blumen = pd.to_numeric(frame[columns["blumen"]], errors="coerce").to_numpy()
    classic = pd.to_numeric(frame[columns["classic"]], errors="coerce").to_numpy()
    gep = pd.to_numeric(frame[columns["gep"]], errors="coerce").to_numpy()

    error_classic = np.abs(gep - classic)
    error_blumen = np.abs(gep - blumen)
    vmin, vmax = finite_log_limits(error_classic, error_blumen)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("magma")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.8, 6.2),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.875,
        bottom=0.13,
        top=0.90,
        wspace=0.12,
    )

    panels = [
        (
            axes[0],
            error_classic,
            "PINN + GEP versus classical shooting",
        ),
        (
            axes[1],
            error_blumen,
            "PINN + GEP versus digitized Blumen",
        ),
    ]

    for axis, values, title in panels:
        axis.scatter(
            alpha,
            mach,
            c=safe_log_values(values, vmin, vmax),
            s=38,
            marker="o",
            cmap=cmap,
            norm=norm,
            edgecolors="none",
        )
        axis.set_title(title, fontsize=16)
        axis.set_xlabel(r"Wavenumber $\alpha$", fontsize=13)
        axis.grid(alpha=0.22)

    axes[0].set_ylabel(r"Mach number $M$", fontsize=13)

    # Dedicated colorbar axis outside both data panels.
    colorbar_axis = fig.add_axes([0.90, 0.16, 0.018, 0.68])
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, cax=colorbar_axis)
    colorbar.set_label(r"Absolute $c_i$ error", fontsize=13)

    save_figure(
        fig,
        asset_root
        / "figures"
        / "Fig_Blumen_exact_points_absolute_errors",
    )


def build_exact_value_figure(asset_root: Path) -> None:
    path = asset_root / "data" / "Blumen_exact_point_comparison.csv"
    frame = pd.read_csv(path)
    columns = exact_columns(frame)

    alpha = pd.to_numeric(frame[columns["alpha"]], errors="coerce").to_numpy()
    mach = pd.to_numeric(frame[columns["Mach"]], errors="coerce").to_numpy()
    blumen = pd.to_numeric(frame[columns["blumen"]], errors="coerce").to_numpy()
    classic = pd.to_numeric(frame[columns["classic"]], errors="coerce").to_numpy()
    gep = pd.to_numeric(frame[columns["gep"]], errors="coerce").to_numpy()

    all_values = np.concatenate([blumen, classic, gep])
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        raise RuntimeError("No finite ci values in exact Blumen table.")

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.2, 6.2),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.89,
        bottom=0.13,
        top=0.82,
        wspace=0.08,
    )

    panels = [
        (axes[0], blumen, "Blumen digitized values"),
        (axes[1], classic, "Classical shooting at identical points"),
        (axes[2], gep, "PINN + GEP at identical points"),
    ]

    for axis, values, title in panels:
        axis.scatter(
            alpha,
            mach,
            c=values,
            s=36,
            marker="o",
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=0.25,
        )
        axis.set_title(title, fontsize=14)
        axis.set_xlabel(r"Wavenumber $\alpha$", fontsize=12)
        axis.grid(alpha=0.22)

    axes[0].set_ylabel(r"Mach number $M$", fontsize=12)
    fig.suptitle(
        r"Pointwise comparison at the exact digitized Blumen "
        r"$(\alpha,M)$ pairs",
        fontsize=18,
        y=0.96,
    )

    # Dedicated colorbar axis to the right of all three panels.
    colorbar_axis = fig.add_axes([0.915, 0.16, 0.016, 0.60])
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, cax=colorbar_axis)
    colorbar.set_label(r"$c_i$", fontsize=13)

    save_figure(
        fig,
        asset_root
        / "figures"
        / "Fig_Blumen_exact_points_classical_PINN_GEP",
    )


def canonical_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {
        "Mach": resolve_column(frame, ["Mach", "M"], "Mach"),
        "alpha": resolve_column(frame, ["alpha", "Alpha"], "alpha"),
        "classic": resolve_column(
            frame,
            ["ci_ref", "ci_classic", "ci_shooting", "ci_reference"],
            "classical ci",
        ),
        "gep": resolve_column(
            frame,
            ["ci_final", "pinn_matched_ci", "ci_gep", "ci_pinn_gep"],
            "PINN+GEP ci",
        ),
    }


def regular_interpolated_fields(
    frame: pd.DataFrame,
    columns: dict[str, str],
    grid_size: int = 900,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray, np.ma.MaskedArray]:
    work = pd.DataFrame(
        {
            "Mach": pd.to_numeric(frame[columns["Mach"]], errors="coerce"),
            "alpha": pd.to_numeric(frame[columns["alpha"]], errors="coerce"),
            "classic": pd.to_numeric(frame[columns["classic"]], errors="coerce"),
            "gep": pd.to_numeric(frame[columns["gep"]], errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan)

    work = (
        work.dropna()
        .groupby(["Mach", "alpha"], as_index=False)
        .agg(classic=("classic", "mean"), gep=("gep", "mean"))
    )

    if len(work) < 10:
        raise RuntimeError("Too few canonical points for isoline interpolation.")

    mach = work["Mach"].to_numpy(dtype=float)
    alpha = work["alpha"].to_numpy(dtype=float)

    triangulation = Triangulation(mach, alpha)
    classic_interpolator = LinearTriInterpolator(
        triangulation,
        work["classic"].to_numpy(dtype=float),
    )
    gep_interpolator = LinearTriInterpolator(
        triangulation,
        work["gep"].to_numpy(dtype=float),
    )

    mach_grid = np.linspace(float(mach.min()), float(mach.max()), grid_size)
    alpha_grid = np.linspace(float(alpha.min()), float(alpha.max()), grid_size)
    MM, AA = np.meshgrid(mach_grid, alpha_grid)

    classic_grid = np.ma.asarray(classic_interpolator(MM, AA))
    gep_grid = np.ma.asarray(gep_interpolator(MM, AA))

    # Keep only the validated atlas: M in [0.02,0.98],
    # eta=alpha/sqrt(1-M²) in [0.02,0.98].
    denominator = np.sqrt(np.clip(1.0 - MM**2, 1.0e-15, None))
    eta_grid = AA / denominator
    physical_mask = (
        (MM < 0.02)
        | (MM > 0.98)
        | (eta_grid < 0.02)
        | (eta_grid > 0.98)
    )

    common_mask = (
        physical_mask
        | np.ma.getmaskarray(classic_grid)
        | np.ma.getmaskarray(gep_grid)
        | ~np.isfinite(np.ma.filled(classic_grid, np.nan))
        | ~np.isfinite(np.ma.filled(gep_grid, np.nan))
    )

    classic_grid = np.ma.array(classic_grid, mask=common_mask)
    gep_grid = np.ma.array(gep_grid, mask=common_mask)
    return MM, AA, classic_grid, gep_grid


def build_isoline_figure(asset_root: Path) -> None:
    path = asset_root / "data" / "validation_pointwise_canonical.csv"
    frame = pd.read_csv(path)
    columns = canonical_columns(frame)

    MM, AA, classic_grid, gep_grid = regular_interpolated_fields(
        frame,
        columns,
    )

    levels = [0.05, 0.10, 0.15, 0.175]

    fig, axis = plt.subplots(figsize=(11.8, 8.4))

    # Draw PINN+GEP first with a thick dashed line; draw classical on top
    # with a thinner black line so coincident contours remain distinguishable.
    gep_contours = axis.contour(
        MM,
        AA,
        gep_grid,
        levels=levels,
        colors="tab:orange",
        linestyles="--",
        linewidths=3.0,
        zorder=2,
    )
    classic_contours = axis.contour(
        MM,
        AA,
        classic_grid,
        levels=levels,
        colors="black",
        linestyles="-",
        linewidths=1.25,
        zorder=3,
    )

    axis.clabel(
        classic_contours,
        levels=levels,
        fmt=lambda value: f"{value:g}",
        inline=True,
        fontsize=10,
    )

    mach_neutral = np.linspace(0.0, 1.0, 800)
    alpha_neutral = np.sqrt(
        np.clip(1.0 - mach_neutral**2, 0.0, None)
    )
    axis.plot(
        mach_neutral,
        alpha_neutral,
        color="tab:blue",
        linestyle=":",
        linewidth=1.8,
        zorder=1,
    )

    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.5,
                label="Classical shooting",
            ),
            Line2D(
                [0],
                [0],
                color="tab:orange",
                linewidth=3.0,
                linestyle="--",
                label="PINN + GEP",
            ),
            Line2D(
                [0],
                [0],
                color="tab:blue",
                linewidth=1.8,
                linestyle=":",
                label="Neutral boundary",
            ),
        ],
        frameon=False,
        loc="upper right",
        fontsize=12,
    )

    axis.set(
        xlabel=r"Mach number $M$",
        ylabel=r"Wavenumber $\alpha$",
        title=r"Constant-$c_i$ isolines: classical shooting versus PINN + GEP",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.20)
    fig.tight_layout()

    save_figure(
        fig,
        asset_root
        / "figures"
        / "Fig_ci_isolines_classical_vs_PINN_GEP",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()

    build_exact_error_figure(args.asset_root)
    build_exact_value_figure(args.asset_root)
    build_isoline_figure(args.asset_root)

    for name in (
        "Fig_Blumen_exact_points_absolute_errors",
        "Fig_Blumen_exact_points_classical_PINN_GEP",
        "Fig_ci_isolines_classical_vs_PINN_GEP",
    ):
        print(args.asset_root / "figures" / f"{name}.pdf")
        print(args.asset_root / "figures" / f"{name}.png")


if __name__ == "__main__":
    main()
