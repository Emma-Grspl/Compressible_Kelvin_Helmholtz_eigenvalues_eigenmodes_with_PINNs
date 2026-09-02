#!/usr/bin/env python3
"""
Pure local plotting script: interpolate an M=0.5 spectral slice from the
already validated canonical atlas and display it next to the existing
pointwise spectral-error map.

No checkpoint, PyTorch, shooting or GEP computation is required.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


VALUE_COLUMNS = ("ci_ref", "ci_seed", "ci_final")


def load_canonical(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    required = {
        "Mach",
        "eta",
        "alpha",
        "ci_ref",
        "ci_seed",
        "ci_final",
        "ci_final_abs_err",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Canonical CSV missing columns: {missing}")

    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return (
        frame.dropna(subset=list(required))
        .groupby(["Mach", "eta"], as_index=False)
        .agg(
            alpha=("alpha", "mean"),
            ci_ref=("ci_ref", "mean"),
            ci_seed=("ci_seed", "mean"),
            ci_final=("ci_final", "mean"),
            ci_final_abs_err=("ci_final_abs_err", "mean"),
        )
        .sort_values(["Mach", "eta"])
        .reset_index(drop=True)
    )


def select_bracketing_mach_lines(
    frame: pd.DataFrame,
    target_mach: float,
    minimum_points: int,
) -> tuple[float, pd.DataFrame, float, pd.DataFrame]:
    counts = frame.groupby("Mach").size()
    usable = counts.loc[counts >= minimum_points].index.to_numpy(dtype=float)
    if usable.size < 2:
        usable = counts.sort_values(ascending=False).head(8).index.to_numpy(
            dtype=float
        )

    lower_values = usable[usable < target_mach]
    upper_values = usable[usable > target_mach]

    if lower_values.size == 0 or upper_values.size == 0:
        raise RuntimeError(
            f"Cannot bracket M={target_mach} with sampled Mach lines."
        )

    lower_mach = float(np.max(lower_values))
    upper_mach = float(np.min(upper_values))
    lower = frame.loc[np.isclose(frame["Mach"], lower_mach)].sort_values("eta")
    upper = frame.loc[np.isclose(frame["Mach"], upper_mach)].sort_values("eta")
    return lower_mach, lower, upper_mach, upper


def interpolate_cut(
    frame: pd.DataFrame,
    target_mach: float,
    n_eta: int,
) -> tuple[pd.DataFrame, float, float]:
    lower_mach, lower, upper_mach, upper = select_bracketing_mach_lines(
        frame,
        target_mach,
        minimum_points=8,
    )

    eta_min = max(float(lower["eta"].min()), float(upper["eta"].min()))
    eta_max = min(float(lower["eta"].max()), float(upper["eta"].max()))
    if eta_max <= eta_min:
        raise RuntimeError("The two Mach lines have no common eta interval.")

    eta = np.linspace(eta_min, eta_max, int(n_eta))
    weight = (target_mach - lower_mach) / (upper_mach - lower_mach)

    output = {
        "Mach": np.full_like(eta, target_mach),
        "eta": eta,
        "alpha": eta * math.sqrt(max(1.0 - target_mach**2, 0.0)),
    }

    for column in VALUE_COLUMNS:
        lower_values = np.interp(
            eta,
            lower["eta"].to_numpy(dtype=float),
            lower[column].to_numpy(dtype=float),
        )
        upper_values = np.interp(
            eta,
            upper["eta"].to_numpy(dtype=float),
            upper[column].to_numpy(dtype=float),
        )
        output[column] = (1.0 - weight) * lower_values + weight * upper_values

    cut = pd.DataFrame(output)
    cut["ci_final_abs_err"] = np.abs(cut["ci_final"] - cut["ci_ref"])
    cut["source_mach_lower"] = lower_mach
    cut["source_mach_upper"] = upper_mach
    return cut, lower_mach, upper_mach


def plot_cut_and_heatmap(
    canonical: pd.DataFrame,
    cut: pd.DataFrame,
    target_mach: float,
    lower_mach: float,
    upper_mach: float,
    output_stem: Path,
) -> None:
    errors = canonical["ci_final_abs_err"].to_numpy(dtype=float)
    positive = errors[np.isfinite(errors) & (errors > 0.0)]
    if positive.size == 0:
        raise RuntimeError("No positive finite spectral errors.")
    vmin = max(float(np.min(positive)), 1.0e-14)
    vmax = float(np.max(positive))
    if vmax <= vmin:
        vmax = 10.0 * vmin

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.4, 5.8),
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )

    axes[0].plot(
        cut["alpha"],
        cut["ci_ref"],
        color="black",
        linewidth=2.0,
        label="Classical shooting",
    )
    axes[0].plot(
        cut["alpha"],
        cut["ci_seed"],
        color="tab:blue",
        linewidth=1.7,
        linestyle="--",
        label="Direct PINN",
    )
    axes[0].plot(
        cut["alpha"],
        cut["ci_final"],
        color="tab:orange",
        linewidth=1.8,
        linestyle="-.",
        label="PINN + GEP",
    )
    axes[0].set(
        xlabel=r"Wavenumber $\alpha$",
        ylabel=r"$c_i$",
        title=rf"Interpolated atlas cut at $M={target_mach:.3f}$",
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)
    axes[0].text(
        0.02,
        0.02,
        rf"Interpolated between sampled lines "
        rf"$M={lower_mach:.3f}$ and $M={upper_mach:.3f}$",
        transform=axes[0].transAxes,
        fontsize=8,
        va="bottom",
    )

    scatter = axes[1].scatter(
        canonical["Mach"],
        canonical["alpha"],
        c=np.clip(errors, vmin, vmax),
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        marker="s",
        s=36,
        linewidths=0,
    )
    axes[1].axvline(
        target_mach,
        color="white",
        linewidth=2.6,
        linestyle="--",
        zorder=4,
    )
    axes[1].axvline(
        target_mach,
        color="black",
        linewidth=0.8,
        linestyle="--",
        label=rf"$M={target_mach:.3f}$",
        zorder=5,
    )
    mach_neutral = np.linspace(0.0, 1.0, 600)
    axes[1].plot(
        mach_neutral,
        np.sqrt(np.clip(1.0 - mach_neutral**2, 0.0, None)),
        color="black",
        linestyle=":",
        linewidth=1.0,
    )
    axes[1].set(
        xlabel=r"Mach number $M$",
        ylabel=r"Wavenumber $\alpha$",
        title=r"Pointwise $|c_i^{PINN+GEP}-c_i^{class}|$",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axes[1].grid(alpha=0.12)
    axes[1].legend(frameon=False, loc="lower left")
    colorbar = fig.colorbar(scatter, ax=axes[1], pad=0.025)
    colorbar.set_label(r"Absolute $c_i$ error")

    fig.suptitle(
        "Subsonic spectral cut and global validation error",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=320,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-csv",
        type=Path,
        default=Path(
            "assets/pinn_subsonic/csv/joint_ci_mode_final_assets_v3/data/"
            "Table_validation_pointwise_canonical.csv"
        ),
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(
            "assets/pinn_subsonic/joint_ci_mode_final_assets_v3"
        ),
    )
    parser.add_argument("--mach", type=float, default=0.5)
    parser.add_argument("--n-eta", type=int, default=160)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    canonical_path = (
        args.canonical_csv
        if args.canonical_csv.is_absolute()
        else root / args.canonical_csv
    )
    asset_root = (
        args.asset_root
        if args.asset_root.is_absolute()
        else root / args.asset_root
    )

    canonical = load_canonical(canonical_path)
    cut, lower_mach, upper_mach = interpolate_cut(
        canonical,
        float(args.mach),
        int(args.n_eta),
    )

    data_path = (
        asset_root
        / "data"
        / f"spectral_cut_M{args.mach:.3f}_interpolated.csv"
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    cut.to_csv(data_path, index=False)

    output_stem = (
        asset_root
        / "figures"
        / f"Fig_subsonic_spectral_cut_and_error_heatmap_M"
        f"{int(round(100 * args.mach)):03d}"
    )
    plot_cut_and_heatmap(
        canonical,
        cut,
        float(args.mach),
        lower_mach,
        upper_mach,
        output_stem,
    )

    print(data_path)
    print(output_stem.with_suffix(".pdf"))
    print(output_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
