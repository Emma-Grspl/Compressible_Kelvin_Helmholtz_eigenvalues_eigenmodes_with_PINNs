#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "Mach",
    "alpha",
    "cr",
    "ci",
    "residual_norm",
}


def load_reference(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)

    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        raise KeyError(
            "Colonnes absentes du CSV canonique : "
            + ", ".join(missing)
        )

    numeric_columns = [
        "Mach",
        "alpha",
        "cr",
        "ci",
        "omega_i",
        "residual_norm",
        "matching_y",
        "max_step",
        "rtol",
        "atol",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    data = data.dropna(
        subset=["Mach", "alpha", "cr", "ci", "residual_norm"]
    ).copy()

    if "accepted" in data.columns:
        accepted = (
            data["accepted"]
            .astype(str)
            .str.lower()
            .isin(("true", "1", "yes"))
        )
        data = data.loc[accepted].copy()

    if "omega_i" not in data.columns:
        data["omega_i"] = data["alpha"] * data["ci"]

    data = (
        data.sort_values(["Mach", "alpha"])
        .drop_duplicates(["Mach", "alpha"], keep="last")
        .reset_index(drop=True)
    )

    return data


def save_figure(
    figure: plt.Figure,
    output_base: Path,
) -> None:
    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_branch_tracking(
    data: pd.DataFrame,
    figures_root: Path,
) -> None:
    mach_values = np.sort(data["Mach"].unique())

    norm = Normalize(
        vmin=float(mach_values.min()),
        vmax=float(mach_values.max()),
    )
    cmap = plt.get_cmap("viridis")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.8),
        constrained_layout=True,
        sharex=True,
    )

    ax_ci, ax_cr = axes

    for mach, group in data.groupby("Mach", sort=True):
        group = group.sort_values("alpha")
        color = cmap(norm(float(mach)))

        ax_ci.plot(
            group["alpha"],
            group["ci"],
            color=color,
            linewidth=1.35,
        )
        ax_ci.scatter(
            group["alpha"],
            group["ci"],
            color=color,
            s=10,
            zorder=3,
        )

        ax_cr.plot(
            group["alpha"],
            group["cr"],
            color=color,
            linewidth=1.35,
        )
        ax_cr.scatter(
            group["alpha"],
            group["cr"],
            color=color,
            s=10,
            zorder=3,
        )

    ax_ci.set_xlabel(r"Wavenumber $\alpha$")
    ax_ci.set_ylabel(r"Growth component $c_i$")
    ax_ci.set_title(r"(a) Unstable branches: $c_i(\alpha)$")
    ax_ci.grid(True, alpha=0.22)

    ax_cr.set_xlabel(r"Wavenumber $\alpha$")
    ax_cr.set_ylabel(r"Propagation component $c_r$")
    ax_cr.set_title(r"(b) Unstable branches: $c_r(\alpha)$")
    ax_cr.grid(True, alpha=0.22)

    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])

    colorbar = figure.colorbar(
        scalar,
        ax=axes,
        location="right",
        shrink=0.90,
        pad=0.025,
    )
    colorbar.set_label(r"Mach number $M$")

    save_figure(
        figure,
        figures_root
        / "Fig_supersonic_branch_tracking_cr_ci",
    )


def plot_parameter_maps(
    data: pd.DataFrame,
    figures_root: Path,
) -> None:
    quantities = [
        (
            "ci",
            r"$c_i$",
            "(a) Growth component",
        ),
        (
            "cr",
            r"$c_r$",
            "(b) Propagation component",
        ),
        (
            "omega_i",
            r"$\omega_i=\alpha c_i$",
            "(c) Temporal growth rate",
        ),
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.2, 4.6),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    for axis, (column, colorbar_label, title) in zip(
        axes,
        quantities,
        strict=True,
    ):
        scatter = axis.scatter(
            data["Mach"],
            data["alpha"],
            c=data[column],
            s=18,
            marker="s",
            linewidths=0.0,
            cmap="viridis",
        )

        axis.set_xlabel(r"Mach number $M$")
        axis.set_title(title)
        axis.grid(True, alpha=0.14)

        colorbar = figure.colorbar(
            scatter,
            ax=axis,
            shrink=0.86,
            pad=0.015,
        )
        colorbar.set_label(colorbar_label)

    axes[0].set_ylabel(r"Wavenumber $\alpha$")

    save_figure(
        figure,
        figures_root
        / "Fig_supersonic_reference_parameter_maps",
    )


def plot_residual_diagnostics(
    data: pd.DataFrame,
    figures_root: Path,
) -> pd.DataFrame:
    residual = data["residual_norm"].to_numpy(float)
    residual = np.clip(residual, np.finfo(float).tiny, None)

    grouped = (
        data.groupby("Mach", sort=True)
        .agg(
            n_points=("residual_norm", "size"),
            residual_median=("residual_norm", "median"),
            residual_p95=(
                "residual_norm",
                lambda values: values.quantile(0.95),
            ),
            residual_max=("residual_norm", "max"),
        )
        .reset_index()
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.7),
        constrained_layout=True,
    )

    ax_histogram, ax_by_mach = axes

    ax_histogram.hist(
        np.log10(residual),
        bins=34,
        edgecolor="black",
        linewidth=0.45,
    )
    ax_histogram.axvline(
        np.log10(1.0e-8),
        linestyle="--",
        linewidth=1.2,
        color="black",
        label=r"Acceptance threshold $10^{-8}$",
    )
    ax_histogram.set_xlabel(
        r"$\log_{10}\mathcal{J}$"
    )
    ax_histogram.set_ylabel("Number of reference points")
    ax_histogram.set_title(
        "(a) Distribution of the shooting mismatch"
    )
    ax_histogram.grid(True, alpha=0.20)
    ax_histogram.legend(frameon=False)

    ax_by_mach.plot(
        grouped["Mach"],
        grouped["residual_median"],
        marker="o",
        linewidth=1.5,
        label="Median",
    )
    ax_by_mach.plot(
        grouped["Mach"],
        grouped["residual_p95"],
        marker="s",
        linewidth=1.5,
        label="95th percentile",
    )
    ax_by_mach.plot(
        grouped["Mach"],
        grouped["residual_max"],
        marker="^",
        linewidth=1.5,
        label="Maximum",
    )
    ax_by_mach.axhline(
        1.0e-8,
        linestyle="--",
        linewidth=1.2,
        color="black",
        label=r"Acceptance threshold $10^{-8}$",
    )
    ax_by_mach.set_yscale("log")
    ax_by_mach.set_xlabel(r"Mach number $M$")
    ax_by_mach.set_ylabel(r"Shooting mismatch $\mathcal{J}$")
    ax_by_mach.set_title(
        "(b) Residual statistics by Mach number"
    )
    ax_by_mach.grid(True, which="both", alpha=0.20)
    ax_by_mach.legend(frameon=False, fontsize=8)

    save_figure(
        figure,
        figures_root
        / "Fig_supersonic_global_residual_diagnostics",
    )

    return grouped


def build_grid_table(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for mach, group in data.groupby("Mach", sort=True):
        residual = group["residual_norm"]

        rows.append(
            {
                "Mach": float(mach),
                "n_points": int(len(group)),
                "alpha_min": float(group["alpha"].min()),
                "alpha_max": float(group["alpha"].max()),
                "cr_min": float(group["cr"].min()),
                "cr_max": float(group["cr"].max()),
                "ci_min": float(group["ci"].min()),
                "ci_max": float(group["ci"].max()),
                "omega_i_min": float(group["omega_i"].min()),
                "omega_i_max": float(group["omega_i"].max()),
                "residual_median": float(residual.median()),
                "residual_p95": float(residual.quantile(0.95)),
                "residual_max": float(residual.max()),
                "n_residual_above_1e8": int(
                    (residual > 1.0e-8).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_dataset_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    residual = data["residual_norm"]

    return pd.DataFrame(
        [
            {
                "n_spectral_points": int(len(data)),
                "n_mach_values": int(data["Mach"].nunique()),
                "Mach_min": float(data["Mach"].min()),
                "Mach_max": float(data["Mach"].max()),
                "alpha_min": float(data["alpha"].min()),
                "alpha_max": float(data["alpha"].max()),
                "cr_min": float(data["cr"].min()),
                "cr_max": float(data["cr"].max()),
                "ci_min": float(data["ci"].min()),
                "ci_max": float(data["ci"].max()),
                "omega_i_max": float(data["omega_i"].max()),
                "residual_median": float(residual.median()),
                "residual_p95": float(
                    residual.quantile(0.95)
                ),
                "residual_max": float(residual.max()),
                "n_residual_above_1e8": int(
                    (residual > 1.0e-8).sum()
                ),
            }
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--article-root",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    reference_path = args.reference.resolve()
    article_root = args.article_root.resolve()

    figures_root = article_root / "figures"
    tables_root = article_root / "tables"
    source_root = article_root / "source_data"

    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)

    data = load_reference(reference_path)

    plot_branch_tracking(data, figures_root)
    plot_parameter_maps(data, figures_root)

    residual_by_mach = plot_residual_diagnostics(
        data,
        figures_root,
    )

    grid_table = build_grid_table(data)
    dataset_summary = build_dataset_summary(data)

    grid_csv = (
        tables_root
        / "Tab_supersonic_reference_grid_by_mach.csv"
    )
    residual_csv = (
        tables_root
        / "Tab_supersonic_global_residual_summary.csv"
    )
    summary_csv = (
        tables_root
        / "Tab_supersonic_reference_dataset_summary.csv"
    )

    grid_table.to_csv(grid_csv, index=False)
    residual_by_mach.to_csv(residual_csv, index=False)
    dataset_summary.to_csv(summary_csv, index=False)

    grid_table.to_latex(
        tables_root
        / "Tab_supersonic_reference_grid_by_mach.tex",
        index=False,
        float_format=lambda value: f"{value:.4e}",
    )

    dataset_summary.to_latex(
        tables_root
        / "Tab_supersonic_reference_dataset_summary.tex",
        index=False,
        float_format=lambda value: f"{value:.4e}",
    )

    data.to_csv(
        source_root
        / "classical_supersonic_final_reference.csv",
        index=False,
    )

    print("=== SUPERSONIC SPECTRAL ARTICLE ASSETS ===")
    print("Reference points :", len(data))
    print("Mach values      :", data["Mach"].nunique())
    print(
        "Residual > 1e-8 :",
        int((data["residual_norm"] > 1.0e-8).sum()),
    )
    print()
    print(dataset_summary.to_string(index=False))
    print()
    print("Figures:")
    print(
        figures_root
        / "Fig_supersonic_branch_tracking_cr_ci.pdf"
    )
    print(
        figures_root
        / "Fig_supersonic_reference_parameter_maps.pdf"
    )
    print(
        figures_root
        / "Fig_supersonic_global_residual_diagnostics.pdf"
    )
    print()
    print("Tables:")
    print(grid_csv)
    print(residual_csv)
    print(summary_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
