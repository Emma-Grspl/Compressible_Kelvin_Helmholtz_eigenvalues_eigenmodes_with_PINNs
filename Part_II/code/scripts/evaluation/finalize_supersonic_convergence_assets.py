#!/usr/bin/env python3
"""Finalize supersonic convergence/robustness article assets.

This script deliberately reuses the already-generated, non-quarantine article
figures because the original M=1.4 robustness CSVs and exported high-Mach field
files are absent from the current repository.

It also creates the requested convergence/robustness summary table from the
available article tables:
  - supersonic_high_mach_convergence_by_point.csv
  - supersonic_high_mach_convergence_by_setting.csv
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    start = start.expanduser().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "assets").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate repository root containing assets/.")


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def copy_pair(source_pdf: Path, destination_pdf: Path) -> None:
    source_png = source_pdf.with_suffix(".png")
    destination_png = destination_pdf.with_suffix(".png")

    require_file(source_pdf)
    require_file(source_png)

    destination_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_pdf, destination_pdf)
    shutil.copy2(source_png, destination_png)

    print(f"Copied: {destination_pdf}")
    print(f"Copied: {destination_png}")


def combine_robustness_figures(
    box_png: Path,
    integration_png: Path,
    output_pdf: Path,
    dpi: int,
) -> None:
    require_file(box_png)
    require_file(integration_png)

    box_image = mpimg.imread(box_png)
    integration_image = mpimg.imread(integration_png)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.2, 6.2),
        constrained_layout=True,
    )

    axes[0].imshow(box_image)
    axes[0].set_title("(a) Domain-size robustness")
    axes[0].axis("off")

    axes[1].imshow(integration_image)
    axes[1].set_title("(b) Integration and matching robustness")
    axes[1].axis("off")

    fig.suptitle(
        "Supersonic classical robustness to domain size and matching configuration",
        fontsize=15,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_pdf.with_suffix(".png")

    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)

    print(f"Saved: {output_pdf}")
    print(f"Saved: {output_png}")


def build_representative_summary(
    point_path: Path,
    setting_path: Path,
    maximum_rows: int = 8,
) -> pd.DataFrame:
    points = pd.read_csv(require_file(point_path))
    settings = pd.read_csv(require_file(setting_path))

    required_point_columns = {
        "Mach",
        "alpha",
        "n_success_settings",
        "cr_range",
        "ci_range",
        "convergence_status",
    }
    missing_points = sorted(required_point_columns.difference(points.columns))
    if missing_points:
        raise KeyError(
            f"Missing columns in {point_path}: {missing_points}. "
            f"Available: {list(points.columns)}"
        )

    required_setting_columns = {
        "Mach",
        "alpha",
        "stage1_mismatch",
    }
    missing_settings = sorted(required_setting_columns.difference(settings.columns))
    if missing_settings:
        raise KeyError(
            f"Missing columns in {setting_path}: {missing_settings}. "
            f"Available: {list(settings.columns)}"
        )

    numeric_point_columns = (
        "Mach",
        "alpha",
        "n_success_settings",
        "cr_range",
        "ci_range",
    )
    for column in numeric_point_columns:
        points[column] = pd.to_numeric(points[column], errors="coerce")

    for column in ("Mach", "alpha", "stage1_mismatch"):
        settings[column] = pd.to_numeric(settings[column], errors="coerce")

    residuals = (
        settings.dropna(subset=["Mach", "alpha", "stage1_mismatch"])
        .groupby(["Mach", "alpha"], as_index=False)
        .agg(
            residual_median=("stage1_mismatch", "median"),
            residual_max=("stage1_mismatch", "max"),
        )
    )

    summary = points.merge(
        residuals,
        on=["Mach", "alpha"],
        how="left",
    )

    summary = summary.rename(
        columns={
            "cr_range": "max_delta_cr",
            "ci_range": "max_delta_ci",
            "n_success_settings": "n_configurations",
            "convergence_status": "status",
        }
    )

    # Deterministic representative sampling:
    # low, central and high alpha for each Mach, plus worst spreads.
    selected_indices: list[int] = []

    for _, mach_group in summary.groupby("Mach", sort=True):
        ordered = mach_group.sort_values("alpha", kind="mergesort")
        if ordered.empty:
            continue

        selected_indices.append(int(ordered.index[0]))
        selected_indices.append(int(ordered.index[len(ordered) // 2]))
        selected_indices.append(int(ordered.index[-1]))

        worst_ci = ordered["max_delta_ci"].fillna(-np.inf).idxmax()
        worst_cr = ordered["max_delta_cr"].fillna(-np.inf).idxmax()
        selected_indices.extend((int(worst_ci), int(worst_cr)))

    selected_indices = list(dict.fromkeys(selected_indices))

    if len(selected_indices) < maximum_rows:
        additional = (
            summary.assign(
                _severity=
                summary["max_delta_cr"].fillna(0.0)
                + summary["max_delta_ci"].fillna(0.0)
            )
            .sort_values("_severity", ascending=False, kind="mergesort")
            .index.astype(int)
            .tolist()
        )
        for index in additional:
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= maximum_rows:
                break

    selected_indices = selected_indices[:maximum_rows]

    output = (
        summary.loc[selected_indices]
        .sort_values(["Mach", "alpha"], kind="mergesort")
        .reset_index(drop=True)
    )

    return output[
        [
            "Mach",
            "alpha",
            "max_delta_cr",
            "max_delta_ci",
            "residual_median",
            "residual_max",
            "n_configurations",
            "status",
        ]
    ]


def write_summary_tables(
    summary: pd.DataFrame,
    output_csv: Path,
    output_tex: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    formatters = {
        "Mach": lambda value: f"{value:.2f}",
        "alpha": lambda value: f"{value:.4f}",
        "max_delta_cr": lambda value: (
            f"{value:.3e}" if pd.notna(value) else "--"
        ),
        "max_delta_ci": lambda value: (
            f"{value:.3e}" if pd.notna(value) else "--"
        ),
        "residual_median": lambda value: (
            f"{value:.3e}" if pd.notna(value) else "--"
        ),
        "residual_max": lambda value: (
            f"{value:.3e}" if pd.notna(value) else "--"
        ),
        "n_configurations": lambda value: str(int(value)),
        "status": str,
    }

    latex = summary.to_latex(
        index=False,
        escape=True,
        formatters=formatters,
        column_format="rrcccccl",
        caption=(
            "Representative high-Mach convergence diagnostics. "
            "The eigenvalue spreads are the ranges across successful numerical "
            "settings, and the residual statistics are based on the stage-1 "
            "shooting mismatch."
        ),
        label="tab:supersonic_convergence_robustness_summary",
    )

    output_tex.write_text(latex, encoding="utf-8")

    print(f"Saved: {output_csv}")
    print(f"Saved: {output_tex}")


def main() -> None:
    args = parse_args()
    repo = find_repo_root(args.repo_root)

    article_figure_directory = (
        repo
        / "assets"
        / "article"
        / "classical_supersonic"
        / "figures"
    )
    article_source_directory = (
        repo
        / "assets"
        / "article"
        / "classical_supersonic"
        / "source_data"
    )
    article_table_directory = (
        repo
        / "assets"
        / "article"
        / "classical_supersonic"
        / "tables"
    )

    existing_full = (
        repo
        / "assets"
        / "classic_supersonic"
        / "article"
        / "convergence_modalfix_v1_full"
    )
    existing_v2 = (
        repo
        / "assets"
        / "classic_supersonic"
        / "article"
        / "convergence_modalfix_v2"
    )

    # Existing validated shooting/spectral convergence figure.
    copy_pair(
        existing_full / "fig_classical_shooting_spectral_convergence.pdf",
        article_figure_directory / "Fig_supersonic_eigenvalue_convergence.pdf",
    )

    # Existing validated modal-convergence figure. This is reused because the
    # exported per-setting fields required to recompute phase convergence are absent.
    copy_pair(
        existing_full / "fig_classical_modal_convergence.pdf",
        article_figure_directory
        / "Fig_supersonic_core_tail_phase_convergence.pdf",
    )

    # Combine the existing box and integration/matching figures.
    combine_robustness_figures(
        existing_v2 / "fig_classical_box_convergence.png",
        existing_v2 / "fig_classical_integration_convergence.png",
        article_figure_directory
        / "Fig_supersonic_box_and_matching_robustness.pdf",
        args.dpi,
    )

    summary = build_representative_summary(
        article_table_directory
        / "supersonic_high_mach_convergence_by_point.csv",
        article_source_directory
        / "supersonic_high_mach_convergence_by_setting.csv",
    )

    write_summary_tables(
        summary,
        article_source_directory
        / "Tab_supersonic_convergence_robustness_summary.csv",
        article_source_directory
        / "Tab_supersonic_convergence_robustness_summary.tex",
    )

    print("All available convergence/robustness assets were finalized.")


if __name__ == "__main__":
    main()
