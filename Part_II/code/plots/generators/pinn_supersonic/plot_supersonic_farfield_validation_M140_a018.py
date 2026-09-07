#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_article_module(repo_root: Path):
    module_path = (
        repo_root
        / "code/scripts/data_preparation/prepare_build_supersonic_article_assets_final.py"
    )
    spec = importlib.util.spec_from_file_location(
        "supersonic_article_assets_final",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dedicated far-field validation figure for the canonical supersonic mode."
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    here = Path(__file__).resolve()
    repo_root = args.repo_root or here.parents[3]
    mod = load_article_module(repo_root)

    audit = mod.Audit()

    mode, metadata = mod.load_canonical_mode(repo_root, audit)

    y = mode["y"].to_numpy(float)
    p = mod.complex_from_frame(mode, "p")

    mach = float(metadata["Mach"])
    alpha = float(metadata["alpha"])
    c = complex(metadata["cr"], metadata["ci"])

    base_velocity = np.tanh(y)
    normalized_pressure = np.abs(p) / np.max(np.abs(p))

    left_gamma = mod.theoretical_gamma(alpha, mach, c, -1.0, "left")
    right_gamma = mod.theoretical_gamma(alpha, mach, c, 1.0, "right")

    left_fit = mod.fit_asymptotic_side(
        y, normalized_pressure, base_velocity, "left", left_gamma.real
    )
    right_fit = mod.fit_asymptotic_side(
        y, normalized_pressure, base_velocity, "right", right_gamma.real
    )

    out_fig_dir = repo_root / "assets/article/classical_supersonic/figures"
    out_tab_dir = repo_root / "assets/article/classical_supersonic/tables"
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_tab_dir.mkdir(parents=True, exist_ok=True)

    # table
    fit_rows = []
    for result in (left_fit, right_fit):
        fit_rows.append(
            {
                key: result[key]
                for key in (
                    "side",
                    "y_fit_min",
                    "y_fit_max",
                    "n_fit_points",
                    "kappa_theory",
                    "kappa_fit",
                    "absolute_difference",
                    "relative_difference",
                )
            }
        )
    table_path = out_tab_dir / "Tab_supersonic_farfield_validation_M140_a018.csv"
    pd.DataFrame(fit_rows).to_csv(table_path, index=False)

    # figure
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)

    ax.semilogy(
        y,
        np.maximum(normalized_pressure, 1.0e-16),
        linewidth=1.8,
        label=r"$|\hat p|/\max_y |\hat p|$",
    )

    for result, linestyle, color in (
        (left_fit, "--", "tab:orange"),
        (right_fit, ":", "tab:green"),
    ):
        idx = result["indices"]
        fit_curve = np.exp(result["intercept"] + result["kappa_fit"] * y[idx])

        ax.semilogy(
            y[idx],
            np.maximum(normalized_pressure[idx], 1.0e-16),
            linestyle="none",
            marker="o",
            markersize=3.0,
            color=color,
            alpha=0.75,
            label=(
                f"{result['side']} fit window"
            ),
        )

        ax.semilogy(
            y[idx],
            fit_curve,
            linestyle=linestyle,
            linewidth=2.2,
            color=color,
            label=(
                f"{result['side']} fit: "
                rf"$\kappa_{{fit}}={result['kappa_fit']:.6f}$, "
                rf"$\kappa_{{th}}={result['kappa_theory']:.6f}$"
            ),
        )

    ax.set_xlabel(r"Transverse coordinate $y$")
    ax.set_ylabel(r"Normalized pressure amplitude")
    ax.set_title(
        rf"Far-field validation for the canonical mode: "
        rf"$M={mach:.1f}$, $\alpha={alpha:.2f}$, "
        rf"$c_r={metadata['cr']:.6f}$, $c_i={metadata['ci']:.6f}$"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    pdf_path = out_fig_dir / "Fig_supersonic_farfield_validation_M140_a018.pdf"
    png_path = out_fig_dir / "Fig_supersonic_farfield_validation_M140_a018.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {table_path}")
    print()
    print("LEFT FIT :", left_fit)
    print("RIGHT FIT:", right_fit)


if __name__ == "__main__":
    main()
