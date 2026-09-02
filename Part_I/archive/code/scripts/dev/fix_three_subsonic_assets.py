from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.tri import Triangulation, LinearTriInterpolator


def resolve_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise RuntimeError(
        f"Impossible de trouver la colonne pour {label}. "
        f"Colonnes disponibles : {list(df.columns)}"
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_exact_error_figure(asset_root: Path) -> None:
    csv_path = asset_root / "data" / "Blumen_exact_point_comparison.csv"
    df = pd.read_csv(csv_path)

    col_alpha = resolve_column(df, ["alpha", "Alpha"], "alpha")
    col_mach = resolve_column(df, ["Mach", "M"], "Mach")

    col_blumen = resolve_column(
        df,
        [
            "ci_blumen",
            "ci_digitized",
            "ci_exact",
            "ci_target",
            "ci_blumen_digitized",
        ],
        "ci Blumen",
    )
    col_classic = resolve_column(
        df,
        [
            "ci_classic",
            "ci_ref",
            "ci_shooting",
            "ci_reference",
        ],
        "ci classical",
    )
    col_final = resolve_column(
        df,
        [
            "ci_final",
            "pinn_matched_ci",
            "ci_pinn_gep",
            "ci_gep",
        ],
        "ci PINN+GEP",
    )

    err_classic = np.abs(df[col_final].to_numpy(float) - df[col_classic].to_numpy(float))
    err_blumen = np.abs(df[col_final].to_numpy(float) - df[col_blumen].to_numpy(float))

    positive = np.concatenate(
        [err_classic[err_classic > 0.0], err_blumen[err_blumen > 0.0]]
    )
    vmin = positive.min() if positive.size else 1e-8
    vmax = max(err_classic.max(), err_blumen.max(), vmin * 10.0)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.1), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.12, right=0.90)

    sc0 = axes[0].scatter(
        df[col_alpha],
        df[col_mach],
        c=np.maximum(err_classic, vmin),
        s=42,
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        edgecolors="none",
    )
    axes[0].set_title("PINN + GEP versus classical shooting", fontsize=16)
    axes[0].set_xlabel(r"Wavenumber $\alpha$", fontsize=13)
    axes[0].set_ylabel(r"Mach number $M$", fontsize=13)
    axes[0].grid(True, alpha=0.25)

    sc1 = axes[1].scatter(
        df[col_alpha],
        df[col_mach],
        c=np.maximum(err_blumen, vmin),
        s=42,
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        edgecolors="none",
    )
    axes[1].set_title("PINN + GEP versus digitized Blumen", fontsize=16)
    axes[1].set_xlabel(r"Wavenumber $\alpha$", fontsize=13)
    axes[1].grid(True, alpha=0.25)

    cbar = fig.colorbar(sc1, ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label(r"Absolute $c_i$ error", fontsize=13)

    save_figure(fig, asset_root / "figures" / "Fig_Blumen_exact_points_absolute_errors")


def build_exact_pointwise_figure(asset_root: Path) -> None:
    csv_path = asset_root / "data" / "Blumen_exact_point_comparison.csv"
    df = pd.read_csv(csv_path)

    col_alpha = resolve_column(df, ["alpha", "Alpha"], "alpha")
    col_mach = resolve_column(df, ["Mach", "M"], "Mach")

    col_blumen = resolve_column(
        df,
        [
            "ci_blumen",
            "ci_digitized",
            "ci_exact",
            "ci_target",
            "ci_blumen_digitized",
        ],
        "ci Blumen",
    )
    col_classic = resolve_column(
        df,
        [
            "ci_classic",
            "ci_ref",
            "ci_shooting",
            "ci_reference",
        ],
        "ci classical",
    )
    col_final = resolve_column(
        df,
        [
            "ci_final",
            "pinn_matched_ci",
            "ci_pinn_gep",
            "ci_gep",
        ],
        "ci PINN+GEP",
    )

    all_vals = np.concatenate(
        [
            df[col_blumen].to_numpy(float),
            df[col_classic].to_numpy(float),
            df[col_final].to_numpy(float),
        ]
    )
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.2), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.08, right=0.90, top=0.86)

    datasets = [
        ("Blumen digitized values", col_blumen),
        ("Classical shooting at identical points", col_classic),
        ("PINN + GEP at identical points", col_final),
    ]

    scatter = None
    for ax, (title, col) in zip(axes, datasets):
        scatter = ax.scatter(
            df[col_alpha],
            df[col_mach],
            c=df[col],
            s=40,
            cmap="viridis",
            norm=Normalize(vmin=vmin, vmax=vmax),
            edgecolors="black",
            linewidths=0.25,
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r"Wavenumber $\alpha$", fontsize=12)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"Mach number $M$", fontsize=12)
    fig.suptitle(
        r"Pointwise comparison at the exact digitized Blumen $(\alpha,M)$ pairs",
        fontsize=18,
    )

    cbar = fig.colorbar(scatter, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(r"$c_i$", fontsize=13)

    save_figure(fig, asset_root / "figures" / "Fig_Blumen_exact_points_classical_PINN_GEP")


def build_isolines_figure(asset_root: Path) -> None:
    csv_path = asset_root / "data" / "validation_pointwise_canonical.csv"
    df = pd.read_csv(csv_path)

    col_mach = resolve_column(df, ["Mach", "M"], "Mach")
    col_alpha = resolve_column(df, ["alpha", "Alpha"], "alpha")
    col_ref = resolve_column(
        df,
        ["ci_classic", "ci_ref", "ci_shooting", "ci_reference"],
        "ci classical",
    )
    col_final = resolve_column(
        df,
        ["pinn_matched_ci", "ci_final", "ci_pinn_gep", "ci_gep"],
        "ci PINN+GEP",
    )

    work = df[[col_mach, col_alpha, col_ref, col_final]].dropna().copy()
    work = work.sort_values([col_mach, col_alpha]).reset_index(drop=True)

    M = work[col_mach].to_numpy(float)
    A = work[col_alpha].to_numpy(float)
    Zref = work[col_ref].to_numpy(float)
    Zfin = work[col_final].to_numpy(float)

    tri = Triangulation(M, A)
    interp_ref = LinearTriInterpolator(tri, Zref)
    interp_fin = LinearTriInterpolator(tri, Zfin)

    m_grid = np.linspace(M.min(), M.max(), 700)
    a_grid = np.linspace(A.min(), A.max(), 700)
    MM, AA = np.meshgrid(m_grid, a_grid)

    ZZ_ref = interp_ref(MM, AA)
    ZZ_fin = interp_fin(MM, AA)

    levels = [0.05, 0.10, 0.15, 0.175]

    fig, ax = plt.subplots(figsize=(12.5, 8.5))

    cs_ref = ax.contour(
        MM,
        AA,
        ZZ_ref,
        levels=levels,
        colors="black",
        linewidths=2.0,
    )

    cs_fin = ax.contour(
        MM,
        AA,
        ZZ_fin,
        levels=levels,
        colors="tab:orange",
        linewidths=2.8,
        linestyles="--",
    )

    ax.clabel(cs_ref, fmt="%g", inline=True, fontsize=12)
    ax.clabel(cs_fin, fmt="%g", inline=True, fontsize=12)

    # frontière neutre approximée par alpha max pour chaque Mach
    neutral = (
        work.groupby(col_mach, as_index=False)[col_alpha]
        .max()
        .sort_values(col_mach)
    )
    ax.plot(
        neutral[col_mach],
        neutral[col_alpha],
        linestyle=":",
        linewidth=2.0,
        color="tab:blue",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Mach number $M$", fontsize=14)
    ax.set_ylabel(r"Wavenumber $\alpha$", fontsize=14)
    ax.set_title(
        r"Constant-$c_i$ isolines: classical shooting versus PINN + GEP",
        fontsize=18,
    )
    ax.grid(True, alpha=0.25)

    legend_lines = [
        Line2D([0], [0], color="black", lw=2.0, label="Classical shooting"),
        Line2D([0], [0], color="tab:orange", lw=2.8, ls="--", label="PINN + GEP"),
        Line2D([0], [0], color="tab:blue", lw=2.0, ls=":", label="Neutral boundary"),
    ]
    ax.legend(handles=legend_lines, loc="upper right", fontsize=13, frameon=False)

    save_figure(fig, asset_root / "figures" / "Fig_ci_isolines_classical_vs_PINN_GEP")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True, type=Path)
    args = parser.parse_args()

    build_exact_error_figure(args.asset_root)
    build_exact_pointwise_figure(args.asset_root)
    build_isolines_figure(args.asset_root)

    print("Rebuilt:")
    print(args.asset_root / "figures" / "Fig_Blumen_exact_points_absolute_errors.pdf")
    print(args.asset_root / "figures" / "Fig_Blumen_exact_points_classical_PINN_GEP.pdf")
    print(args.asset_root / "figures" / "Fig_ci_isolines_classical_vs_PINN_GEP.pdf")


if __name__ == "__main__":
    main()
