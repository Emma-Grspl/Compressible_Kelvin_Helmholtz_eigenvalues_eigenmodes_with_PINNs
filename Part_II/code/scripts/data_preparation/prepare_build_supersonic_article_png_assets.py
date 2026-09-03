from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


REPO = Path.cwd()

FREEZE = (
    REPO
    / "assets/pinn_supersonic/datasets/atlas2d_v1/freeze"
)

VAL_ROOT = (
    REPO
    / "assets/pinn_supersonic/atlas2d_v1_continuousM"
)

OUT = (
    REPO
    / "assets/p2-supersonic-pinn-article-png"
)
OUT.mkdir(parents=True, exist_ok=True)

BUDGETS = [76, 60, 48, 36, 24]

# Frozen atlas geometry
M_BANDS = [
    (1.00, 1.25),
    (1.15, 1.45),
    (1.35, 1.65),
    (1.55, 1.90),
]

A_BANDS = [
    (0.05, 0.13),
    (0.10, 0.22),
    (0.19, 0.36),
]


def load_anchors(n: int) -> pd.DataFrame:
    path = FREEZE / f"anchors_N{n}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_validation(n: int) -> pd.DataFrame:
    path = (
        VAL_ROOT
        / f"N{n}"
        / "validation"
        / f"N{n}_validation_predictions_64.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    required = {"Mach", "alpha", "spectral_error"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{path} missing columns: {sorted(missing)}"
        )

    return df


def atlas_base(ax):
    # Draw 12 overlapping charts
    for i, (m0, m1) in enumerate(M_BANDS):
        for j, (a0, a1) in enumerate(A_BANDS):
            rect = Rectangle(
                (m0, a0),
                m1 - m0,
                a1 - a0,
                fill=False,
                linewidth=1.5,
            )
            ax.add_patch(rect)

            ax.text(
                0.5 * (m0 + m1),
                0.5 * (a0 + a1),
                f"C{i}{j}",
                ha="center",
                va="center",
                fontsize=9,
            )

    ax.set_xlim(0.98, 1.92)
    ax.set_ylim(0.04, 0.37)
    ax.set_xlabel("Mach number")
    ax.set_ylabel(r"Wavenumber $\alpha$")
    ax.grid(True, alpha=0.25)


def build_summary_table() -> pd.DataFrame:
    rows = []

    for n in BUDGETS:
        df = load_validation(n)
        err = df["spectral_error"].to_numpy(float)

        rows.append(
            {
                "budget": n,
                "n_points": len(df),
                "mean": float(np.mean(err)),
                "median": float(np.median(err)),
                "p90": float(np.quantile(err, 0.90)),
                "p95": float(np.quantile(err, 0.95)),
                "max": float(np.max(err)),
                "n_le_1e-2": int(np.sum(err <= 1.0e-2)),
                "n_le_1e-3": int(np.sum(err <= 1.0e-3)),
                "n_le_1e-4": int(np.sum(err <= 1.0e-4)),
                "n_le_1e-5": int(np.sum(err <= 1.0e-5)),
            }
        )

    summary = pd.DataFrame(rows).sort_values("budget")
    summary.to_csv(
        OUT / "Tab_supersonic_V64_budget_summary.csv",
        index=False,
    )
    return summary


def fig_atlas_geometry():
    fig, ax = plt.subplots(figsize=(8, 5))
    atlas_base(ax)
    ax.set_title("Supersonic atlas geometry (12 overlapping charts)")
    fig.tight_layout()
    fig.savefig(
        OUT / "Fig_supersonic_atlas_geometry.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def fig_anchor_budget(n: int):
    df = load_anchors(n)

    fig, ax = plt.subplots(figsize=(8, 5))
    atlas_base(ax)

    ax.scatter(
        df["Mach"].to_numpy(float),
        df["alpha"].to_numpy(float),
        s=30,
        marker="o",
        label=f"N{n} anchors",
    )
    ax.legend()
    ax.set_title(f"Supersonic anchor budget N{n}")
    fig.tight_layout()
    fig.savefig(
        OUT / f"Fig_supersonic_anchor_budget_N{n}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def fig_budget_mean_max(summary: pd.DataFrame):
    x = summary["budget"].to_numpy(int)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(x, summary["mean"], marker="o", label="mean spectral error")
    ax.plot(x, summary["p95"], marker="s", label="95th percentile")
    ax.plot(x, summary["max"], marker="^", label="max spectral error")
    ax.set_xlabel("Number of anchors")
    ax.set_ylabel("Validation spectral error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("V64 validation error versus anchor budget")
    fig.tight_layout()
    fig.savefig(
        OUT / "Fig_supersonic_V64_error_vs_budget.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def fig_budget_ecdf():
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for n in BUDGETS:
        df = load_validation(n)
        err = np.sort(df["spectral_error"].to_numpy(float))
        y = np.arange(1, len(err) + 1) / len(err)

        ax.step(
            err,
            y,
            where="post",
            label=f"N{n}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Validation spectral error")
    ax.set_ylabel("Empirical CDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("V64 empirical CDF of spectral errors")
    fig.tight_layout()
    fig.savefig(
        OUT / "Fig_supersonic_V64_error_ecdf.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def fig_budget_boxplot():
    data = []
    labels = []

    for n in BUDGETS:
        df = load_validation(n)
        data.append(df["spectral_error"].to_numpy(float))
        labels.append(f"N{n}")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_yscale("log")
    ax.set_xlabel("Anchor budget")
    ax.set_ylabel("Validation spectral error")
    ax.grid(True, alpha=0.3)
    ax.set_title("Distribution of V64 spectral errors")
    fig.tight_layout()
    fig.savefig(
        OUT / "Fig_supersonic_V64_error_boxplot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def fig_validation_map(n: int):
    df = load_validation(n).copy()

    eps = 1.0e-12
    color_values = np.log10(
        np.maximum(df["spectral_error"].to_numpy(float), eps)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    atlas_base(ax)

    sc = ax.scatter(
        df["Mach"].to_numpy(float),
        df["alpha"].to_numpy(float),
        c=color_values,
        s=55,
    )

    worst = df.nlargest(5, "spectral_error")

    for _, row in worst.iterrows():
        ax.annotate(
            f"({row['Mach']:.2f}, {row['alpha']:.3f})",
            (float(row["Mach"]), float(row["alpha"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\log_{10}(\mathrm{spectral\ error})$")

    ax.set_title(f"V64 validation error map — N{n}")
    fig.tight_layout()
    fig.savefig(
        OUT / f"Fig_supersonic_V64_error_map_N{n}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_readme(summary: pd.DataFrame):
    lines = []
    lines.append("SUPSERSONIC ARTICLE PNG ASSETS")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Generated figures:")
    for p in sorted(OUT.glob("*.png")):
        lines.append(f"- {p.name}")
    lines.append("")
    lines.append("Validation summary:")
    lines.append(summary.to_string(index=False))

    (OUT / "README_generated_assets.txt").write_text(
        "\n".join(lines)
    )


def main():
    print("=" * 80)
    print("BUILDING SUPERSONIC ARTICLE PNG ASSETS")
    print("=" * 80)

    summary = build_summary_table()

    fig_atlas_geometry()
    fig_anchor_budget(24)
    fig_anchor_budget(76)
    fig_budget_mean_max(summary)
    fig_budget_ecdf()
    fig_budget_boxplot()
    fig_validation_map(76)
    fig_validation_map(24)

    write_readme(summary)

    print()
    print("WRITTEN FILES:")
    for p in sorted(OUT.glob("*")):
        print(p)

    print()
    print("DONE")


if __name__ == "__main__":
    main()
