from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SPECTRAL_CSV = ROOT_DIR / "assets" / "classic_supersonic" / "shooting" / "supersonic_reference_core_local_spectral.csv"
MODAL_CSV = ROOT_DIR / "assets" / "classic_supersonic" / "shooting" / "supersonic_reference_core_local_modal.csv"
M140_RECONFIRM_CSV = (
    ROOT_DIR
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "experiment_point_batch_M140_branch_guided_reconfirm_2026-06-09"
    / "assets"
    / "classic_supersonic"
    / "shooting"
    / "supersonic_shooting_point_batch_M140_branch_guided_reconfirm_summary.csv"
)
OUTPUT_DIR = ROOT_DIR / "assets" / "classic_supersonic" / "plots"


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spectral = pd.read_csv(SPECTRAL_CSV)
    modal = pd.read_csv(MODAL_CSV)
    m140 = pd.read_csv(M140_RECONFIRM_CSV).rename(
        columns={
            "best_shooting_ci": "reference_ci",
            "best_shooting_cr": "reference_cr",
        }
    )
    m140["trusted_modal"] = True
    m140["trusted_spectral"] = True
    return spectral, modal, m140


def _error_text(df: pd.DataFrame) -> str:
    rel = ((df["reference_ci"] - df["blumen_ci"]).abs() / df["blumen_ci"]).mean()
    return f"ecart moyen : {100.0 * rel:.0f} %"


def plot_ci_slices(spectral: pd.DataFrame, modal: pd.DataFrame, m140: pd.DataFrame) -> Path:
    colors = {
        1.3: "#0f766e",
        1.4: "#b91c1c",
        1.5: "#d97706",
    }
    titles = {
        1.3: "M = 1.3 : bon accord avec Blumen",
        1.4: "M = 1.4 : branche robuste mais decalee",
        1.5: "M = 1.5 : bonne tendance, ecart encore visible",
    }

    data_by_mach: dict[float, pd.DataFrame] = {
        1.3: spectral.loc[spectral["Mach"] == 1.3].sort_values("alpha").copy(),
        1.4: m140.sort_values("alpha").copy(),
        1.5: spectral.loc[spectral["Mach"] == 1.5].sort_values("alpha").copy(),
    }
    modal_by_mach: dict[float, pd.DataFrame] = {
        1.3: modal.loc[modal["Mach"] == 1.3].sort_values("alpha").copy(),
        1.4: m140.sort_values("alpha").copy(),
        1.5: modal.loc[modal["Mach"] == 1.5].sort_values("alpha").copy(),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), sharey=True)
    for ax, mach in zip(axes, [1.3, 1.4, 1.5]):
        df = data_by_mach[mach]
        modal_df = modal_by_mach[mach]
        color = colors[mach]

        ax.plot(
            df["alpha"],
            df["blumen_ci"],
            color="black",
            linestyle="--",
            linewidth=1.8,
            marker="o",
            markersize=3.4,
            label="Blumen",
            zorder=2,
        )
        ax.plot(
            df["alpha"],
            df["reference_ci"],
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=5.2,
            label="Classique",
            zorder=3,
        )
        if not modal_df.empty:
            ax.scatter(
                modal_df["alpha"],
                modal_df["reference_ci"],
                s=90,
                marker="*",
                facecolor="#0f172a",
                edgecolor="white",
                linewidth=0.5,
                label="Mode robuste",
                zorder=4,
            )

        ax.set_title(titles[mach])
        ax.set_xlabel(r"$\alpha$")
        ax.grid(True, linestyle=":", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.03,
            0.08,
            _error_text(df.loc[df["best_status"] == "validated"] if "best_status" in df.columns else df),
            transform=ax.transAxes,
            fontsize=10,
            color=color,
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.9, "boxstyle": "round,pad=0.25"},
        )

    axes[0].set_ylabel(r"$c_i$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(r"Comparaison a Blumen sur trois lignes de Mach", y=1.08, fontsize=15)
    fig.tight_layout()

    out = OUTPUT_DIR / "08_supersonic_ci_mach_slices_presentation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_status_line(
    ax: plt.Axes,
    y: float,
    spectral_df: pd.DataFrame,
    modal_df: pd.DataFrame,
    line_color: str,
    marker_color: str,
    label_prefix: str,
) -> None:
    spectral_df = spectral_df.sort_values("alpha")
    if spectral_df.empty:
        return
    ax.hlines(y, spectral_df["alpha"].min(), spectral_df["alpha"].max(), color=line_color, linewidth=1.4, alpha=0.55)
    ax.scatter(
        spectral_df["alpha"],
        np.full(len(spectral_df), y),
        s=56,
        marker="o",
        facecolor=marker_color,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    if not modal_df.empty:
        modal_df = modal_df.sort_values("alpha")
        ax.scatter(
            modal_df["alpha"],
            np.full(len(modal_df), y),
            s=150,
            marker="*",
            facecolor="#0f766e",
            edgecolor="black",
            linewidth=0.55,
            zorder=4,
        )
        ax.plot(modal_df["alpha"], np.full(len(modal_df), y), color="#0f766e", linewidth=2.2, alpha=0.8, zorder=2)

    ax.text(
        spectral_df["alpha"].min() - 0.018,
        y,
        label_prefix,
        ha="right",
        va="center",
        fontsize=11,
        color="#111827",
    )


def plot_modal_difficulty(spectral: pd.DataFrame, modal: pd.DataFrame, m140: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10.4, 4.8))

    _plot_status_line(
        ax,
        1.2,
        spectral.loc[spectral["Mach"] == 1.2],
        modal.loc[modal["Mach"] == 1.2],
        line_color="#94a3b8",
        marker_color="#f59e0b",
        label_prefix="M = 1.2",
    )
    _plot_status_line(
        ax,
        1.3,
        spectral.loc[spectral["Mach"] == 1.3],
        modal.loc[modal["Mach"] == 1.3],
        line_color="#94a3b8",
        marker_color="#f59e0b",
        label_prefix="M = 1.3",
    )
    _plot_status_line(
        ax,
        1.5,
        spectral.loc[spectral["Mach"] == 1.5],
        modal.loc[modal["Mach"] == 1.5],
        line_color="#94a3b8",
        marker_color="#f59e0b",
        label_prefix="M = 1.5",
    )

    m140 = m140.sort_values("alpha")
    ax.hlines(1.4, m140["alpha"].min(), m140["alpha"].max(), color="#c084fc", linewidth=2.4, alpha=0.85)
    ax.scatter(
        m140["alpha"],
        np.full(len(m140), 1.4),
        s=75,
        marker="D",
        facecolor="#7c3aed",
        edgecolor="black",
        linewidth=0.55,
        zorder=4,
    )
    ax.text(
        m140["alpha"].min() - 0.018,
        1.4,
        "M = 1.4",
        ha="right",
        va="center",
        fontsize=11,
        color="#111827",
    )

    ax.annotate(
        "a M=1.5, on garde souvent la croissance\nmais on perd vite le mode robuste",
        xy=(0.19, 1.5),
        xytext=(0.214, 1.585),
        arrowprops={"arrowstyle": "->", "color": "#9a3412", "lw": 1.2},
        fontsize=10,
        color="#9a3412",
        ha="left",
    )
    ax.annotate(
        "a M=1.4, on trouve un mode robuste,\nmais sur une branche qui ne colle pas a Blumen",
        xy=(0.151, 1.4),
        xytext=(0.185, 1.335),
        arrowprops={"arrowstyle": "->", "color": "#6d28d9", "lw": 1.2},
        fontsize=10,
        color="#6d28d9",
        ha="left",
    )

    spectral_handle = plt.Line2D(
        [], [], linestyle="none", marker="o", markerfacecolor="#f59e0b", markeredgecolor="black", markersize=7, label=r"Point avec $c_i$ fiable"
    )
    modal_handle = plt.Line2D(
        [], [], linestyle="none", marker="*", markerfacecolor="#0f766e", markeredgecolor="black", markersize=12, label="Point avec mode robuste"
    )
    alt_branch_handle = plt.Line2D(
        [], [], linestyle="none", marker="D", markerfacecolor="#7c3aed", markeredgecolor="black", markersize=7, label="Branche robuste mais decalee"
    )
    ax.legend(handles=[spectral_handle, modal_handle, alt_branch_handle], loc="lower right", frameon=True)

    ax.set_title("Pourquoi la reference modale supersonique reste fragmentaire")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$M$")
    ax.set_xlim(0.085, 0.285)
    ax.set_ylim(1.16, 1.62)
    ax.set_yticks([1.2, 1.3, 1.4, 1.5])
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUTPUT_DIR / "09_supersonic_modal_difficulty_presentation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    _setup_style()
    spectral, modal, m140 = _load_data()
    out1 = plot_ci_slices(spectral, modal, m140)
    out2 = plot_modal_difficulty(spectral, modal, m140)
    print(out1)
    print(out2)


if __name__ == "__main__":
    main()
