from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path("assets/pinn_subsonic/article/figures")
OUTPUT_STEM = "Fig_spectral_modal_architecture_subsonic"


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    shade: float = 0.96,
    linewidth: float = 1.2,
    fontsize: float = 10.0,
) -> None:
    """Add a rounded rectangular architecture block."""

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        facecolor=str(shade),
        edgecolor="black",
        linewidth=linewidth,
    )
    ax.add_patch(patch)

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
    linewidth: float = 1.3,
) -> None:
    """Add a directed connection between two blocks."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        linestyle="--" if dashed else "-",
        color="black",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def draw_local_architecture(ax: plt.Axes) -> None:
    """Panel (a): local spectral--modal PINN."""

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        0.15,
        9.55,
        r"$\mathbf{(a)}$ Local spectral--modal PINN",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
    )

    # Inputs
    add_box(
        ax,
        0.4,
        6.9,
        1.5,
        1.0,
        r"Spectral input" "\n" r"$\alpha$",
        shade=0.98,
    )

    add_box(
        ax,
        0.4,
        2.1,
        1.5,
        1.2,
        r"Modal inputs" "\n" r"$(\xi,\alpha)$",
        shade=0.98,
    )

    # Spectral branch
    add_box(
        ax,
        2.7,
        6.75,
        2.1,
        1.3,
        "Spectral branch"
        "\n"
        r"$\mathcal{N}_{c_i,\phi}$",
        shade=0.90,
    )

    add_box(
        ax,
        5.7,
        6.9,
        1.4,
        1.0,
        r"$c_{i,\phi}(\alpha)$",
        shade=0.98,
    )

    # Modal experts
    add_box(
        ax,
        2.55,
        2.0,
        1.7,
        1.35,
        "Modal expert 1"
        "\n"
        r"$\alpha\leq\alpha_s$",
        shade=0.92,
    )

    add_box(
        ax,
        2.55,
        4.0,
        1.7,
        1.35,
        "Modal expert 2"
        "\n"
        r"$\alpha>\alpha_s$",
        shade=0.92,
    )

    add_box(
        ax,
        5.15,
        3.0,
        2.0,
        1.35,
        "Modal prediction"
        "\n"
        r"$\widehat{\mathbf{z}}_{\theta}$",
        shade=0.98,
    )

    # Physics coupling and losses
    add_box(
        ax,
        7.9,
        4.7,
        1.7,
        1.6,
        "Governing equation"
        "\n"
        "and far-field"
        "\n"
        "conditions",
        shade=0.88,
        fontsize=9.5,
    )

    add_box(
        ax,
        7.9,
        1.9,
        1.7,
        1.35,
        "Physics-informed"
        "\n"
        "modal loss",
        shade=0.96,
    )

    add_box(
        ax,
        5.45,
        8.55,
        1.9,
        0.9,
        "Sparse spectral"
        "\n"
        "anchors",
        shade=0.98,
        fontsize=9.5,
    )

    # Connections
    add_arrow(ax, (1.9, 7.4), (2.7, 7.4))
    add_arrow(ax, (4.8, 7.4), (5.7, 7.4))

    add_arrow(ax, (1.9, 2.7), (2.55, 2.7))
    add_arrow(ax, (1.9, 2.7), (2.55, 4.65))

    add_arrow(ax, (4.25, 2.7), (5.15, 3.45))
    add_arrow(ax, (4.25, 4.65), (5.15, 3.9))

    add_arrow(ax, (7.1, 7.4), (8.4, 6.3))
    add_arrow(ax, (7.15, 3.7), (7.9, 5.0))
    add_arrow(ax, (8.75, 4.7), (8.75, 3.25))

    add_arrow(
        ax,
        (6.4, 8.55),
        (6.4, 7.9),
        dashed=True,
    )

    ax.text(
        4.55,
        5.25,
        r"$\alpha_s=0.4$",
        fontsize=9.5,
        ha="center",
    )

    ax.text(
        7.5,
        6.9,
        "physical"
        "\n"
        "coupling",
        fontsize=8.5,
        ha="center",
        va="center",
    )


def draw_atlas_architecture(ax: plt.Axes) -> None:
    """Panel (b): parametric atlas architecture."""

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        0.15,
        9.55,
        r"$\mathbf{(b)}$ Parametric PINN atlas",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
    )

    # Inputs
    add_box(
        ax,
        0.35,
        6.9,
        1.55,
        1.1,
        "Physical"
        "\n"
        r"parameters $(M,\eta)$",
        shade=0.98,
    )

    add_box(
        ax,
        0.35,
        2.2,
        1.55,
        1.2,
        "Modal inputs"
        "\n"
        r"$(\xi,M,\eta)$",
        shade=0.98,
    )

    # Spectral interpolation
    add_box(
        ax,
        2.65,
        6.65,
        2.0,
        1.6,
        "Sparse spectral"
        "\n"
        "anchors and"
        "\n"
        "interpolation",
        shade=0.90,
        fontsize=9.5,
    )

    add_box(
        ax,
        5.45,
        6.9,
        1.55,
        1.1,
        r"$c_i(M,\eta)$",
        shade=0.98,
    )

    # Chart assignment and mode network
    add_box(
        ax,
        2.65,
        2.15,
        2.0,
        1.4,
        "Chart assignment"
        "\n"
        r"$k=k(M,\eta)$",
        shade=0.92,
    )

    add_box(
        ax,
        5.25,
        2.0,
        2.1,
        1.7,
        "Local modal network"
        "\n"
        r"$\mathcal{N}^{(k)}_{m,\theta}$",
        shade=0.90,
    )

    add_box(
        ax,
        8.05,
        2.2,
        1.55,
        1.25,
        "Modal"
        "\n"
        "prediction",
        shade=0.98,
    )

    # Physics loss
    add_box(
        ax,
        7.8,
        5.0,
        1.9,
        1.55,
        "Governing equation,"
        "\n"
        "boundary and"
        "\n"
        "normalization losses",
        shade=0.88,
        fontsize=9.2,
    )

    # Connections
    add_arrow(ax, (1.9, 7.45), (2.65, 7.45))
    add_arrow(ax, (4.65, 7.45), (5.45, 7.45))

    add_arrow(ax, (1.9, 2.8), (2.65, 2.8))
    add_arrow(ax, (4.65, 2.85), (5.25, 2.85))
    add_arrow(ax, (7.35, 2.85), (8.05, 2.85))

    add_arrow(ax, (7.0, 7.45), (8.15, 6.45))
    add_arrow(ax, (8.8, 3.45), (8.8, 5.0))

    ax.text(
        6.65,
        5.5,
        "physical"
        "\n"
        "coupling",
        fontsize=8.5,
        ha="center",
        va="center",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.5),
        constrained_layout=True,
    )

    draw_local_architecture(axes[0])
    draw_atlas_architecture(axes[1])

    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Written: {png_path}")
    print(f"Written: {pdf_path}")


if __name__ == "__main__":
    main()
