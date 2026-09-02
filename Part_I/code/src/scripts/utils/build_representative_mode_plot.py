from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]

DATA = (
    ROOT
    / "assets/pinn_subsonic/csv/article/N340/"
    "Table_Data_representative_mode_M05_a05_N340.csv"
)

OUTDIR = (
    ROOT
    / "assets/pinn_subsonic/article/N340"
)


def main():
    df = pd.read_csv(DATA)

    y = df["y"].to_numpy(float)

    fields = [
        ("p", r"\hat{p}"),
        ("rho", r"\hat{\rho}"),
        ("u", r"\hat{u}"),
        ("v", r"\hat{v}"),
    ]

    styles = {
        "classic": {
            "label": "Classical",
            "color": "tab:blue",
            "linestyle": "-",
            "linewidth": 2.0,
        },
        "direct": {
            "label": "Direct PINN",
            "color": "tab:orange",
            "linestyle": "--",
            "linewidth": 1.8,
        },
        "gep": {
            "label": "PINN-seeded GEP",
            "color": "tab:green",
            "linestyle": ":",
            "linewidth": 2.3,
        },
    }

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(10.5, 13.5),
        sharex=True,
    )

    for row, (field, latex_field) in enumerate(fields):
        for col, component in enumerate(["real", "imag"]):
            ax = axes[row, col]

            for method in ["classic", "direct", "gep"]:
                column = f"{method}_{field}_{component}"

                ax.plot(
                    y,
                    df[column].to_numpy(float),
                    label=styles[method]["label"],
                    color=styles[method]["color"],
                    linestyle=styles[method]["linestyle"],
                    linewidth=styles[method]["linewidth"],
                )

            prefix = "Re" if component == "real" else "Im"

            ax.set_title(
                rf"${prefix}({latex_field})$"
            )

            ax.grid(
                True,
                alpha=0.22,
            )

            if row == 3:
                ax.set_xlabel(r"$y$")

    axes[0, 0].legend(
        loc="upper right",
        framealpha=0.95,
    )

    # Main title
    fig.suptitle(
        r"Eigenmode reconstruction at $M=0.5,\ \alpha=0.5$",
        fontsize=14,
        y=0.992,
    )

    # Plain-text subtitle: no broken LaTeX rendering.
    fig.text(
        0.5,
        0.968,
        (
            "c_i(classical) = 0.267326"
            "    |    "
            "c_i(PINN) = 0.267282"
            "    |    "
            "c_i(GEP) = 0.267384"
        ),
        ha="center",
        va="top",
        fontsize=11,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.945]
    )

    png = (
        OUTDIR
        / "Fig_representative_mode_M05_a05_N340.png"
    )

    pdf = (
        OUTDIR
        / "Fig_representative_mode_M05_a05_N340.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Wrote:", png)
    print("Wrote:", pdf)


if __name__ == "__main__":
    main()
