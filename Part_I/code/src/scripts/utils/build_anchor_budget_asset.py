from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]

TABLE = (
    ROOT
    / "assets/pinn_subsonic/csv/article/N340/"
    "Table_anchor_budget_comparison.csv"
)

OUTDIR = (
    ROOT
    / "assets/pinn_subsonic/article/N340"
)

NSTAR = 340


def get_column(df, *names):
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(
        f"None of {names} found. "
        f"Available columns: {df.columns.tolist()}"
    )


def main():
    df = pd.read_csv(TABLE).copy()

    col_N = get_column(df, "N", "budget")
    col_pinn = get_column(df, "pinn_p95", "ci_pinn_p95")
    col_gep = get_column(df, "gep_p95", "ci_gep_p95")
    col_frac = get_column(
        df,
        "fraction_gep_lt_1pct",
        "fraction_joint_below_1pct",
        "fraction_ci_below_1pct",
    )
    col_wrong = get_column(df, "n_wrong_branch", "n_oracle_mismatch")
    col_cat = get_column(df, "n_gep_gt_10pct", "n_ci_catastrophic_10pct")

    df = df.sort_values(col_N).reset_index(drop=True)

    budgets = df[col_N].to_numpy(int)
    pinn_p95 = df[col_pinn].to_numpy(float)
    gep_p95 = df[col_gep].to_numpy(float)
    frac = df[col_frac].to_numpy(float)
    wrong = df[col_wrong].fillna(0).to_numpy(int)
    catastrophic = df[col_cat].fillna(0).to_numpy(int)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
    })

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.2),
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.93,
        bottom=0.14,
        top=0.83,
        wspace=0.20,
    )

    # ============================================================
    # LEFT — DIRECT PINN
    # ============================================================

    ax1.plot(
        budgets,
        pinn_p95,
        "-o",
        color="tab:blue",
        linewidth=2.2,
        markersize=7,
        label="Direct PINN p95",
    )

    ax1.axvline(
        NSTAR,
        color="black",
        linestyle="--",
        linewidth=1.7,
        alpha=0.9,
    )

    ax1.set_yscale("log")
    ax1.set_xlabel(r"Global anchor budget $N$")
    ax1.set_ylabel(r"Scaled $c_i$ error")
    ax1.set_title("Direct PINN spectral accuracy")
    ax1.grid(True, which="both", alpha=0.22)
    ax1.legend(loc="upper right", framealpha=0.95)

    ax1.annotate(
        r"$N^\star=340$",
        xy=(NSTAR, ax1.get_ylim()[1]),
        xycoords="data",
        xytext=(8, -8),
        textcoords="offset points",
        ha="left",
        va="top",
        color="black",
        fontsize=10,
    )

    # ============================================================
    # RIGHT — GEP
    # ============================================================

    gep_line = ax2.plot(
        budgets,
        gep_p95,
        "-o",
        color="tab:blue",
        linewidth=2.2,
        markersize=7,
        label="PINN-seeded GEP p95",
    )

    ax2.axvline(
        NSTAR,
        color="black",
        linestyle="--",
        linewidth=1.7,
        alpha=0.9,
    )

    ax2.set_xlabel(r"Global anchor budget $N$")
    ax2.set_ylabel(r"Scaled $c_i$ error", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("GEP refinement and branch robustness")
    ax2.grid(True, which="both", alpha=0.22)

    ax2b = ax2.twinx()

    frac_line = ax2b.plot(
        budgets,
        frac,
        "--s",
        color="tab:orange",
        linewidth=2.2,
        markersize=7,
        label="Fraction <1%",
    )

    ax2b.set_ylabel(
        "Fraction with GEP error <1%",
        color="tab:orange",
    )
    ax2b.tick_params(axis="y", labelcolor="tab:orange")

    lower = min(0.90, float(np.min(frac)) - 0.01)
    ax2b.set_ylim(lower, 1.005)

    # ------------------------------------------------------------
    # Readable annotations: below for high points, side for the last.
    # ------------------------------------------------------------
    offsets = {
        96:  (0, -34),   # below
        224: (0, -34),   # below
        340: (0, 12),    # above
        520: (0, 12),    # above
        640: (0, 12),    # above
        705: (-40, 0),   # to the left
    }

    alignments = {
        96: "center",
        224: "center",
        340: "center",
        520: "center",
        640: "center",
        705: "right",
    }

    for x, y, nw, nc in zip(budgets, gep_p95, wrong, catastrophic):
        dx, dy = offsets.get(int(x), (0, 12))
        ha = alignments.get(int(x), "center")
        va = "top" if dy < 0 else "bottom"

        if nw == 0 and nc == 0:
            text = "0 wrong"
        else:
            text = f"{nw} wrong\n{nc} >10%"

        ax2.annotate(
            text,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=9,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="0.75",
                linewidth=0.5,
                alpha=0.90,
            ),
        )

    ax2.annotate(
        r"$N^\star=340$",
        xy=(NSTAR, ax2.get_ylim()[1]),
        xytext=(7, -8),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10,
        color="black",
    )

    handles = gep_line + frac_line
    labels = [h.get_label() for h in handles]

    ax2.legend(
        handles,
        labels,
        loc="upper right",
        framealpha=0.95,
    )

    fig.suptitle(
        r"Global classical-information budget ablation "
        r"($N^\star=340$)",
        fontsize=18,
        y=0.965,
    )

    png = OUTDIR / "Fig_anchor_budget_comparison.png"
    pdf = OUTDIR / "Fig_anchor_budget_comparison.pdf"

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
