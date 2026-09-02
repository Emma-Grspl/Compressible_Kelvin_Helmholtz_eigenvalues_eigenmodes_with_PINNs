from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]

DATA = (
    ROOT
    / "assets/pinn_subsonic/csv/article_work/gep_selection_N340/"
    "Table_pointwise.csv"
)

OUTDIR = (
    ROOT
    / "assets/pinn_subsonic/article/N340"
)


def get_column(df, *names):
    for name in names:
        if name in df.columns:
            return name

    raise KeyError(
        f"None of {names} found.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


def main():
    df = pd.read_csv(DATA).copy()

    col_M = get_column(
        df,
        "Mach",
        "M",
    )

    col_eta = get_column(
        df,
        "eta",
        "Eta",
    )

    col_classic = get_column(
        df,
        "ci_ref",
        "classical_ci",
        "ci_classic",
    )

    col_pinn = get_column(
        df,
        "ci_pinn",
        "pinn_ci",
        "direct_pinn_ci",
    )

    col_gep = get_column(
        df,
        "pinn_seeded_ci",
        "scalar_gep_ci",
        "gep_ci",
        "hybrid_ci",
    )

    # ------------------------------------------------------------
    # Important fix:
    # first restrict to the near-neutral data,
    # THEN determine the nearest available Mach slice.
    # ------------------------------------------------------------

    near = df.loc[
        df[col_eta].astype(float) >= 0.775
    ].copy()

    if near.empty:
        raise RuntimeError(
            "No points found with eta >= 0.775."
        )

    available_M = np.sort(
        near[col_M]
        .astype(float)
        .unique()
    )

    targets = [
        0.10,
        0.50,
        0.70,
    ]

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.5, 4.8),
        sharey=True,
    )

    print("Available near-neutral Mach slices:")
    print(available_M)
    print()

    for ax, target in zip(
        axes,
        targets,
    ):
        # Nearest Mach that ACTUALLY contains
        # near-neutral points.
        actual_M = float(
            available_M[
                np.argmin(
                    np.abs(
                        available_M - target
                    )
                )
            ]
        )

        sub = near.loc[
            np.isclose(
                near[col_M].astype(float),
                actual_M,
                rtol=0.0,
                atol=1.0e-12,
            )
        ].copy()

        sub = (
            sub.sort_values(col_eta)
            .drop_duplicates(
                subset=[col_eta],
                keep="first",
            )
        )

        if len(sub) < 2:
            raise RuntimeError(
                f"Only {len(sub)} point(s) for "
                f"target M={target:.2f}, "
                f"nearest M={actual_M:.8f}."
            )

        eta = sub[col_eta].to_numpy(float)

        ci_classic = (
            sub[col_classic]
            .to_numpy(float)
        )

        ci_pinn = (
            sub[col_pinn]
            .to_numpy(float)
        )

        ci_gep = (
            sub[col_gep]
            .to_numpy(float)
        )

        ax.plot(
            eta,
            ci_classic,
            color="tab:blue",
            linewidth=2.2,
            label="Classical",
        )

        ax.plot(
            eta,
            ci_pinn,
            color="tab:orange",
            linewidth=2.2,
            label="Direct PINN",
        )

        ax.plot(
            eta,
            ci_gep,
            color="tab:green",
            linewidth=2.2,
            label="PINN-seeded GEP",
        )

        # If exact target exists, keep clean title M=0.50.
        # Otherwise show the actual slice used.
        if abs(actual_M - target) < 1.0e-8:
            title = rf"$M={target:.2f}$"
        else:
            title = (
                rf"$M\simeq{actual_M:.3f}$"
            )

        ax.set_title(title)

        ax.set_xlabel(r"$\eta$")

        ax.set_xlim(
            max(
                0.765,
                eta.min() - 0.01,
            ),
            min(
                0.99,
                eta.max() + 0.01,
            ),
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        print(
            f"target M={target:.2f} -> "
            f"used M={actual_M:.8f} | "
            f"n={len(sub)} | "
            f"eta=[{eta.min():.4f}, "
            f"{eta.max():.4f}]"
        )

    axes[0].set_ylabel(r"$c_i$")

    axes[-1].legend(
        loc="upper right",
        framealpha=0.95,
    )

    fig.tight_layout()

    png = (
        OUTDIR
        / "Fig_near_neutral_growth_rate_cuts_N340.png"
    )

    pdf = (
        OUTDIR
        / "Fig_near_neutral_growth_rate_cuts_N340.pdf"
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

    print()
    print("Columns used:")
    print("  Mach      :", col_M)
    print("  eta       :", col_eta)
    print("  classical :", col_classic)
    print("  PINN      :", col_pinn)
    print("  GEP       :", col_gep)
    print()
    print("Wrote:", png)
    print("Wrote:", pdf)


if __name__ == "__main__":
    main()
