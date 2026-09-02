#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

OUT = ROOT / "assets/pinn_subsonic/article/N340"
OUT.mkdir(parents=True, exist_ok=True)

RUN = ROOT / "assets/pinn_subsonic/anchor_budget_runs/N340"

POINTWISE = (
    RUN
    / "summary"
    / "validation_pointwise_canonical.csv"
)

ANCHORS = RUN / "anchors.csv"

PLAN = (
    ROOT
    / "archive/csv/assets/pinn_subsonic/"
    "joint_ci_mode_atlas_v2/training_plan.tsv"
)

RUNTIME_DIR = (
    ROOT
    / "assets/pinn_subsonic/article_work/"
    "runtime_N340"
)

RUNTIME_SUMMARY = (
    RUNTIME_DIR
    / "Table_runtime_summary.csv"
)

MAP1000 = (
    RUNTIME_DIR
    / "Table_map_1000_ci_checks.csv"
)

SELECTION_SUMMARY = (
    ROOT
    / "assets/pinn_subsonic/article_work/"
    "gep_selection_N340/"
    "Table_summary.csv"
)

BUDGET_ROOT = (
    ROOT
    / "assets/pinn_subsonic/"
    "anchor_budget_runs"
)


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 140,
    "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(
        OUT / f"{name}.png",
        bbox_inches="tight",
        dpi=300,
    )
    fig.savefig(
        OUT / f"{name}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Wrote:", OUT / f"{name}.png")


def alpha_from_eta(M, eta):
    return (
        np.asarray(eta, dtype=float)
        * np.sqrt(
            np.clip(
                1.0
                - np.asarray(M, dtype=float) ** 2,
                0.0,
                None,
            )
        )
    )


# =====================================================================
# 1. ATLAS GEOMETRY
# =====================================================================

def plot_atlas():
    plan = pd.read_csv(PLAN, sep="\t").copy()

    fig, ax = plt.subplots(
        figsize=(11.5, 8.5)
    )

    palette = plt.cm.tab20(
        np.linspace(0, 1, 20)
    )

    for i, row in plan.reset_index(drop=True).iterrows():
        m0 = float(row["mach_min"])
        m1 = float(row["mach_max"])
        e0 = float(row["eta_min"])
        e1 = float(row["eta_max"])

        mm = np.linspace(m0, m1, 160)

        a0 = alpha_from_eta(mm, e0)
        a1 = alpha_from_eta(mm, e1)

        color = palette[i % len(palette)]

        ax.plot(mm, a0, lw=0.75, color=color)
        ax.plot(mm, a1, lw=0.75, color=color)

        ax.plot(
            [m0, m0],
            [
                alpha_from_eta(m0, e0),
                alpha_from_eta(m0, e1),
            ],
            lw=0.75,
            color=color,
        )

        ax.plot(
            [m1, m1],
            [
                alpha_from_eta(m1, e0),
                alpha_from_eta(m1, e1),
            ],
            lw=0.75,
            color=color,
        )

        mc = 0.5 * (m0 + m1)
        ec = 0.5 * (e0 + e1)
        ac = alpha_from_eta(mc, ec)

        ax.text(
            mc,
            ac,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=6,
        )

    M = np.linspace(0.0, 1.0, 500)
    boundary = np.sqrt(
        np.clip(1.0 - M**2, 0.0, None)
    )

    ax.plot(
        M,
        boundary,
        "k--",
        lw=1.5,
        label=r"$\alpha^2+M^2=1$",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel(r"Mach number $M$")
    ax.set_ylabel(r"Wavenumber $\alpha$")

    ax.set_title(
        "Joint PINN atlas: 49 local charts"
    )

    ax.grid(alpha=0.20)
    ax.legend(loc="upper right")

    save(
        fig,
        "Fig_atlas_49_charts_Mach_alpha",
    )


# =====================================================================
# 2. N340 ANCHOR MAP
# =====================================================================

def plot_anchor_map():
    anchors = pd.read_csv(ANCHORS).copy()

    if "Mach" not in anchors:
        if "M" in anchors:
            anchors = anchors.rename(
                columns={"M": "Mach"}
            )

    if "alpha" not in anchors:
        anchors["alpha"] = alpha_from_eta(
            anchors["Mach"],
            anchors["eta"],
        )

    # Count chart-local rows actually used by N340.
    chart_files = sorted(
        (RUN / "joint").glob(
            "*/ci_anchor_points.csv"
        )
    )

    chart_rows = 0

    for path in chart_files:
        chart_rows += len(pd.read_csv(path))

    fig, ax = plt.subplots(
        figsize=(11.5, 8.5)
    )

    # faint chart boundaries
    plan = pd.read_csv(PLAN, sep="\t")

    for _, row in plan.iterrows():
        m0 = float(row["mach_min"])
        m1 = float(row["mach_max"])
        e0 = float(row["eta_min"])
        e1 = float(row["eta_max"])

        mm = np.linspace(m0, m1, 100)

        ax.plot(
            mm,
            alpha_from_eta(mm, e0),
            lw=0.35,
            alpha=0.22,
        )

        ax.plot(
            mm,
            alpha_from_eta(mm, e1),
            lw=0.35,
            alpha=0.22,
        )

    ax.scatter(
        anchors["Mach"],
        anchors["alpha"],
        s=24,
        edgecolors="black",
        linewidths=0.5,
        label=(
            f"Global physical anchors: "
            f"{len(anchors)}"
        ),
    )

    M = np.linspace(0.0, 1.0, 500)

    ax.plot(
        M,
        np.sqrt(
            np.clip(1.0 - M**2, 0, None)
        ),
        "k--",
        lw=1.4,
        label=r"$\alpha^2+M^2=1$",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel(r"Mach number $M$")
    ax.set_ylabel(r"Wavenumber $\alpha$")

    ax.set_title(
        r"Sparse spectral supervision: "
        r"$N^\star=340$ global $c_i$ anchors"
    )

    text = (
        f"Unique physical locations: "
        f"{len(anchors)}\n"
        f"Chart-local usages: "
        f"{chart_rows}"
    )

    ax.text(
        0.98,
        0.90,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
        ),
    )

    ax.grid(alpha=0.18)
    ax.legend(loc="upper right")

    save(
        fig,
        "Fig_sparse_ci_anchor_map_Mach_alpha_N340",
    )


# =====================================================================
# 3. 1000-POINT SPECTRAL ERROR MAP
# =====================================================================

def plot_heatmap():
    df = pd.read_csv(MAP1000).copy()

    ci_hybrid = df["hybrid_ci"].to_numpy(float)
    ci_classic = df["classical_ci"].to_numpy(float)

    # Scaled spectral error in percent.
    err_scaled_pct = (
        100.0
        * np.abs(ci_hybrid - ci_classic)
        / np.maximum(
            np.abs(ci_classic),
            0.05,
        )
    )

    M = df["Mach"].to_numpy(float)
    alpha = df["alpha"].to_numpy(float)

    tri = mtri.Triangulation(M, alpha)

    fig, ax = plt.subplots(
        figsize=(10.5, 8.0)
    )

    # Linear colour scale.
    # Values above 5% are shown with the maximum colour.
    vmax = 5.0

    levels = np.linspace(
        0.0,
        vmax,
        31,
    )

    artist = ax.tricontourf(
        tri,
        err_scaled_pct,
        levels=levels,
        cmap="viridis",
        extend="max",
    )

    Mline = np.linspace(
        0.0,
        1.0,
        500,
    )

    ax.plot(
        Mline,
        np.sqrt(
            np.clip(
                1.0 - Mline**2,
                0.0,
                None,
            )
        ),
        "k--",
        lw=1.3,
        label=r"$\alpha^2+M^2=1$",
    )

    cb = fig.colorbar(
        artist,
        ax=ax,
    )

    cb.set_label(
        r"Scaled spectral error "
        r"$100\,|c_i^{\rm PINN+GEP}"
        r"-c_i^{\rm classical}|"
        r"/\max(|c_i^{\rm classical}|,0.05)$ [%]"
    )

    ax.set_xlabel(
        r"Mach number $M$"
    )

    ax.set_ylabel(
        r"Wavenumber $\alpha$"
    )

    ax.set_title(
        "Spectral error: N340 PINN + GEP "
        "versus classical shooting"
    )

    ax.set_xlim(
        M.min() - 0.01,
        1.0,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.legend(
        loc="upper right"
    )

    print()
    print("=== MAP 1000 SCALED ERROR [%] ===")
    print("median:", np.median(err_scaled_pct))
    print("p95   :", np.quantile(err_scaled_pct, 0.95))
    print("p99   :", np.quantile(err_scaled_pct, 0.99))
    print("max   :", np.max(err_scaled_pct))
    print("> 1%  :", int((err_scaled_pct > 1.0).sum()))
    print("> 5%  :", int((err_scaled_pct > 5.0).sum()))
    print("> 10% :", int((err_scaled_pct > 10.0).sum()))

    save(
        fig,
        "Fig_ci_error_heatmap_PINN_GEP_vs_classical_N340",
    )

# =====================================================================
# 4. NEAR-NEUTRAL ERROR SCALING
# =====================================================================

def plot_neutral_scaling():
    df = pd.read_csv(POINTWISE).copy()

    subset = df[
        df["eta"] >= 0.775
    ].copy()

    subset["err_pinn"] = np.abs(
        subset["ci_pinn"]
        - subset["ci_ref"]
    )

    subset["err_gep"] = np.abs(
        subset["scalar_gep_ci"]
        - subset["ci_ref"]
    )

    rows = []

    for eta, g in subset.groupby("eta"):
        rows.append({
            "eta": eta,
            "distance": 1.0 - eta,
            "pinn_median":
                g["err_pinn"].median(),
            "pinn_p90":
                g["err_pinn"].quantile(0.90),
            "gep_median":
                g["err_gep"].median(),
            "gep_p90":
                g["err_gep"].quantile(0.90),
        })

    out = (
        pd.DataFrame(rows)
        .sort_values("distance")
    )

    fig, ax = plt.subplots(
        figsize=(9.0, 7.0)
    )

    ax.plot(
        out["distance"],
        out["pinn_median"],
        "-o",
        label="Direct PINN: median",
    )

    ax.plot(
        out["distance"],
        out["pinn_p90"],
        "--o",
        label="Direct PINN: 90th percentile",
    )

    ax.plot(
        out["distance"],
        out["gep_median"],
        "-s",
        label="PINN-seeded GEP: median",
    )

    ax.plot(
        out["distance"],
        out["gep_p90"],
        "--s",
        label="PINN-seeded GEP: 90th percentile",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        r"Distance to the neutral boundary, "
        r"$1-\eta$"
    )

    ax.set_ylabel(
        r"$|c_i-c_i^{\rm classical}|$"
    )

    ax.set_title(
        "Spectral error near the neutral boundary "
        "(N340)"
    )

    ax.grid(
        alpha=0.25,
        which="both",
    )

    ax.legend()

    save(
        fig,
        "Fig_near_neutral_error_scaling_N340",
    )


# =====================================================================
# 5. NEAR-NEUTRAL CUTS
# =====================================================================

def plot_neutral_cuts():
    df = pd.read_csv(POINTWISE).copy()

    targets = [0.10, 0.50, 0.70]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
        sharey=True,
    )

    for ax, target in zip(axes, targets):
        machs = np.sort(
            df["Mach"].unique()
        )

        M = float(
            machs[
                np.argmin(
                    np.abs(machs-target)
                )
            ]
        )

        g = df[
            np.isclose(
                df["Mach"],
                M,
                atol=1e-10,
            )
            & (df["eta"] >= 0.775)
        ].copy()

        g = g.sort_values("eta")

        ax.plot(
            g["eta"],
            g["ci_ref"],
            label="Classical",
        )

        ax.plot(
            g["eta"],
            g["ci_pinn"],
            label="Direct PINN",
        )

        ax.plot(
            g["eta"],
            g["scalar_gep_ci"],
            label="PINN-seeded GEP",
        )

        ax.set_title(
            rf"$M={M:.2f}$"
        )

        ax.set_xlabel(r"$\eta$")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(r"$c_i$")

    axes[-1].legend(
        loc="best"
    )

    save(
        fig,
        "Fig_near_neutral_growth_rate_cuts_N340",
    )


# =====================================================================
# 6. ANCHOR-BUDGET COMPARISON
# =====================================================================

def collect_budget_metrics():
    rows = []

    for d in sorted(
        BUDGET_ROOT.glob("N[0-9][0-9][0-9]")
    ):
        try:
            N = int(d.name[1:])
        except ValueError:
            continue

        p = (
            d
            / "summary"
            / "validation_pointwise_canonical.csv"
        )

        if not p.is_file():
            continue

        df = pd.read_csv(p).copy()

        required = {
            "ci_pinn_scaled_err",
            "ci_gep_scaled_err",
            "scalar_gep_cr",
        }

        if not required.issubset(df.columns):
            continue

        ep = df["ci_pinn_scaled_err"]
        eg = df["ci_gep_scaled_err"]

        rows.append({
            "N": N,

            "pinn_p50":
                ep.quantile(0.50),

            "pinn_p95":
                ep.quantile(0.95),

            "pinn_p99":
                ep.quantile(0.99),

            "pinn_max":
                ep.max(),

            "gep_p50":
                eg.quantile(0.50),

            "gep_p95":
                eg.quantile(0.95),

            "gep_p99":
                eg.quantile(0.99),

            "gep_max":
                eg.max(),

            "fraction_gep_lt_1pct":
                (eg < 0.01).mean(),

            "n_gep_gt_10pct":
                int((eg > 0.10).sum()),

            "n_wrong_branch":
                int(
                    (
                        np.abs(
                            df["scalar_gep_cr"]
                        )
                        > 1.0e-3
                    ).sum()
                ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("N")
        .reset_index(drop=True)
    )


def plot_budget_comparison():
    table = collect_budget_metrics()

    table.to_csv(
        OUT / "Table_anchor_budget_comparison.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.2),
    )

    ax = axes[0]

    ax.plot(
        table["N"],
        table["pinn_p95"],
        "-o",
        label="Direct PINN p95",
    )

    ax.set_yscale("log")

    ax.axvline(
        340,
        ls="--",
        lw=1.2,
    )

    ax.set_xlabel(
        r"Global anchor budget $N$"
    )

    ax.set_ylabel(
        r"Scaled $c_i$ error"
    )

    ax.set_title(
        "Direct PINN spectral accuracy"
    )

    ax.grid(
        alpha=0.25,
        which="both",
    )

    ax.legend()

    ax = axes[1]

    ax.plot(
        table["N"],
        table["gep_p95"],
        "-o",
        label="PINN-seeded GEP p95",
    )

    ax.set_yscale("log")

    ax.axvline(
        340,
        ls="--",
        lw=1.2,
    )

    ax.set_xlabel(
        r"Global anchor budget $N$"
    )

    ax.set_ylabel(
        r"Scaled $c_i$ error"
    )

    ax.set_title(
        "GEP refinement and branch robustness"
    )

    ax.grid(
        alpha=0.25,
        which="both",
    )

    ax2 = ax.twinx()

    ax2.plot(
        table["N"],
        table["fraction_gep_lt_1pct"],
        "--s",
        label="Fraction <1%",
    )

    ax2.set_ylim(0.90, 1.005)

    ax2.set_ylabel(
        "Fraction with GEP error <1%"
    )

    for _, row in table.iterrows():
        ax.annotate(
            (
                f"wrong={int(row['n_wrong_branch'])}\n"
                f">10%={int(row['n_gep_gt_10pct'])}"
            ),
            (
                row["N"],
                row["gep_p95"],
            ),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    lines1, labels1 = (
        ax.get_legend_handles_labels()
    )

    lines2, labels2 = (
        ax2.get_legend_handles_labels()
    )

    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
    )

    fig.suptitle(
        r"Global classical-information budget ablation "
        r"($N^\star=340$)"
    )

    save(
        fig,
        "Fig_anchor_budget_comparison",
    )


# =====================================================================
# 7. RUNTIME FIGURE
# =====================================================================

def plot_runtime():
    df = pd.read_csv(
        RUNTIME_SUMMARY
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.0),
    )

    specifications = [
        (
            "single_point",
            "One representative point",
        ),
        (
            "map_1000",
            "Map of 1000 points",
        ),
    ]

    for ax, (workload, title) in zip(
        axes,
        specifications,
    ):
        g = df[
            df["workload"] == workload
        ].copy()

        order = [
            "pinn_seeded_gep",
            "classical",
        ]

        g = (
            g.set_index("method")
            .loc[order]
            .reset_index()
        )

        values = (
            g["median_wall_time_s"]
            .to_numpy(float)
        )

        bars = ax.bar(
            [
                "PINN-seeded\nGEP",
                "Classical\nfull mode",
            ],
            values,
        )

        ax.set_yscale("log")
        ax.set_ylabel("Wall time [s]")
        ax.set_title(title)
        ax.grid(
            axis="y",
            alpha=0.25,
            which="both",
        )

        for bar, value in zip(
            bars,
            values,
        ):
            ax.text(
                bar.get_x()
                + bar.get_width()/2,
                value * 1.06,
                f"{value:.3f} s",
                ha="center",
                va="bottom",
            )

        speedup = (
            values[1] / values[0]
        )

        ax.text(
            0.5,
            0.93,
            f"speedup = {speedup:.3f}×",
            transform=ax.transAxes,
            ha="center",
            va="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
            ),
        )

    fig.suptitle(
        "Inference benchmark — N340 frozen pipeline"
    )

    save(
        fig,
        "Fig_inference_benchmark_N340",
    )


# =====================================================================
# 8. GEP SELECTION BASELINES
# =====================================================================

def plot_selection():
    if not SELECTION_SUMMARY.is_file():
        return

    df = pd.read_csv(
        SELECTION_SUMMARY
    )

    labels = {
        "pinn_seeded":
            "PINN-seeded",
        "naive_central_growth":
            "Central branch\n+ max growth",
        "naive_max_growth":
            "Global\nmax growth",
    }

    names = [
        labels.get(x, x)
        for x in df["method"]
    ]

    success = (
        df["oracle_match_fraction"]
        .to_numpy(float)
    )

    catastrophic = (
        df["n_ci_catastrophic_10pct"]
        .to_numpy(int)
    )

    fig, ax = plt.subplots(
        figsize=(8.0, 5.4)
    )

    bars = ax.bar(
        names,
        success,
    )

    ax.set_ylim(0.93, 1.005)

    ax.set_ylabel(
        "Correct eigenpair fraction"
    )

    ax.set_title(
        "GEP branch-selection comparison (N340)"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, s, ncat in zip(
        bars,
        success,
        catastrophic,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width()/2,
            s + 0.001,
            (
                f"{100*s:.2f}%\n"
                f">10% errors: {ncat}"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save(
        fig,
        "Fig_gep_selection_comparison_N340",
    )


def main():
    plot_atlas()
    plot_anchor_map()
    plot_heatmap()
    plot_neutral_scaling()
    plot_neutral_cuts()
    plot_budget_comparison()
    plot_runtime()
    plot_selection()

    print()
    print("=" * 72)
    print("CPU ARTICLE ASSETS COMPLETE")
    print("=" * 72)
    print("Output:", OUT)


if __name__ == "__main__":
    main()
