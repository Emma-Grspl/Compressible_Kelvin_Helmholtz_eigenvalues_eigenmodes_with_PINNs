from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import pandas as pd


REPO = Path.cwd()

DATA_ROOT = (
    REPO
    / "assets/pinn_supersonic/datasets/atlas2d_v1"
)

OUT = (
    REPO
    / "assets/p2-supersonic-pinn-method"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Frozen atlas geometry
# =============================================================================

M_BANDS = {
    "M0": (1.00, 1.25),
    "M1": (1.15, 1.45),
    "M2": (1.35, 1.65),
    "M3": (1.55, 1.90),
}

A_BANDS = {
    "A0": (0.05, 0.13),
    "A1": (0.10, 0.22),
    "A2": (0.19, 0.36),
}

CHARTS = {}

for i, (mname, (m0, m1)) in enumerate(
    M_BANDS.items()
):
    for j, (aname, (a0, a1)) in enumerate(
        A_BANDS.items()
    ):
        CHARTS[f"C{i}{j}"] = {
            "m0": m0,
            "m1": m1,
            "a0": a0,
            "a1": a1,
        }


# =============================================================================
# Small drawing helpers
# =============================================================================

def add_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    fontsize=10,
    linewidth=1.3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02",
        fill=False,
        linewidth=linewidth,
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )

    return patch


def arrow(
    ax,
    x0,
    y0,
    x1,
    y1,
    *,
    mutation_scale=14,
):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="->",
            mutation_scale=mutation_scale,
            linewidth=1.2,
        )
    )


# =============================================================================
# FIGURE 1
# Spectral-modal PINN + corrected inference pipeline
# =============================================================================

fig, ax = plt.subplots(
    figsize=(12.0, 7.4)
)

ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")


# -------------------------------------------------------------------------
# Training representation
# -------------------------------------------------------------------------

ax.text(
    0.4,
    7.55,
    "(a) Physics-informed spectral-modal representation",
    fontsize=13,
    fontweight="bold",
)

add_box(
    ax,
    0.5,
    5.65,
    1.65,
    0.9,
    r"Spectral inputs"
    "\n"
    r"$(M,\alpha)$",
)

add_box(
    ax,
    3.0,
    5.55,
    2.0,
    1.1,
    "Spectral network",
)

add_box(
    ax,
    5.85,
    5.55,
    1.8,
    1.1,
    r"$c_r(M,\alpha)$"
    "\n"
    r"$c_i(M,\alpha)$",
)

arrow(ax, 2.15, 6.1, 3.0, 6.1)
arrow(ax, 5.0, 6.1, 5.85, 6.1)


add_box(
    ax,
    0.5,
    3.7,
    1.65,
    1.0,
    r"Modal inputs"
    "\n"
    r"$(\xi,M,\alpha)$",
)

add_box(
    ax,
    3.0,
    3.65,
    2.0,
    1.1,
    "Modal network",
)

add_box(
    ax,
    5.85,
    3.5,
    1.8,
    1.4,
    r"$\kappa(\xi,M,\alpha)$"
    "\n"
    r"$q(\xi,M,\alpha)$"
    "\n"
    r"$\log |\hat p|$",
)

arrow(ax, 2.15, 4.2, 3.0, 4.2)
arrow(ax, 5.0, 4.2, 5.85, 4.2)


add_box(
    ax,
    8.35,
    5.45,
    2.7,
    1.3,
    "Sparse classical"
    "\n"
    r"spectral anchors $A_N$"
    "\n"
    r"$\mathcal{L}_{\mathrm{spec}}$",
)

arrow(
    ax,
    8.35,
    6.1,
    7.65,
    6.1,
)


add_box(
    ax,
    8.35,
    3.45,
    2.7,
    1.5,
    "Continuous-parameter"
    "\n"
    r"physics sampling in $(M,\alpha)$"
    "\n"
    r"Riccati/PDE + BC losses",
)

arrow(
    ax,
    8.35,
    4.2,
    7.65,
    4.2,
)

arrow(
    ax,
    6.75,
    5.55,
    6.75,
    4.9,
)


ax.text(
    0.55,
    2.85,
    "The modal network is used as a joint physics-informed representation "
    "during training; the final corrected eigenmode is reconstructed by "
    "the classical shooting solution.",
    fontsize=9.5,
)


# -------------------------------------------------------------------------
# Online corrected inference
# -------------------------------------------------------------------------

ax.text(
    0.4,
    2.15,
    "(b) Corrected inference",
    fontsize=13,
    fontweight="bold",
)

add_box(
    ax,
    0.5,
    0.65,
    1.7,
    0.9,
    r"Query $(M,\alpha)$",
)

add_box(
    ax,
    2.9,
    0.55,
    2.1,
    1.1,
    "Deterministic"
    "\n"
    "primary chart",
)

add_box(
    ax,
    5.7,
    0.55,
    2.0,
    1.1,
    r"PINN seed"
    "\n"
    r"$(c_r^{(0)},c_i^{(0)})$",
)

add_box(
    ax,
    8.35,
    0.45,
    2.4,
    1.3,
    "Local complex"
    "\n"
    "Riccati shooting",
)

add_box(
    ax,
    11.15,
    0.3,
    0.75,
    1.6,
    r"$c_r,c_i$"
    "\n"
    r"$\hat p(y)$",
    fontsize=9,
)

arrow(ax, 2.2, 1.1, 2.9, 1.1)
arrow(ax, 5.0, 1.1, 5.7, 1.1)
arrow(ax, 7.7, 1.1, 8.35, 1.1)
arrow(ax, 10.75, 1.1, 11.15, 1.1)

fig.savefig(
    OUT / "Fig_supersonic_pinn_hybrid_architecture.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT / "Fig_supersonic_pinn_hybrid_architecture.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# FIGURE 2
# Atlas geometry + nested anchor budgets
# =============================================================================

budgets = [76, 60, 48, 36, 24]

anchor_frames = {}

for budget in budgets:

    p = DATA_ROOT / f"anchors_N{budget}.csv"

    if not p.is_file():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)

    required = {"Mach", "alpha"}

    if not required.issubset(df.columns):
        raise RuntimeError(
            f"{p}: missing {required - set(df.columns)}"
        )

    anchor_frames[budget] = df


fig, axes = plt.subplots(
    2,
    3,
    figsize=(13.2, 8.3),
    sharex=True,
    sharey=True,
)

axes = axes.ravel()


# Panel a: atlas
ax = axes[0]

for chart, b in CHARTS.items():

    rect = Rectangle(
        (b["m0"], b["a0"]),
        b["m1"] - b["m0"],
        b["a1"] - b["a0"],
        fill=False,
        linewidth=1.2,
    )

    ax.add_patch(rect)

    ax.text(
        0.5 * (b["m0"] + b["m1"]),
        0.5 * (b["a0"] + b["a1"]),
        chart,
        ha="center",
        va="center",
        fontsize=8,
    )

ax.set_title("(a) 12-chart overlapping atlas")


# Panels b-f: budgets
for panel, budget in enumerate(
    budgets,
    start=1,
):

    ax = axes[panel]

    # chart boundaries
    for _, b in CHARTS.items():

        rect = Rectangle(
            (b["m0"], b["a0"]),
            b["m1"] - b["m0"],
            b["a1"] - b["a0"],
            fill=False,
            linewidth=0.55,
            alpha=0.5,
        )

        ax.add_patch(rect)

    df = anchor_frames[budget]

    ax.scatter(
        df["Mach"],
        df["alpha"],
        s=18,
    )

    ax.set_title(
        f"({chr(ord('a') + panel)}) "
        f"$N={budget}$ — {len(df)} anchors"
    )


for ax in axes:

    ax.set_xlim(0.985, 1.915)
    ax.set_ylim(0.04, 0.37)

    ax.grid(
        True,
        linewidth=0.4,
        alpha=0.3,
    )


for ax in axes[3:]:
    ax.set_xlabel(r"Mach number $M$")

for ax in axes[::3]:
    ax.set_ylabel(r"Wavenumber $\alpha$")


fig.suptitle(
    "Supersonic local atlas and nested sparse spectral-anchor budgets",
    fontsize=14,
)

fig.tight_layout()

fig.savefig(
    OUT / "Fig_supersonic_atlas_and_anchor_budgets.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT / "Fig_supersonic_atlas_and_anchor_budgets.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# FIGURE 3
# Validation / selection / final-test protocol
# =============================================================================

fig, ax = plt.subplots(
    figsize=(13, 6.5)
)

ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis("off")


ax.text(
    0.45,
    6.55,
    "Frozen model-selection and evaluation protocol",
    fontsize=14,
    fontweight="bold",
)


# Training
add_box(
    ax,
    0.5,
    4.6,
    2.0,
    1.1,
    r"Nested anchors"
    "\n"
    r"$A_{24}\subset\cdots\subset A_{76}$",
)

add_box(
    ax,
    3.2,
    4.6,
    2.0,
    1.1,
    "Independent training"
    "\n"
    "for each budget",
)

add_box(
    ax,
    5.9,
    4.6,
    1.8,
    1.1,
    r"V64"
    "\n"
    "validation",
)

add_box(
    ax,
    8.4,
    4.35,
    2.6,
    1.6,
    "PINN-seeded shooting"
    "\n"
    "branch recovery"
    "\n"
    r"$\varepsilon_c\leq10^{-4}$",
)

add_box(
    ax,
    11.55,
    4.55,
    1.1,
    1.2,
    r"Select"
    "\n"
    r"$N^\star$",
)

arrow(ax, 2.5, 5.15, 3.2, 5.15)
arrow(ax, 5.2, 5.15, 5.9, 5.15)
arrow(ax, 7.7, 5.15, 8.4, 5.15)
arrow(ax, 11.0, 5.15, 11.55, 5.15)


# Final test
add_box(
    ax,
    4.55,
    2.05,
    2.2,
    1.2,
    r"Frozen $N^\star$"
    "\n"
    "no further tuning",
)

add_box(
    ax,
    7.6,
    2.05,
    2.0,
    1.2,
    r"Sealed T401"
    "\n"
    "final accuracy",
)

add_box(
    ax,
    10.45,
    2.05,
    2.0,
    1.2,
    "Final article"
    "\n"
    "performance",
)

arrow(ax, 12.1, 4.55, 6.3, 3.25)
arrow(ax, 6.75, 2.65, 7.6, 2.65)
arrow(ax, 9.6, 2.65, 10.45, 2.65)


# Cost benchmark
add_box(
    ax,
    7.6,
    0.3,
    2.0,
    1.0,
    "COST500"
    "\n"
    "runtime grid",
)

add_box(
    ax,
    10.45,
    0.3,
    2.0,
    1.0,
    "Classical vs"
    "\n"
    "hybrid cost",
)

arrow(ax, 6.75, 2.25, 7.6, 0.8)
arrow(ax, 9.6, 0.8, 10.45, 0.8)


ax.text(
    0.55,
    0.45,
    "V64 is used for budget selection only. "
    "T401 is not inspected before the budget is frozen. "
    "COST500 is reserved for computational-performance comparisons.",
    fontsize=9.5,
)


fig.savefig(
    OUT / "Fig_supersonic_validation_protocol.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT / "Fig_supersonic_validation_protocol.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# TABLE
# Only protocol items already frozen and independent of final results
# =============================================================================

table = r"""
\begin{table}[!h]
\centering
\begin{tabular}{ll}
\hline
Item & Supersonic PINN atlas configuration \\
\hline
Parameter domain
&
$1.00\leq M\leq1.90$, $0.05\leq\alpha\leq0.36$
\\

Atlas
&
$4$ overlapping Mach bands $\times$ $3$ overlapping wavenumber bands
\\

Number of charts
&
$12$
\\

Mach bands
&
$[1.00,1.25]$, $[1.15,1.45]$, $[1.35,1.65]$, $[1.55,1.90]$
\\

Wavenumber bands
&
$[0.05,0.13]$, $[0.10,0.22]$, $[0.19,0.36]$
\\

Spectral outputs
&
$(c_r,c_i)$
\\

Modal outputs
&
$(\kappa,q,\log|\hat p|)$
\\

Spectral supervision
&
Sparse classical eigenvalue anchors
\\

Modal supervision
&
None
\\

Physics sampling
&
Continuous in Mach number and over the full local chart rectangle
\\

Interior samples per optimization step
&
$1024$
\\

Boundary samples per optimization step
&
$96$
\\

Spectral prefit stage
&
$2000$ steps
\\

Frozen-spectrum modal stage
&
$4000$ steps
\\

Joint stage
&
$4000$ steps
\\

Anchor budgets
&
$N\in\{76,60,48,36,24\}$
\\

Budget relation
&
$A_{24}\subset A_{36}\subset A_{48}\subset A_{60}\subset A_{76}$
\\

Initialization
&
Identical chart-specific deterministic seed across budgets
\\

Warm start between budgets
&
None
\\

Validation set
&
$64$ frozen points
\\

Final test set
&
$401$ sealed points
\\

Runtime benchmark
&
$500$ fixed query points
\\
\hline
\end{tabular}
\caption{
Frozen methodological configuration of the supersonic physics-informed
spectral-modal atlas. Numerical architecture and optimizer parameters
shared with the baseline implementation are reported separately where
appropriate.
}
\label{tab:supersonic_pinn_protocol}
\end{table}
""".strip()

(
    OUT
    / "Tab_supersonic_training_protocol.tex"
).write_text(
    table + "\n"
)


# =============================================================================
# README
# =============================================================================

readme = """
METHOD ASSETS — SUPERSONIC PINN
===============================

Fig_supersonic_pinn_hybrid_architecture.pdf
    Spectral-modal training representation and corrected PINN-seeded
    shooting inference pipeline.

Fig_supersonic_atlas_and_anchor_budgets.pdf
    Frozen 12-chart atlas and nested anchor budgets
    N = 76, 60, 48, 36, 24.

Fig_supersonic_validation_protocol.pdf
    Frozen V64 model-selection protocol, sealed T401 final test,
    and independent COST500 runtime benchmark.

Tab_supersonic_training_protocol.tex
    Result-independent methodological parameters.

These assets do not depend on the outcome of the lower-budget
shooting benchmarks and can therefore be frozen before N* is selected.
""".strip()

(
    OUT
    / "README_method_assets.txt"
).write_text(
    readme + "\n"
)


print("=" * 90)
print("SUPERSONIC PINN METHOD ASSETS")
print("=" * 90)

for p in sorted(OUT.iterdir()):
    print(p)

print()
print("DONE")
