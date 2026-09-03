from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    ROOT
    / "assets"
    / "complementary"
    / "anchors_only"
    / "multiseed_physics_vs_anchors_per_seed.csv"
)

OUT_DIR = ROOT / "assets" / "complementary" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "Table_T2_multiseed_physics_vs_anchors.csv"
OUT_TEX = OUT_DIR / "Table_T2_multiseed_physics_vs_anchors.tex"


# =============================================================================
# Load and validate
# =============================================================================

if not INPUT_CSV.is_file():
    raise FileNotFoundError(
        f"{INPUT_CSV}\n"
        "Run build_multiseed_physics_vs_anchors_ablation.py first."
    )

df = pd.read_csv(INPUT_CSV)

required = [
    "condition",
    "seed",
    "direct_mean",
    "direct_p95",
    "technical_success",
    "target_recovery",
    "corrected_median",
    "corrected_p95",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

conditions = [
    "Physics-informed",
    "Anchors-only",
]

for condition in conditions:
    g = df[df["condition"] == condition]

    if len(g) != 3:
        raise RuntimeError(
            f"{condition}: expected 3 seeds, found {len(g)}."
        )

    seeds = sorted(g["seed"].astype(int).tolist())

    if seeds != [1, 2, 3]:
        raise RuntimeError(
            f"{condition}: expected seeds [1,2,3], found {seeds}."
        )


# =============================================================================
# Numerical checks
# =============================================================================

if not np.all(df["technical_success"].to_numpy(dtype=int) == 64):
    raise RuntimeError(
        "Expected 64/64 technical convergence for every realization."
    )

if not np.all(df["target_recovery"].to_numpy(dtype=int) == 63):
    raise RuntimeError(
        "Expected 63/63 target-branch recovery for every realization."
    )


# =============================================================================
# Build publication table source
# =============================================================================

rows = []

for condition in conditions:
    g = (
        df[df["condition"] == condition]
        .sort_values("seed")
        .copy()
    )

    for _, row in g.iterrows():
        rows.append(
            {
                "condition": condition,
                "seed": str(int(row["seed"])),
                "direct_mean": float(row["direct_mean"]),
                "direct_p95": float(row["direct_p95"]),
                "corrected_median": float(row["corrected_median"]),
                "corrected_p95": float(row["corrected_p95"]),
                "technical": f"{int(row['technical_success'])}/64",
                "target_recovery": f"{int(row['target_recovery'])}/63",
                "row_type": "seed",
            }
        )

    rows.append(
        {
            "condition": condition,
            "seed": "mean ± SD",
            "direct_mean": float(g["direct_mean"].mean()),
            "direct_p95": float(g["direct_p95"].mean()),
            "corrected_median": float(g["corrected_median"].mean()),
            "corrected_p95": float(g["corrected_p95"].mean()),
            "direct_mean_sd": float(g["direct_mean"].std(ddof=1)),
            "direct_p95_sd": float(g["direct_p95"].std(ddof=1)),
            "corrected_median_sd": float(
                g["corrected_median"].std(ddof=1)
            ),
            "corrected_p95_sd": float(
                g["corrected_p95"].std(ddof=1)
            ),
            "technical": f"{int(g['technical_success'].sum())}/192",
            "target_recovery": f"{int(g['target_recovery'].sum())}/189",
            "row_type": "aggregate",
        }
    )

table = pd.DataFrame(rows)

table.to_csv(
    OUT_CSV,
    index=False,
)


# =============================================================================
# Terminal audit
# =============================================================================

print("=" * 110)
print("T2 — MULTI-SEED PHYSICS-INFORMED VS ANCHORS-ONLY ABLATION")
print("=" * 110)

for condition in conditions:
    g = df[df["condition"] == condition]

    print(f"\n{condition}")

    for _, r in g.sort_values("seed").iterrows():
        print(
            f"  seed {int(r['seed'])}: "
            f"direct mean={r['direct_mean']:.6e}  "
            f"P95={r['direct_p95']:.6e}  "
            f"corr med={r['corrected_median']:.6e}  "
            f"corr P95={r['corrected_p95']:.6e}  "
            f"tech={int(r['technical_success'])}/64  "
            f"target={int(r['target_recovery'])}/63"
        )

    print(
        f"  aggregate direct mean      = "
        f"{g['direct_mean'].mean():.6e} "
        f"± {g['direct_mean'].std(ddof=1):.6e}"
    )

    print(
        f"  aggregate direct P95       = "
        f"{g['direct_p95'].mean():.6e} "
        f"± {g['direct_p95'].std(ddof=1):.6e}"
    )

    print(
        f"  aggregate corrected median = "
        f"{g['corrected_median'].mean():.6e} "
        f"± {g['corrected_median'].std(ddof=1):.6e}"
    )

    print(
        f"  aggregate corrected P95    = "
        f"{g['corrected_p95'].mean():.6e} "
        f"± {g['corrected_p95'].std(ddof=1):.6e}"
    )


physics = df[df["condition"] == "Physics-informed"]
anchors = df[df["condition"] == "Anchors-only"]

direct_mean_delta = (
    anchors["direct_mean"].mean()
    / physics["direct_mean"].mean()
    - 1.0
)

direct_p95_delta = (
    anchors["direct_p95"].mean()
    / physics["direct_p95"].mean()
    - 1.0
)

corr_med_delta = (
    anchors["corrected_median"].mean()
    / physics["corrected_median"].mean()
    - 1.0
)

corr_p95_delta = (
    anchors["corrected_p95"].mean()
    / physics["corrected_p95"].mean()
    - 1.0
)

print("\nAnchors-only relative to physics-informed:")
print(f"  direct mean      : {direct_mean_delta:+.3%}")
print(f"  direct P95       : {direct_p95_delta:+.3%}")
print(f"  corrected median : {corr_med_delta:+.3%}")
print(f"  corrected P95    : {corr_p95_delta:+.3%}")


# =============================================================================
# LaTeX helpers
# =============================================================================

def sci(x):
    return f"{x:.3e}"


def metric_cell(row, metric):
    value = float(row[metric])

    if row["row_type"] == "seed":
        return sci(value)

    sd = float(row[f"{metric}_sd"])

    return (
        rf"${sci(value)} \pm {sci(sd)}$"
    )


# =============================================================================
# LaTeX rows
# =============================================================================

latex_rows = []

for condition in conditions:
    subset = table[
        table["condition"] == condition
    ]

    for _, row in subset.iterrows():
        if row["row_type"] == "aggregate":
            seed_label = r"\textbf{Mean $\pm$ SD}"
            condition_label = rf"\textbf{{{condition}}}"

            cells = [
                condition_label,
                seed_label,
                metric_cell(row, "direct_mean"),
                metric_cell(row, "direct_p95"),
                metric_cell(row, "corrected_median"),
                metric_cell(row, "corrected_p95"),
                rf"\textbf{{{row['technical']}}}",
                rf"\textbf{{{row['target_recovery']}}}",
            ]

        else:
            cells = [
                condition,
                str(row["seed"]),
                metric_cell(row, "direct_mean"),
                metric_cell(row, "direct_p95"),
                metric_cell(row, "corrected_median"),
                metric_cell(row, "corrected_p95"),
                row["technical"],
                row["target_recovery"],
            ]

        latex_rows.append(
            " & ".join(cells) + r" \\"
        )

    if condition != conditions[-1]:
        latex_rows.append(r"\midrule")


# =============================================================================
# LaTeX table
# =============================================================================

latex = r"""\begin{table*}[t]
\centering
\caption{
Multi-seed ablation of the physics-informed terms at fixed $N=76$
spectral-anchor information. Three independent realizations are reported
for each condition. Direct errors refer to the neural spectral prediction
before classical correction; corrected errors are evaluated after
PINN-seeded Riccati shooting. Technical convergence is reported over all
64 validation coordinates. Target-branch recovery is reported over the
63 non-ambiguous coordinates after excluding the predeclared multibranch
case. The aggregate rows report the mean $\pm$ sample standard deviation
over the three seeds.
}
\label{tab:multiseed_physics_anchors_ablation}
\small
\setlength{\tabcolsep}{4.0pt}
\begin{tabular}{llcccccc}
\toprule
Condition
& Seed
& Direct mean
& Direct $P_{95}$
& Corrected median
& Corrected $P_{95}$
& Technical
& Target recovery \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""

OUT_TEX.write_text(latex)

print()
print(f"Saved CSV : {OUT_CSV}")
print(f"Saved TeX : {OUT_TEX}")
