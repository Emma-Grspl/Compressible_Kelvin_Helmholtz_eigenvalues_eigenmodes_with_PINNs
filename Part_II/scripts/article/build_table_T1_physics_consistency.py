from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PHASE1 = ROOT / "results" / "physics_residual_audit" / "independent"
PHASE2 = ROOT / "results" / "physics_residual_audit" / "stage_comparison"

FINAL_CSV = PHASE1 / "physics_residual_coordinate_summary.csv"
PAIRED_CSV = PHASE2 / "stage2_vs_stage3_paired_coordinates.csv"

OUT_DIR = ROOT / "assets" / "complementary" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "Table_T1_physics_consistency.csv"
OUT_TEX = OUT_DIR / "Table_T1_physics_consistency.tex"


# =============================================================================
# Helpers
# =============================================================================

def select_t401(df):
    if "dataset" not in df.columns:
        raise RuntimeError("Missing 'dataset' column.")

    mask = (
        df["dataset"]
        .astype(str)
        .str.lower()
        .str.contains("t401", regex=False)
    )

    out = df.loc[mask].copy()

    if len(out) != 401:
        raise RuntimeError(
            f"Expected 401 T401 rows, found {len(out)}."
        )

    return out


def stats(values):
    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(dtype=float)

    values = values[np.isfinite(values)]

    return {
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def sci(x):
    return f"{x:.2e}"


# =============================================================================
# Load
# =============================================================================

final = select_t401(
    pd.read_csv(FINAL_CSV)
)

paired = select_t401(
    pd.read_csv(PAIRED_CSV)
)


# =============================================================================
# Metric definition
# =============================================================================

metrics = [
    {
        "metric": r"$R_\kappa$",
        "final": "r_kappa_normalized_rms",
        "stage2": "r_kappa_normalized_rms_stage2",
        "stage3": "r_kappa_normalized_rms_stage3",
    },
    {
        "metric": r"$R_q$",
        "final": "r_q_normalized_rms",
        "stage2": "r_q_normalized_rms_stage2",
        "stage3": "r_q_normalized_rms_stage3",
    },
    {
        "metric": r"$R_\ell$",
        "final": "r_logamp_normalized_rms",
        "stage2": "r_logamp_normalized_rms_stage2",
        "stage3": "r_logamp_normalized_rms_stage3",
    },
    {
        "metric": r"$BC_\kappa$",
        "final": "bc_kappa_rmse",
        "stage2": "bc_kappa_rmse_stage2",
        "stage3": "bc_kappa_rmse_stage3",
    },
    {
        "metric": r"$BC_q$",
        "final": "bc_q_rmse",
        "stage2": "bc_q_rmse_stage2",
        "stage3": "bc_q_rmse_stage3",
    },
]


# =============================================================================
# Build table
# =============================================================================

rows = []

for m in metrics:
    for c in [m["final"], m["stage2"], m["stage3"]]:
        source = final if c == m["final"] else paired

        if c not in source.columns:
            raise RuntimeError(
                f"Missing required column: {c}"
            )

    final_stats = stats(final[m["final"]])

    stage2 = pd.to_numeric(
        paired[m["stage2"]],
        errors="coerce",
    ).to_numpy(dtype=float)

    stage3 = pd.to_numeric(
        paired[m["stage3"]],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(stage2)
        & np.isfinite(stage3)
    )

    stage2 = stage2[valid]
    stage3 = stage3[valid]

    if len(stage2) != 401:
        raise RuntimeError(
            f"{m['metric']}: expected 401 paired values, "
            f"found {len(stage2)}."
        )

    med2 = float(np.median(stage2))
    med3 = float(np.median(stage3))

    reduction = 1.0 - med3 / med2
    fraction_improved = float(
        np.mean(stage3 < stage2)
    )

    rows.append(
        {
            "metric": m["metric"],
            "final_median": final_stats["median"],
            "final_p95": final_stats["p95"],
            "stage2_median": med2,
            "stage3_median": med3,
            "median_reduction": reduction,
            "fraction_improved": fraction_improved,
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

print("=" * 92)
print("T1 — INDEPENDENT PHYSICS-CONSISTENCY SUMMARY")
print("=" * 92)

print(
    table.to_string(
        index=False,
        formatters={
            "final_median": lambda x: f"{x:.6e}",
            "final_p95": lambda x: f"{x:.6e}",
            "stage2_median": lambda x: f"{x:.6e}",
            "stage3_median": lambda x: f"{x:.6e}",
            "median_reduction": lambda x: f"{x:.1%}",
            "fraction_improved": lambda x: f"{x:.1%}",
        },
    )
)


# =============================================================================
# LaTeX
# =============================================================================

latex_rows = []

for _, row in table.iterrows():
    latex_rows.append(
        " & ".join(
            [
                row["metric"],
                sci(row["final_median"]),
                sci(row["final_p95"]),
                sci(row["stage2_median"]),
                sci(row["stage3_median"]),
                f"{100 * row['median_reduction']:.1f}\\%",
                f"{100 * row['fraction_improved']:.1f}\\%",
            ]
        )
        + r" \\"
    )

latex = r"""\begin{table}[t]
\centering
\caption{
Independent physics-consistency audit on the sealed $T_{401}$ set.
Final residual statistics are evaluated independently of the training
collocation points. Stage~2 and Stage~3 values are paired coordinate by
coordinate. The median reduction is
$1-\mathrm{median}_{\mathrm{S3}}/\mathrm{median}_{\mathrm{S2}}$.
}
\label{tab:independent_physics_consistency}
\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lcccccc}
\toprule
Metric
& Final med.
& Final $P_{95}$
& Stage 2 med.
& Stage 3 med.
& Med. reduction
& Improved \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

OUT_TEX.write_text(latex)

print()
print(f"Saved CSV : {OUT_CSV}")
print(f"Saved TeX : {OUT_TEX}")
