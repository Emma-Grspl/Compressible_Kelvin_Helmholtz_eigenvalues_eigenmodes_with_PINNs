from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PHASE6_CSV = (
    ROOT
    / "results"
    / "matched_branch_localization"
    / "generic_vs_seeded"
    / "results"
    / "analysis"
    / "condition_summary.csv"
)

PHASE7_CSV = (
    ROOT
    / "results"
    / "matched_branch_localization"
    / "budget_matched"
    / "budget_matched_T401"
    / "analysis"
    / "phase7_condition_summary.csv"
)

OUT_DIR = ROOT / "assets" / "complementary" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = (
    OUT_DIR
    / "Table_T3T4_matched_branch_localization.csv"
)

OUT_TEX = (
    OUT_DIR
    / "Table_T3T4_matched_branch_localization.tex"
)


# =============================================================================
# Helpers
# =============================================================================

def normalize_condition(value):
    s = str(value).strip().lower()

    if "seed" in s or "pinn" in s or "atlas" in s:
        return "PINN-seeded"

    if "generic" in s:
        return "Matched generic"

    raise RuntimeError(
        f"Could not classify condition: {value!r}"
    )


def require_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{name}: missing columns {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def as_count(value, denominator):
    """
    Accept either an explicit integer count or a fraction/rate.
    """
    value = float(value)

    if value <= 1.0 + 1e-12:
        return int(round(value * denominator))

    return int(round(value))


def sci(x):
    return f"{float(x):.3e}"


# =============================================================================
# Load
# =============================================================================

v64 = pd.read_csv(PHASE6_CSV)
t401 = pd.read_csv(PHASE7_CSV)


# =============================================================================
# Validate Phase 6 schema
# =============================================================================

phase6_required = [
    "condition",
    "technical_success",
    "technical_total",
    "target_recovery",
    "target_total",
    "target_error_median",
    "target_error_p95",
    "target_error_le_1e5",
    "target_error_le_1e4",
]

require_columns(
    v64,
    phase6_required,
    "Phase 6 V64",
)


# =============================================================================
# Validate Phase 7 schema
# =============================================================================

phase7_required = [
    "condition",
    "n",
    "technical_success",
    "target_recovery",
    "target_error_median",
    "target_error_p95",
    "error_le_1e-5",
    "error_le_1e-4",
]

require_columns(
    t401,
    phase7_required,
    "Phase 7 T401",
)


# =============================================================================
# Parse Phase 6 — V64
# =============================================================================

rows = []

for _, row in v64.iterrows():

    technical_total = int(row["technical_total"])
    target_total = int(row["target_total"])

    rows.append(
        {
            "benchmark": "V64",
            "condition": normalize_condition(row["condition"]),

            "technical_count": int(round(float(
                row["technical_success"]
            ))),
            "technical_total": technical_total,

            "target_count": int(round(float(
                row["target_recovery"]
            ))),
            "target_total": target_total,

            "target_error_median": float(
                row["target_error_median"]
            ),
            "target_error_p95": float(
                row["target_error_p95"]
            ),

            "error_le_1e-5": int(round(float(
                row["target_error_le_1e5"]
            ))),

            "error_le_1e-4": int(round(float(
                row["target_error_le_1e4"]
            ))),
        }
    )


# =============================================================================
# Parse Phase 7 — T401
# =============================================================================

for _, row in t401.iterrows():

    n = int(row["n"])

    rows.append(
        {
            "benchmark": "T401",
            "condition": normalize_condition(row["condition"]),

            "technical_count": int(round(float(
                row["technical_success"]
            ))),
            "technical_total": n,

            "target_count": int(round(float(
                row["target_recovery"]
            ))),
            "target_total": n,

            "target_error_median": float(
                row["target_error_median"]
            ),
            "target_error_p95": float(
                row["target_error_p95"]
            ),

            "error_le_1e-5": int(round(float(
                row["error_le_1e-5"]
            ))),

            "error_le_1e-4": int(round(float(
                row["error_le_1e-4"]
            ))),
        }
    )


table = pd.DataFrame(rows)


# =============================================================================
# Validate expected 2 x 2 structure
# =============================================================================

expected = {
    ("V64", "PINN-seeded"),
    ("V64", "Matched generic"),
    ("T401", "PINN-seeded"),
    ("T401", "Matched generic"),
}

observed = set(
    zip(
        table["benchmark"],
        table["condition"],
    )
)

if observed != expected:
    raise RuntimeError(
        "Unexpected benchmark/condition combinations.\n"
        f"Expected: {expected}\n"
        f"Observed: {observed}"
    )


# =============================================================================
# Sort
# =============================================================================

benchmark_order = {
    "V64": 0,
    "T401": 1,
}

condition_order = {
    "PINN-seeded": 0,
    "Matched generic": 1,
}

table["_b"] = table["benchmark"].map(
    benchmark_order
)

table["_c"] = table["condition"].map(
    condition_order
)

table = (
    table
    .sort_values(["_b", "_c"])
    .drop(columns=["_b", "_c"])
    .reset_index(drop=True)
)


# =============================================================================
# Hard scientific checks
# =============================================================================

def get_row(benchmark, condition):
    g = table[
        (table["benchmark"] == benchmark)
        & (table["condition"] == condition)
    ]

    if len(g) != 1:
        raise RuntimeError(
            f"Expected one row for {benchmark}/{condition}, "
            f"found {len(g)}."
        )

    return g.iloc[0]


checks = [
    (
        "V64 seeded technical",
        get_row("V64", "PINN-seeded")["technical_count"],
        64,
    ),
    (
        "V64 generic technical",
        get_row("V64", "Matched generic")["technical_count"],
        64,
    ),
    (
        "V64 seeded target recovery",
        get_row("V64", "PINN-seeded")["target_count"],
        63,
    ),
    (
        "V64 generic target recovery",
        get_row("V64", "Matched generic")["target_count"],
        0,
    ),
    (
        "T401 seeded technical",
        get_row("T401", "PINN-seeded")["technical_count"],
        401,
    ),
    (
        "T401 generic technical",
        get_row("T401", "Matched generic")["technical_count"],
        401,
    ),
    (
        "T401 seeded target recovery",
        get_row("T401", "PINN-seeded")["target_count"],
        400,
    ),
    (
        "T401 generic target recovery",
        get_row("T401", "Matched generic")["target_count"],
        1,
    ),
]

for label, observed_value, expected_value in checks:
    observed_value = int(observed_value)

    if observed_value != expected_value:
        raise RuntimeError(
            f"{label}: expected {expected_value}, "
            f"found {observed_value}."
        )


# =============================================================================
# Save source CSV
# =============================================================================

table.to_csv(
    OUT_CSV,
    index=False,
)


# =============================================================================
# Terminal audit
# =============================================================================

print("=" * 116)
print("T3/T4 — MATCHED BRANCH-LOCALIZATION CONTROLS")
print("=" * 116)

display = table.copy()

display["technical"] = (
    display["technical_count"].astype(int).astype(str)
    + "/"
    + display["technical_total"].astype(int).astype(str)
)

display["target_recovery"] = (
    display["target_count"].astype(int).astype(str)
    + "/"
    + display["target_total"].astype(int).astype(str)
)

display["le_1e-5"] = (
    display["error_le_1e-5"].astype(int).astype(str)
    + "/"
    + display["target_total"].astype(int).astype(str)
)

display["le_1e-4"] = (
    display["error_le_1e-4"].astype(int).astype(str)
    + "/"
    + display["target_total"].astype(int).astype(str)
)

print(
    display[
        [
            "benchmark",
            "condition",
            "technical",
            "target_recovery",
            "target_error_median",
            "target_error_p95",
            "le_1e-5",
            "le_1e-4",
        ]
    ].to_string(
        index=False,
        formatters={
            "target_error_median":
                lambda x: f"{x:.6e}",
            "target_error_p95":
                lambda x: f"{x:.6e}",
        },
    )
)


# =============================================================================
# LaTeX rows
# =============================================================================

latex_rows = []

for benchmark in ["V64", "T401"]:

    subset = table[
        table["benchmark"] == benchmark
    ]

    first = True

    for _, row in subset.iterrows():

        benchmark_cell = benchmark if first else ""
        first = False

        technical = (
            f"{int(row['technical_count'])}/"
            f"{int(row['technical_total'])}"
        )

        target = (
            f"{int(row['target_count'])}/"
            f"{int(row['target_total'])}"
        )

        med = sci(
            row["target_error_median"]
        )

        p95 = sci(
            row["target_error_p95"]
        )

        le5 = (
            f"{int(row['error_le_1e-5'])}/"
            f"{int(row['target_total'])}"
        )

        le4 = (
            f"{int(row['error_le_1e-4'])}/"
            f"{int(row['target_total'])}"
        )

        if row["condition"] == "PINN-seeded":
            condition = r"\textbf{PINN-seeded}"
            target = rf"\textbf{{{target}}}"
            med = rf"\textbf{{{med}}}"
            p95 = rf"\textbf{{{p95}}}"
            le5 = rf"\textbf{{{le5}}}"
            le4 = rf"\textbf{{{le4}}}"
        else:
            condition = "Matched generic"

        latex_rows.append(
            " & ".join(
                [
                    benchmark_cell,
                    condition,
                    technical,
                    target,
                    med,
                    p95,
                    le5,
                    le4,
                ]
            )
            + r" \\"
        )

    if benchmark != "T401":
        latex_rows.append(r"\midrule")


# =============================================================================
# LaTeX table
# =============================================================================

latex = r"""\begin{table*}[t]
\centering
\caption{
Matched branch-localization controls for the classical Riccati corrector.
The PINN-seeded and generic conditions use the same external local-search
policy and search budget. Technical convergence denotes successful
completion of the classical correction and must therefore be distinguished
from recovery of the intended spectral branch. For V64, target-branch
statistics are evaluated over the 63 branch-labelled coordinates after
excluding the predeclared multibranch case; T401 uses all 401 sealed test
coordinates.
}
\label{tab:matched_branch_localization}
\small
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{llcccccc}
\toprule
Benchmark
& Initialization
& Technical
& Target recovery
& Median error
& $P_{95}$ error
& $\leq 10^{-5}$
& $\leq 10^{-4}$ \\
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
