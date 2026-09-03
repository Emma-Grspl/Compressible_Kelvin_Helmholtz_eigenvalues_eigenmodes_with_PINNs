from pathlib import Path
import json

import pandas as pd


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

ROUTING_DIR = (
    ROOT
    / "results"
    / "routing_audit"
    / "results"
    / "routing_boundary_audit"
)

SUMMARY_JSON = ROUTING_DIR / "routing_summary.json"
EPS_CSV = ROUTING_DIR / "routing_summary_by_eps.csv"
BOUNDARIES_CSV = ROUTING_DIR / "routing_boundaries.csv"

OUT_DIR = ROOT / "assets" / "complementary" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "Table_T5_routing_boundary_summary.csv"
OUT_TEX = OUT_DIR / "Table_T5_routing_boundary_summary.tex"


# =============================================================================
# Load
# =============================================================================

with open(SUMMARY_JSON) as f:
    summary = json.load(f)

eps = pd.read_csv(EPS_CSV)
boundaries = pd.read_csv(BOUNDARIES_CSV)


# =============================================================================
# Validate
# =============================================================================

if len(boundaries) != 85:
    raise RuntimeError(
        f"Expected 85 representative boundary locations, found {len(boundaries)}."
    )

n_pairs = boundaries["pair_id"].nunique()

if n_pairs != 17:
    raise RuntimeError(
        f"Expected 17 neighboring chart interfaces, found {n_pairs}."
    )

required_summary = [
    "n_boundaries",
    "n_unique_chart_pairs",
    "samepoint_c_median",
    "samepoint_c_p95",
    "samepoint_c_max",
    "samepoint_cr_median",
    "samepoint_cr_p95",
    "samepoint_cr_max",
    "samepoint_ci_median",
    "samepoint_ci_p95",
    "samepoint_ci_max",
]

missing = [k for k in required_summary if k not in summary]

if missing:
    raise RuntimeError(
        f"routing_summary.json missing keys: {missing}"
    )

required_eps = [
    "eps",
    "n",
    "route_changes",
    "routed_c_median",
    "routed_c_p95",
    "routed_c_max",
    "routed_cr_median",
    "routed_cr_p95",
    "routed_cr_max",
    "routed_ci_median",
    "routed_ci_p95",
    "routed_ci_max",
]

missing = [c for c in required_eps if c not in eps.columns]

if missing:
    raise RuntimeError(
        f"routing_summary_by_eps.csv missing columns: {missing}"
    )


# =============================================================================
# Build source table
# =============================================================================

rows = [
    {
        "evaluation": "Same point",
        "eps": None,
        "n": int(summary["n_boundaries"]),
        "route_changes": None,
        "c_median": float(summary["samepoint_c_median"]),
        "c_p95": float(summary["samepoint_c_p95"]),
        "c_max": float(summary["samepoint_c_max"]),
        "cr_median": float(summary["samepoint_cr_median"]),
        "cr_p95": float(summary["samepoint_cr_p95"]),
        "cr_max": float(summary["samepoint_cr_max"]),
        "ci_median": float(summary["samepoint_ci_median"]),
        "ci_p95": float(summary["samepoint_ci_p95"]),
        "ci_max": float(summary["samepoint_ci_max"]),
    }
]

for _, row in eps.sort_values("eps").iterrows():
    rows.append(
        {
            "evaluation": f"Routed eps={row['eps']:.0e}",
            "eps": float(row["eps"]),
            "n": int(row["n"]),
            "route_changes": int(row["route_changes"]),
            "c_median": float(row["routed_c_median"]),
            "c_p95": float(row["routed_c_p95"]),
            "c_max": float(row["routed_c_max"]),
            "cr_median": float(row["routed_cr_median"]),
            "cr_p95": float(row["routed_cr_p95"]),
            "cr_max": float(row["routed_cr_max"]),
            "ci_median": float(row["routed_ci_median"]),
            "ci_p95": float(row["routed_ci_p95"]),
            "ci_max": float(row["routed_ci_max"]),
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

print("=" * 112)
print("T5 — ROUTING-BOUNDARY DISCONTINUITY SUMMARY")
print("=" * 112)

print(f"Unique interfaces          : {n_pairs}")
print(f"Representative locations  : {len(boundaries)}")

print()
print(
    table.to_string(
        index=False,
        formatters={
            "eps": lambda x: "—" if pd.isna(x) else f"{x:.0e}",
            "c_median": lambda x: f"{x:.6e}",
            "c_p95": lambda x: f"{x:.6e}",
            "c_max": lambda x: f"{x:.6e}",
            "cr_median": lambda x: f"{x:.6e}",
            "cr_p95": lambda x: f"{x:.6e}",
            "cr_max": lambda x: f"{x:.6e}",
            "ci_median": lambda x: f"{x:.6e}",
            "ci_p95": lambda x: f"{x:.6e}",
            "ci_max": lambda x: f"{x:.6e}",
        },
    )
)


# =============================================================================
# LaTeX helpers
# =============================================================================

def sci(x):
    return f"{float(x):.3e}"


# =============================================================================
# LaTeX rows
# =============================================================================

latex_rows = []

# Same-point row
r = table.iloc[0]

latex_rows.append(
    " & ".join(
        [
            "Same point",
            "--",
            str(int(r["n"])),
            "--",
            sci(r["c_median"]),
            sci(r["c_p95"]),
            sci(r["c_max"]),
            sci(r["cr_median"]),
            sci(r["ci_median"]),
        ]
    )
    + r" \\"
)

latex_rows.append(r"\midrule")

# Routed rows
for _, r in table.iloc[1:].iterrows():
    latex_rows.append(
        " & ".join(
            [
                "Routed",
                rf"${r['eps']:.0e}$",
                str(int(r["n"])),
                str(int(r["route_changes"])),
                sci(r["c_median"]),
                sci(r["c_p95"]),
                sci(r["c_max"]),
                sci(r["cr_median"]),
                sci(r["ci_median"]),
            ]
        )
        + r" \\"
    )


# =============================================================================
# LaTeX
# =============================================================================

latex = r"""\begin{table*}[t]
\centering
\caption{
Routing-boundary discontinuity audit over 17 neighboring chart interfaces
and 85 representative boundary locations. Same-point discrepancies compare
the two adjacent chart predictions at the identical parameter coordinate.
Routed discrepancies compare predictions at parameter points displaced by
$\pm\varepsilon$ across the routing boundary. Persistence of the routed
jump as $\varepsilon\rightarrow0$ indicates a genuine chart-switching
discontinuity rather than a finite-offset artifact.
}
\label{tab:routing_boundary_summary}
\small
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{llccccccc}
\toprule
Evaluation
& $\varepsilon$
& $N$
& Route changes
& Med. $|\Delta c|$
& $P_{95}$ $|\Delta c|$
& Max $|\Delta c|$
& Med. $|\Delta c_r|$
& Med. $|\Delta c_i|$ \\
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
