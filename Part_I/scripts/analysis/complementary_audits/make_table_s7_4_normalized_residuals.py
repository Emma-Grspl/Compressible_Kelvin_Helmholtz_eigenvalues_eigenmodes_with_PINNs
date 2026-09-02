from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path("assets/p1c_chartwise_active_normalized_residuals.csv")
OUTDIR = Path("assets")
OUTDIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUTDIR / "Table_S7_4_normalized_residuals.csv"
TEX_OUT = OUTDIR / "Table_S7_4_normalized_residuals.tex"

df = pd.read_csv(SRC)

# ------------------------------------------------------------------
# Find columns
# ------------------------------------------------------------------
def find_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not find any of {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )

family_col = find_col([
    "field_family",
    "family",
    "chart_family",
    "formulation",
    "modal_family",
])

ode_col = find_col([
    "ode_norm",
    "ode_normalized",
    "ode_norm_active",
    "active_ode_norm",
])

compat_col = find_col([
    "compat_norm",
    "compatibility_norm",
    "compat_norm_active",
    "active_compat_norm",
])

print("Using:")
print(" family :", family_col)
print(" ODE    :", ode_col)
print(" compat :", compat_col)

# ------------------------------------------------------------------
# Clean
# ------------------------------------------------------------------
work = df[[family_col, ode_col, compat_col]].copy()
work[ode_col] = pd.to_numeric(work[ode_col], errors="coerce")
work[compat_col] = pd.to_numeric(work[compat_col], errors="coerce")

work = work.dropna(subset=[family_col, ode_col, compat_col])

print()
print("Charts:", len(work))
print(work[family_col].value_counts().to_string())

if len(work) != 49:
    raise RuntimeError(f"Expected 49 charts, found {len(work)}")

# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------
def stats(x):
    x = np.asarray(x, dtype=float)
    return {
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }

rows = []

for family in ["pQscaled", "pq_etaaware", "pq_legacy"]:
    sub = work.loc[work[family_col].astype(str) == family]

    if len(sub) == 0:
        raise RuntimeError(f"No rows found for family {family}")

    s = stats(sub[ode_col])

    rows.append({
        "Diagnostic": "Normalized ODE residual",
        "Family": family,
        "Median": s["median"],
        "P95": s["p95"],
        "Maximum": s["max"],
    })

s = stats(work[compat_col])

rows.append({
    "Diagnostic": "Normalized compatibility residual",
    "Family": "All charts",
    "Median": s["median"],
    "P95": s["p95"],
    "Maximum": s["max"],
})

out = pd.DataFrame(rows)

# ------------------------------------------------------------------
# Sanity checks against retained results
# ------------------------------------------------------------------
expected = {
    "pQscaled": {
        "median": 0.5061074132054806,
        "p95": 0.7922461699675383,
        "max": 0.8359466905258366,
    },
    "pq_etaaware": {
        "median": 0.009247137409351182,
        "p95": 0.05876382034890375,
        "max": 0.18092375764576485,
    },
    "pq_legacy": {
        "median": 0.004266195611887004,
        "p95": 0.06165683475006245,
        "max": 0.07887585755573355,
    },
}

for family, target in expected.items():
    row = out[
        (out["Diagnostic"] == "Normalized ODE residual")
        & (out["Family"] == family)
    ].iloc[0]

    for key, column in [
        ("median", "Median"),
        ("p95", "P95"),
        ("max", "Maximum"),
    ]:
        if not np.isclose(row[column], target[key], rtol=1e-8, atol=1e-12):
            raise RuntimeError(
                f"{family} {column}: got {row[column]}, "
                f"expected {target[key]}"
            )

compat = out[
    out["Diagnostic"] == "Normalized compatibility residual"
].iloc[0]

assert np.isclose(compat["Median"], 0.001086534395504195, rtol=1e-8)
assert np.isclose(compat["P95"], 0.0029116995067997313, rtol=1e-8)
assert np.isclose(compat["Maximum"], 0.0036314791945782163, rtol=1e-8)

# ------------------------------------------------------------------
# CSV
# ------------------------------------------------------------------
out.to_csv(CSV_OUT, index=False)

# ------------------------------------------------------------------
# LaTeX
# ------------------------------------------------------------------
def sci(x):
    if x == 0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exponent)
    return rf"${mantissa:.2f}\times10^{{{exponent}}}$"

lines = [
    r"\begin{table}[!htbp]",
    r"\centering",
    r"\small",
    r"\caption{",
    r"Chart-wise normalized residuals evaluated in the active modal region.",
    r"The ODE residual is reported separately for the three modal formulations.",
    r"The compatibility residual is summarized over the complete 49-chart atlas.",
    r"}",
    r"\label{tab:supp_chartwise_normalized_residuals}",
    r"\begin{tabular}{llccc}",
    r"\hline",
    r"Diagnostic & Formulation & Median & 95th percentile & Maximum \\",
    r"\hline",
]

for _, row in out.iterrows():
    diagnostic = str(row["Diagnostic"])
    family = str(row["Family"])

    if diagnostic == "Normalized ODE residual":
        diag_tex = r"Normalized ODE residual"
    else:
        diag_tex = r"Normalized compatibility residual"

    family_tex = family.replace("_", r"\_")

    lines.append(
        f"{diag_tex} & {family_tex} & "
        f"{sci(row['Median'])} & "
        f"{sci(row['P95'])} & "
        f"{sci(row['Maximum'])} \\\\"
    )

lines += [
    r"\hline",
    r"\end{tabular}",
    r"\end{table}",
]

TEX_OUT.write_text("\n".join(lines) + "\n")

# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------
print()
print("=" * 100)
print("TABLE S7.4")
print("=" * 100)
print(out.to_string(index=False))

print()
print("WROTE:", CSV_OUT)
print("WROTE:", TEX_OUT)
