from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    ROOT
    / "results"
    / "blumen_cr_validation"
    / "results"
    / "phase11_final_stratified_cr_validation.csv"
)

OUT_DIR = ROOT / "assets" / "complementary" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "Table_T7_blumen_cr_validation.csv"
OUT_TEX = OUT_DIR / "Table_T7_blumen_cr_validation.tex"


# =============================================================================
# Helpers
# =============================================================================

def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def sci(x, digits=4):
    if pd.isna(x):
        return "--"

    return f"{float(x):.{digits}f}"


# =============================================================================
# Load
# =============================================================================

df = pd.read_csv(INPUT_CSV)


# =============================================================================
# Validate
# =============================================================================

required = [
    "Mach",
    "alpha_target",
    "bootstrap_valid_count",
    "bootstrap_valid_fraction",
    "bootstrap_q05",
    "bootstrap_q50",
    "bootstrap_q95",
    "reference_cr",
    "reference_inside_ci90",
    "reference_zscore",
    "externally_available",
    "externally_usable",
    "weakly_constrained",
    "status",
    "abs_ref_minus_q50",
    "relative_ref_minus_q50",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

if len(df) != 9:
    raise RuntimeError(
        f"Expected 9 predeclared validation coordinates, found {len(df)}."
    )


df["available"] = as_bool(df["externally_available"])
df["usable"] = as_bool(df["externally_usable"])
df["weak"] = as_bool(df["weakly_constrained"])
df["inside_ci90"] = as_bool(df["reference_inside_ci90"])


# =============================================================================
# Structural checks
# =============================================================================

alphas = np.sort(df["alpha_target"].unique())
machs = np.sort(df["Mach"].unique())

if len(alphas) != 3:
    raise RuntimeError(
        f"Expected 3 alpha levels, found {alphas}."
    )

if len(machs) != 3:
    raise RuntimeError(
        f"Expected 3 Mach levels, found {machs}."
    )

for alpha in alphas:
    group = df[np.isclose(df["alpha_target"], alpha)]

    if len(group) != 3:
        raise RuntimeError(
            f"alpha={alpha}: expected 3 Mach points, found {len(group)}."
        )


# =============================================================================
# Audit counts
# =============================================================================

n_planned = len(df)
n_available = int(df["available"].sum())
n_usable = int(df["usable"].sum())
n_weak = int(df["weak"].sum())
n_unavailable = int((~df["available"]).sum())

usable = df[df["usable"]].copy()

n_inside = int(usable["inside_ci90"].sum())

if n_planned != 9:
    raise RuntimeError("Expected 9 planned coordinates.")

if n_available != 8:
    raise RuntimeError(
        f"Expected 8 externally available coordinates, found {n_available}."
    )

if n_usable != 7:
    raise RuntimeError(
        f"Expected 7 externally usable coordinates, found {n_usable}."
    )

if n_weak != 1:
    raise RuntimeError(
        f"Expected 1 weakly constrained coordinate, found {n_weak}."
    )

if n_unavailable != 1:
    raise RuntimeError(
        f"Expected 1 unavailable coordinate, found {n_unavailable}."
    )

if n_inside != 4:
    raise RuntimeError(
        f"Expected 4/7 usable references inside nominal CI90, found {n_inside}/7."
    )


# =============================================================================
# Publication status
# =============================================================================

def publication_status(row):
    if not row["available"]:
        return "Unavailable"

    if row["weak"]:
        return "Weak"

    if row["usable"]:
        return "Usable"

    return "Available"


df["publication_status"] = df.apply(
    publication_status,
    axis=1,
)


# =============================================================================
# Save clean CSV
# =============================================================================

table_cols = [
    "alpha_target",
    "Mach",
    "publication_status",
    "bootstrap_valid_count",
    "bootstrap_valid_fraction",
    "bootstrap_q05",
    "bootstrap_q50",
    "bootstrap_q95",
    "reference_cr",
    "abs_ref_minus_q50",
    "relative_ref_minus_q50",
    "reference_inside_ci90",
    "reference_zscore",
]

table = (
    df[table_cols]
    .sort_values(["alpha_target", "Mach"])
    .reset_index(drop=True)
)

table.to_csv(
    OUT_CSV,
    index=False,
)


# =============================================================================
# Aggregate numerical audit
# =============================================================================

abs_disc = pd.to_numeric(
    usable["abs_ref_minus_q50"],
    errors="coerce",
).to_numpy(dtype=float)

rel_disc = pd.to_numeric(
    usable["relative_ref_minus_q50"],
    errors="coerce",
).to_numpy(dtype=float)

print("=" * 102)
print("T7 — INDEPENDENT BLUMEN c_r VALIDATION")
print("=" * 102)

print(f"Planned coordinates        : {n_planned}")
print(f"Externally available       : {n_available}")
print(f"Externally usable          : {n_usable}")
print(f"Weakly constrained         : {n_weak}")
print(f"Unavailable                : {n_unavailable}")
print(f"Inside nominal CI90        : {n_inside}/{n_usable}")

print()
print("Usable-point discrepancy statistics")
print(
    f"  median |Delta cr|        : "
    f"{np.nanmedian(abs_disc):.6e}"
)
print(
    f"  P95 |Delta cr|           : "
    f"{np.nanquantile(abs_disc, 0.95):.6e}"
)
print(
    f"  max |Delta cr|           : "
    f"{np.nanmax(abs_disc):.6e}"
)

print(
    f"  median relative          : "
    f"{np.nanmedian(rel_disc):.3%}"
)
print(
    f"  P95 relative             : "
    f"{np.nanquantile(rel_disc, 0.95):.3%}"
)
print(
    f"  max relative             : "
    f"{np.nanmax(rel_disc):.3%}"
)

print()
print(
    table.to_string(
        index=False,
        formatters={
            "bootstrap_valid_fraction":
                lambda x: f"{x:.3f}",
            "bootstrap_q05":
                lambda x: "--" if pd.isna(x) else f"{x:.6f}",
            "bootstrap_q50":
                lambda x: "--" if pd.isna(x) else f"{x:.6f}",
            "bootstrap_q95":
                lambda x: "--" if pd.isna(x) else f"{x:.6f}",
            "reference_cr":
                lambda x: f"{x:.6f}",
            "abs_ref_minus_q50":
                lambda x: "--" if pd.isna(x) else f"{x:.6e}",
            "relative_ref_minus_q50":
                lambda x: "--" if pd.isna(x) else f"{x:.3%}",
            "reference_zscore":
                lambda x: "--" if pd.isna(x) else f"{x:.3f}",
        },
    )
)


# =============================================================================
# LaTeX helpers
# =============================================================================

def bootstrap_interval(row):
    if not row["available"]:
        return r"--"

    q05 = row["bootstrap_q05"]
    q95 = row["bootstrap_q95"]

    if pd.isna(q05) or pd.isna(q95):
        return r"--"

    return (
        rf"$[{float(q05):.4f},\,{float(q95):.4f}]$"
    )


def q50_cell(row):
    if not row["available"] or pd.isna(row["bootstrap_q50"]):
        return r"--"

    return f"{float(row['bootstrap_q50']):.4f}"


def discrepancy_cell(row):
    if not row["usable"]:
        return r"--"

    return f"{float(row['abs_ref_minus_q50']):.3e}"


def relative_cell(row):
    if not row["usable"]:
        return r"--"

    return f"{100 * float(row['relative_ref_minus_q50']):.2f}\\%"


def inside_cell(row):
    if not row["usable"]:
        return r"--"

    return "yes" if row["inside_ci90"] else "no"


def valid_fraction_cell(row):
    if not row["available"]:
        return r"--"

    return f"{100 * float(row['bootstrap_valid_fraction']):.1f}\\%"


def status_cell(row):
    status = row["publication_status"]

    if status == "Usable":
        return "usable"

    if status == "Weak":
        return r"\textit{weak}"

    if status == "Unavailable":
        return r"\textit{unavailable}"

    return status.lower()


# =============================================================================
# LaTeX rows
# =============================================================================

latex_rows = []

df_sorted = (
    df.sort_values(["alpha_target", "Mach"])
    .reset_index(drop=True)
)

last_alpha = None

for i, row in df_sorted.iterrows():

    alpha = float(row["alpha_target"])

    if last_alpha is not None and not np.isclose(alpha, last_alpha):
        latex_rows.append(r"\midrule")

    alpha_cell = (
        f"{alpha:.2f}"
        if last_alpha is None or not np.isclose(alpha, last_alpha)
        else ""
    )

    cells = [
        alpha_cell,
        f"{float(row['Mach']):.2f}",
        status_cell(row),
        valid_fraction_cell(row),
        q50_cell(row),
        bootstrap_interval(row),
        f"{float(row['reference_cr']):.4f}",
        discrepancy_cell(row),
        relative_cell(row),
        inside_cell(row),
    ]

    latex_rows.append(
        " & ".join(cells) + r" \\"
    )

    last_alpha = alpha


# =============================================================================
# LaTeX table
# =============================================================================

latex = r"""\begin{table*}[t]
\centering
\caption{
Independent validation of the real phase velocity $c_r$ against the
digitized Blumen reference on the predeclared $3\times3$ $(M,\alpha)$
grid. The Blumen estimate is reported as the bootstrap median $q_{50}$
with the nominal $q_{05}$--$q_{95}$ digitization/interpolation range.
This range quantifies sensitivity of the digitization and local
interpolation procedure and is not a calibrated statistical uncertainty
interval. Absolute and relative discrepancies are reported only for the
seven externally usable reconstructions. ``Inside'' indicates whether the
final classical reference lies inside the nominal bootstrap range.
}
\label{tab:blumen_cr_validation}
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{cccccccccc}
\toprule
$\alpha$
& $M$
& Status
& Valid boot.
& Blumen $q_{50}$
& $[q_{05},q_{95}]$
& $c_r^{\rm ref}$
& $|\Delta c_r|$
& Rel. diff.
& Inside \\
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
