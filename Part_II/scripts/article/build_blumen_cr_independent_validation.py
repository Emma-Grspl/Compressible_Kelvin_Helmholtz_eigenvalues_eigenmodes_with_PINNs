from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

OUT_DIR = (
    ROOT
    / "assets"
    / "complementary"
    / "blumen_cr"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Fig_blumen_cr_independent_validation.png"
OUT_PDF = OUT_DIR / "Fig_blumen_cr_independent_validation.pdf"

# Useful directly for the future Supplement table T7.
OUT_TABLE = OUT_DIR / "blumen_cr_validation_table_source.csv"
OUT_SUMMARY = OUT_DIR / "blumen_cr_validation_summary.csv"


# =============================================================================
# Helpers
# =============================================================================

def as_bool(series: pd.Series) -> pd.Series:
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


def finite_or_nan(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


# =============================================================================
# Load and validate
# =============================================================================

df = pd.read_csv(INPUT_CSV)

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
        f"Expected the predeclared 3x3 grid (9 rows), found {len(df)}."
    )

df["available"] = as_bool(df["externally_available"])
df["usable"] = as_bool(df["externally_usable"])
df["weak"] = as_bool(df["weakly_constrained"])
df["inside_ci90"] = as_bool(df["reference_inside_ci90"])

alphas = np.sort(
    pd.to_numeric(
        df["alpha_target"],
        errors="raise",
    ).unique()
)

machs = np.sort(
    pd.to_numeric(
        df["Mach"],
        errors="raise",
    ).unique()
)

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
            f"Expected 3 Mach points at alpha={alpha}, found {len(group)}."
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

if len(usable) == 0:
    raise RuntimeError("No externally usable points detected.")

abs_disc = finite_or_nan(usable["abs_ref_minus_q50"])
rel_disc = finite_or_nan(usable["relative_ref_minus_q50"])

n_inside_usable = int(
    (
        usable["inside_ci90"]
    ).sum()
)

n_outside_usable = len(usable) - n_inside_usable


# =============================================================================
# Save clean table source
# =============================================================================

table_cols = [
    "Mach",
    "alpha_target",
    "bootstrap_valid_count",
    "bootstrap_valid_fraction",
    "bootstrap_q05",
    "bootstrap_q50",
    "bootstrap_q95",
    "reference_cr",
    "reference_minus_q50",
    "abs_ref_minus_q50",
    "relative_ref_minus_q50",
    "reference_inside_ci90",
    "externally_available",
    "externally_usable",
    "weakly_constrained",
    "status",
]

available_table_cols = [
    c for c in table_cols if c in df.columns
]

df[
    available_table_cols
].sort_values(
    ["alpha_target", "Mach"]
).to_csv(
    OUT_TABLE,
    index=False,
)


summary_df = pd.DataFrame(
    [
        {
            "n_planned": n_planned,
            "n_available": n_available,
            "n_usable": n_usable,
            "n_weak": n_weak,
            "n_unavailable": n_unavailable,
            "n_inside_ci90_usable": n_inside_usable,
            "n_outside_ci90_usable": n_outside_usable,
            "median_abs_discrepancy": float(
                np.nanmedian(abs_disc)
            ),
            "p95_abs_discrepancy": float(
                np.nanquantile(abs_disc, 0.95)
            ),
            "max_abs_discrepancy": float(
                np.nanmax(abs_disc)
            ),
            "median_relative_discrepancy": float(
                np.nanmedian(rel_disc)
            ),
            "p95_relative_discrepancy": float(
                np.nanquantile(rel_disc, 0.95)
            ),
            "max_relative_discrepancy": float(
                np.nanmax(rel_disc)
            ),
        }
    ]
)

summary_df.to_csv(
    OUT_SUMMARY,
    index=False,
)


# =============================================================================
# Console audit
# =============================================================================

print("=" * 86)
print("INDEPENDENT BLUMEN c_r CONSISTENCY AUDIT")
print("=" * 86)

print(f"Planned coordinates       : {n_planned}")
print(f"Externally available      : {n_available}")
print(f"Externally usable         : {n_usable}")
print(f"Weakly constrained        : {n_weak}")
print(f"Unavailable               : {n_unavailable}")
print(
    f"Inside nominal CI90       : "
    f"{n_inside_usable}/{n_usable} usable points"
)

print()
print("USABLE-POINT DISCREPANCIES")

print(
    "Absolute discrepancy:"
    f"\n  median = {np.nanmedian(abs_disc):.6e}"
    f"\n  P95    = {np.nanquantile(abs_disc, 0.95):.6e}"
    f"\n  max    = {np.nanmax(abs_disc):.6e}"
)

print(
    "Relative discrepancy:"
    f"\n  median = {np.nanmedian(rel_disc):.3%}"
    f"\n  P95    = {np.nanquantile(rel_disc, 0.95):.3%}"
    f"\n  max    = {np.nanmax(rel_disc):.3%}"
)

print()
print("POINTWISE STATUS")

display_cols = [
    "Mach",
    "alpha_target",
    "status",
    "bootstrap_valid_fraction",
    "bootstrap_q50",
    "bootstrap_q05",
    "bootstrap_q95",
    "reference_cr",
    "abs_ref_minus_q50",
    "relative_ref_minus_q50",
    "inside_ci90",
]

print(
    df[display_cols]
    .sort_values(["alpha_target", "Mach"])
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6e}",
    )
)


# =============================================================================
# Publication style
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9.3,
        "axes.labelsize": 9.8,
        "axes.titlesize": 10.3,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.4,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# =============================================================================
# Colors / markers
# =============================================================================

COLOR_REF = "#111111"
COLOR_BLUMEN = "#0072B2"
COLOR_TENSION = "#D55E00"
COLOR_UNAVAILABLE = "0.55"

MARKER_REF = "D"
MARKER_BLUMEN = "o"


# =============================================================================
# Determine common y range
# =============================================================================

y_values = []

for _, row in df.iterrows():
    ref = row["reference_cr"]

    if np.isfinite(ref):
        y_values.append(float(ref))

    if row["available"]:
        for col in [
            "bootstrap_q05",
            "bootstrap_q50",
            "bootstrap_q95",
        ]:
            value = row[col]

            if np.isfinite(value):
                y_values.append(float(value))

if not y_values:
    raise RuntimeError("No finite c_r values found for plotting.")

y_min = min(y_values)
y_max = max(y_values)
y_span = y_max - y_min

if y_span <= 0:
    y_span = 0.05

ylim = (
    y_min - 0.10 * y_span,
    y_max + 0.12 * y_span,
)


# =============================================================================
# Figure
# =============================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(10.6, 3.75),
    sharey=True,
)


# =============================================================================
# Each alpha slice
# =============================================================================

for idx, (ax, alpha) in enumerate(
    zip(axes, alphas)
):
    group = (
        df[
            np.isclose(
                df["alpha_target"],
                alpha,
            )
        ]
        .sort_values("Mach")
        .copy()
    )

    M = group["Mach"].to_numpy(dtype=float)
    ref = group["reference_cr"].to_numpy(dtype=float)

    # -----------------------------------------------------------------
    # Classical reference
    # -----------------------------------------------------------------

    # Thin connection only as a visual guide across the three
    # predeclared Mach values.
    ax.plot(
        M,
        ref,
        color=COLOR_REF,
        linewidth=1.0,
        alpha=0.55,
        zorder=1,
    )

    ax.scatter(
        M,
        ref,
        marker=MARKER_REF,
        s=43,
        facecolor=COLOR_REF,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )

    # -----------------------------------------------------------------
    # Blumen reconstruction
    # -----------------------------------------------------------------

    for _, row in group.iterrows():
        m = float(row["Mach"])

        # Unavailable external reconstruction
        if not row["available"]:
            # Status marker only: no external c_r reconstruction exists
            # at this predeclared validation coordinate.  It is placed
            # near the lower edge so that it cannot be interpreted as
            # an estimated c_r value.
            unavailable_y = (
                ylim[0]
                + 0.055 * (ylim[1] - ylim[0])
            )

            ax.scatter(
                m,
                unavailable_y,
                marker="x",
                s=55,
                color=COLOR_UNAVAILABLE,
                linewidth=1.5,
                zorder=8,
            )

            ax.annotate(
                "unavailable",
                xy=(m, unavailable_y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color=COLOR_UNAVAILABLE,
            )

            continue

        q05 = float(row["bootstrap_q05"])
        q50 = float(row["bootstrap_q50"])
        q95 = float(row["bootstrap_q95"])

        if not (
            np.isfinite(q05)
            and np.isfinite(q50)
            and np.isfinite(q95)
        ):
            raise RuntimeError(
                f"Available row at M={m}, alpha={alpha} "
                "has non-finite bootstrap quantiles."
            )

        lower = q50 - q05
        upper = q95 - q50

        # Weak point = open blue marker
        if row["weak"]:
            face = "white"
            edge = COLOR_BLUMEN
        else:
            face = COLOR_BLUMEN
            edge = "white"

        ax.errorbar(
            m,
            q50,
            yerr=np.array([[lower], [upper]]),
            fmt=MARKER_BLUMEN,
            markersize=6.0,
            markerfacecolor=face,
            markeredgecolor=edge,
            markeredgewidth=0.8,
            ecolor=COLOR_BLUMEN,
            elinewidth=1.1,
            capsize=3.0,
            capthick=1.0,
            zorder=4,
        )

        # Usable reference outside nominal bootstrap interval:
        # unobtrusive orange ring around the classical-reference diamond.
        if (
            row["usable"]
            and not row["inside_ci90"]
        ):
            ax.scatter(
                m,
                float(row["reference_cr"]),
                marker="o",
                s=92,
                facecolors="none",
                edgecolors=COLOR_TENSION,
                linewidths=1.1,
                zorder=8,
            )

    # -----------------------------------------------------------------
    # Axes
    # -----------------------------------------------------------------

    ax.set_xlim(
        machs.min() - 0.025,
        machs.max() + 0.025,
    )

    ax.set_ylim(*ylim)

    ax.set_xticks(machs)

    ax.set_xlabel(r"Mach number $M$")

    panel_letter = chr(ord("a") + idx)

    ax.set_title(
        rf"({panel_letter}) $\alpha={alpha:.2f}$",
        loc="left",
        pad=7,
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.50,
    )

    ax.set_axisbelow(True)

axes[0].set_ylabel(
    r"Real phase velocity $c_r$"
)


# =============================================================================
# Shared legend
# =============================================================================

legend_handles = [
    Line2D(
        [0],
        [0],
        marker=MARKER_REF,
        linestyle="-",
        color=COLOR_REF,
        linewidth=1.0,
        markerfacecolor=COLOR_REF,
        markeredgecolor="white",
        markersize=6,
        label="Final classical reference",
    ),
    Line2D(
        [0],
        [0],
        marker=MARKER_BLUMEN,
        linestyle="none",
        markerfacecolor=COLOR_BLUMEN,
        markeredgecolor="white",
        markersize=6,
        label=r"Blumen bootstrap median ($q_{50}$)",
    ),
    Line2D(
        [0],
        [0],
        marker=MARKER_BLUMEN,
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor=COLOR_BLUMEN,
        markersize=6,
        label="Weakly constrained reconstruction",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor=COLOR_TENSION,
        markeredgewidth=1.2,
        markersize=8,
        label="Reference outside nominal $q_{05}$--$q_{95}$ interval",
    ),
    Line2D(
        [0],
        [0],
        marker="x",
        linestyle="none",
        color=COLOR_UNAVAILABLE,
        markersize=6,
        markeredgewidth=1.3,
        label="External reconstruction unavailable",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.075),
    ncol=3,
    frameon=False,
    columnspacing=1.6,
    handletextpad=0.65,
    labelspacing=0.8,
)


# =============================================================================
# Small note
# =============================================================================

fig.text(
    0.5,
    0.025,
    (
        r"Error bars show the nominal bootstrap "
        r"$q_{05}$--$q_{95}$ digitization/interpolation range; "
        r"they are not calibrated statistical uncertainty intervals."
    ),
    ha="center",
    va="bottom",
    fontsize=7.7,
    color="0.30",
)


# =============================================================================
# Layout
# =============================================================================

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    top=0.91,
    bottom=0.34,
    wspace=0.10,
)


# =============================================================================
# Save
# =============================================================================

fig.savefig(
    OUT_PNG,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print()
print(f"Saved PNG     : {OUT_PNG}")
print(f"Saved PDF     : {OUT_PDF}")
print(f"Saved table   : {OUT_TABLE}")
print(f"Saved summary : {OUT_SUMMARY}")
