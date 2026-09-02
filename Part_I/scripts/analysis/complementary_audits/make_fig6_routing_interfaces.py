from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SRC = Path(
    "assets/pinn_subsonic/article_work/"
    "p1d_routing_continuity/"
    "p1d_real_routing_audit.csv"
)

OUTDIR = Path("assets/section4_results")
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "Fig_atlas_overlap_consistency_N340.png"
PDF = OUTDIR / "Fig_atlas_overlap_consistency_N340.pdf"

df = pd.read_csv(SRC)

required = [
    "boundary_id",
    "samepoint_ci_abs_diff",
    "samepoint_p_overlap",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(
        f"Missing columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

# The audit contains several epsilon evaluations for each boundary.
# The same-point quantities are identical, so keep one row/interface.
d = (
    df[required]
    .dropna()
    .drop_duplicates(subset="boundary_id")
    .reset_index(drop=True)
)

if len(d) != 473:
    raise RuntimeError(
        f"Expected 473 unique routing interfaces, found {len(d)}"
    )

dc = d["samepoint_ci_abs_diff"].to_numpy(float)
overlap = d["samepoint_p_overlap"].to_numpy(float)
defect = 1.0 - overlap

# ------------------------------------------------------------
# Sanity check against manuscript values
# ------------------------------------------------------------
stats = {
    "dc_median": np.median(dc),
    "dc_p95": np.quantile(dc, 0.95),
    "dc_max": np.max(dc),
    "ov_median": np.median(overlap),
    "ov_p05": np.quantile(overlap, 0.05),
    "ov_min": np.min(overlap),
}

print("=" * 80)
print("FIGURE 6 — ROUTING INTERFACE CHECK")
print("=" * 80)
print("Unique interfaces :", len(d))
print(
    "delta ci          : "
    f"median={stats['dc_median']:.6e} "
    f"p95={stats['dc_p95']:.6e} "
    f"max={stats['dc_max']:.6e}"
)
print(
    "pressure overlap  : "
    f"median={stats['ov_median']:.6f} "
    f"p05={stats['ov_p05']:.6f} "
    f"min={stats['ov_min']:.6f}"
)
print("|delta ci| > 1e-3 :", np.sum(dc > 1e-3))
print("|delta ci| > 2e-3 :", np.sum(dc > 2e-3))
print("overlap < 0.98    :", np.sum(overlap < 0.98))
print("overlap < 0.95    :", np.sum(overlap < 0.95))

# Strong sanity checks against retained manuscript values
assert np.isclose(stats["dc_median"], 1.794397e-4, rtol=5e-3)
assert np.isclose(stats["dc_p95"], 1.578631e-3, rtol=5e-3)
assert np.isclose(stats["dc_max"], 2.985744e-3, rtol=5e-3)
assert np.isclose(stats["ov_median"], 0.999271, rtol=1e-4)
assert np.sum(dc > 1e-3) == 55
assert np.sum(dc > 2e-3) == 6
assert np.sum(overlap < 0.98) == 21
assert np.sum(overlap < 0.95) == 0

# ------------------------------------------------------------
# ECDF helper
# ------------------------------------------------------------
def ecdf(x):
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

# Panel A: eigenvalue jump
x, y = ecdf(dc)

axes[0].plot(x, y, linewidth=2)
axes[0].axvline(
    1e-3,
    linestyle="--",
    linewidth=1.2,
    label=r"$|\Delta c_i|=10^{-3}$",
)
axes[0].axvline(
    2e-3,
    linestyle=":",
    linewidth=1.2,
    label=r"$|\Delta c_i|=2\times10^{-3}$",
)

axes[0].set_xscale("log")
axes[0].set_xlabel(r"Eigenvalue jump $|\Delta c_i|$")
axes[0].set_ylabel("Fraction of interfaces")
axes[0].set_title("(a) Eigenvalue prediction")
axes[0].grid(alpha=0.25)
axes[0].legend(fontsize=9)

# Panel B: pressure-mode disagreement
x, y = ecdf(defect)

axes[1].plot(x, y, linewidth=2)
axes[1].axvline(
    0.02,
    linestyle="--",
    linewidth=1.2,
    label=r"$\mathcal{O}_p=0.98$",
)

axes[1].set_xscale("log")
axes[1].set_xlabel(r"Pressure overlap defect $1-\mathcal{O}_p$")
axes[1].set_ylabel("Fraction of interfaces")
axes[1].set_title("(b) Pressure mode")
axes[1].grid(alpha=0.25)
axes[1].legend(fontsize=9)

fig.suptitle(
    "Consistency across the 473 routing interfaces",
    fontsize=13,
)

fig.tight_layout()

fig.savefig(PNG, dpi=300, bbox_inches="tight")
fig.savefig(PDF, bbox_inches="tight")

print()
print("WROTE:", PNG)
print("WROTE:", PDF)
