from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SRC = Path(
    "assets/pinn_subsonic/article_work/"
    "p1i_physics_vs_anchors/"
    "p1i_pointwise_384.csv"
)

OUTDIR = Path("assets/section4_results")
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "SuppFig_holdout_anchors_only_vs_physics_N340.png"
PDF = OUTDIR / "SuppFig_holdout_anchors_only_vs_physics_N340.pdf"

df = pd.read_csv(SRC)

required = [
    "point_id",
    "e_joint",
    "e_anchor",
    "delta_e",
    "physics_better",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(
        f"Missing columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

d = df[required].copy()

for c in ["e_joint", "e_anchor", "delta_e"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")

d = d.dropna(subset=["e_joint", "e_anchor", "delta_e"]).reset_index(drop=True)

if len(d) != 384:
    raise RuntimeError(f"Expected 384 points, found {len(d)}")

joint = d["e_joint"].to_numpy(float)
anchor = d["e_anchor"].to_numpy(float)
delta = d["delta_e"].to_numpy(float)

# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------
median_joint = float(np.median(joint))
median_anchor = float(np.median(anchor))
median_delta = float(np.median(delta))
mean_delta = float(np.mean(delta))
fraction_joint_better = float(np.mean(delta < 0.0))

print("=" * 90)
print("FIGURE 7 — JOINT MODEL VS ANCHORS-ONLY")
print("=" * 90)

print("N points               :", len(d))
print("Median joint error     :", f"{median_joint:.10e}")
print("Median anchors error   :", f"{median_anchor:.10e}")
print("Median delta e         :", f"{median_delta:.10e}")
print("Mean delta e           :", f"{mean_delta:.10e}")
print(
    "Joint model better    :",
    f"{fraction_joint_better:.8f}",
    f"({100*fraction_joint_better:.2f}%)",
)

# Values retained in the manuscript
assert np.isclose(median_joint, 2.016787752276e-4, rtol=1e-5)
assert np.isclose(median_anchor, 2.093583231428e-4, rtol=1e-5)
assert np.isclose(median_delta, -2.1662220874660992e-6, rtol=1e-5)
assert np.isclose(fraction_joint_better, 0.6171875, atol=1e-12)

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 6.0))

x = anchor
y = joint

# Avoid issues on log axes if an exact zero ever appears.
positive = (x > 0.0) & (y > 0.0)
x = x[positive]
y = y[positive]

ax.scatter(
    x,
    y,
    s=26,
    alpha=0.72,
    edgecolors="none",
)

vmin = min(x.min(), y.min())
vmax = max(x.max(), y.max())

lo = 10 ** np.floor(np.log10(vmin))
hi = 10 ** np.ceil(np.log10(vmax))

grid = np.logspace(
    np.log10(lo),
    np.log10(hi),
    300,
)

ax.plot(
    grid,
    grid,
    linestyle="--",
    linewidth=1.4,
    label="Equal error",
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)

ax.set_xlabel(
    r"Anchors-only absolute error "
    r"$|c_i^{\mathrm{anchor}}-c_i^{\mathrm{ref}}|$"
)

ax.set_ylabel(
    r"Joint-model absolute error "
    r"$|c_i^{\mathrm{joint}}-c_i^{\mathrm{ref}}|$"
)

ax.set_title(
    "Independent 384-point holdout"
)

ax.grid(alpha=0.25)
ax.legend(loc="lower right", fontsize=9)

text = (
    f"Joint better: {100*fraction_joint_better:.1f}%\n"
    f"Median joint: {median_joint:.2e}\n"
    f"Median anchors-only: {median_anchor:.2e}"
)

ax.text(
    0.04,
    0.96,
    text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9.5,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        alpha=0.90,
    ),
)

fig.tight_layout()

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF,
    bbox_inches="tight",
)

print()
print("WROTE:", PNG)
print("WROTE:", PDF)
