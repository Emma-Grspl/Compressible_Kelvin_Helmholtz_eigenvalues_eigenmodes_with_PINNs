#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

indir = Path(
    "assets/classic_subsonic/convergence_box_resolution/"
    "article_test_a050_M050_exact_solver"
)

raw = indir / "subsonic_classical_convergence_raw.csv"
outdir = indir / "article_figures"
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw)

# Garder uniquement le vrai sweep L/n_scan, pas les lignes du sweep de tolérance.
if "sweep" in df.columns:
    df = df[df["sweep"].isna()].copy()

df = df[df["solver"].astype(str).eq("secondary_mstab17")].copy()
df = df[df["success"].astype(str).str.lower().isin(["true", "1"])].copy()

df["L"] = pd.to_numeric(df["L"], errors="coerce")
df["n_scan"] = pd.to_numeric(df["n_scan"], errors="coerce")
df["ci_abs_error_vs_robust"] = pd.to_numeric(df["ci_abs_error_vs_robust"], errors="coerce")

# C'EST LE SEUL CHANGEMENT DEMANDÉ : exclure L=80.
df = df[df["L"] != 80.0].copy()

# Éviter les doublons éventuels.
df = df.sort_values(["L", "n_scan"]).drop_duplicates(["L", "n_scan"], keep="first")

alpha = float(df["alpha"].iloc[0])
Mach = float(df["Mach"].iloc[0])

fig, ax = plt.subplots(figsize=(7.5, 5.0))

for L in sorted(df["L"].unique()):
    sub = df[df["L"] == L].sort_values("n_scan")
    ax.plot(
        sub["n_scan"],
        sub["ci_abs_error_vs_robust"],
        marker="o",
        linewidth=2,
        label=f"L={L:g}",
    )

ax.set_xscale("log")
ax.set_xlabel("n_scan")
ax.set_ylabel(r"$|c_i - c_{i,\mathrm{robust}}|$")
ax.set_title(f"secondary_mstab17, alpha={alpha:g}, M={Mach:g}")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

png = outdir / "secondary_mstab17_ci_error_vs_nscan_no_L80.png"
pdf = outdir / "secondary_mstab17_ci_error_vs_nscan_no_L80.pdf"

fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)

print("[OK] wrote")
print(png)
print(pdf)
