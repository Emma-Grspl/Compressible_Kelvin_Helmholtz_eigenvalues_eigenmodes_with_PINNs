#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "assets"
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUTDIR / "Fig_ci_error_heatmap_PINN_GEP_vs_classical_N340.png"
PDF_OUT = OUTDIR / "Fig_ci_error_heatmap_PINN_GEP_vs_classical_N340.pdf"


def pick_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Impossible de trouver map_1000_ci_checks.csv.\n"
        "Chemins testés :\n" + "\n".join(str(p) for p in candidates)
    )


def pick_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    raise RuntimeError(
        f"Aucune des colonnes attendues {names} n'a été trouvée.\n"
        f"Colonnes disponibles : {list(df.columns)}"
    )


INPUT = pick_existing([
    ROOT / "assets" / "map_1000_ci_checks.csv",
    ROOT / "assets" / "section4_results" / "map_1000_ci_checks.csv",
    ROOT / "assets" / "pinn_subsonic" / "article_work" / "runtime_N340_frozen" / "map_1000_ci_checks.csv",
    ROOT / "assets" / "runtime_N340_frozen" / "map_1000_ci_checks.csv",
])

df = pd.read_csv(INPUT).copy()

mach_col = pick_col(df, ["Mach", "mach", "M"])
alpha_col = pick_col(df, ["alpha", "Alpha"])
ref_col = pick_col(df, ["classical_ci", "ci_ref", "ci_classic", "ci_classical"])
gep_col = pick_col(df, ["hybrid_ci", "scalar_gep_ci", "ci_gep", "ci_selected_gep", "ci_pinn_gep"])

M = pd.to_numeric(df[mach_col], errors="coerce").to_numpy()
alpha = pd.to_numeric(df[alpha_col], errors="coerce").to_numpy()
ci_ref = pd.to_numeric(df[ref_col], errors="coerce").to_numpy()
ci_gep = pd.to_numeric(df[gep_col], errors="coerce").to_numpy()

mask = np.isfinite(M) & np.isfinite(alpha) & np.isfinite(ci_ref) & np.isfinite(ci_gep)
M = M[mask]
alpha = alpha[mask]
ci_ref = ci_ref[mask]
ci_gep = ci_gep[mask]

# erreur régularisée comme dans le papier, exprimée en %
scaled_err_percent = 100.0 * np.abs(ci_gep - ci_ref) / np.maximum(np.abs(ci_ref), 0.05)


fig, ax = plt.subplots(figsize=(10.5, 7.8))

tri = mtri.Triangulation(M, alpha)

levels = np.linspace(0.0, 5.0, 31)
cont = ax.tricontourf(
    tri,
    np.clip(scaled_err_percent, 0.0, 5.0),
    levels=levels,
    vmin=0.0,
    vmax=5.0,
    cmap="viridis",
    extend="max",
)

# frontière neutre
mline = np.linspace(0.0, 0.999, 600)
aline = np.sqrt(np.clip(1.0 - mline**2, 0.0, None))
ax.plot(mline, aline, "--", linewidth=1.6, label="Neutral boundary")

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel(r"Mach number $M$")
ax.set_ylabel(r"Wavenumber $\alpha$")
ax.set_title("Selected GEP spectral error")

cbar = fig.colorbar(cont, ax=ax)
ticks = [0, 1, 2, 3, 4, 5]
cbar.set_ticks(ticks)
cbar.set_ticklabels(["0", "1", "2", "3", "4", "5"])
cbar.set_label("Scaled spectral error (%)")

ax.legend(loc="upper left", frameon=True)

fig.tight_layout()
fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight")
fig.savefig(PDF_OUT, bbox_inches="tight")
plt.close(fig)

print("=" * 90)
print("FIGURE REGENERATED")
print("=" * 90)
print("Input CSV :", INPUT)
print("PNG out   :", PNG_OUT)
print("PDF out   :", PDF_OUT)
print()
print(f"n points            : {len(scaled_err_percent)}")
print(f"median error (%)    : {np.median(scaled_err_percent):.6f}")
print(f"p95 error (%)       : {np.quantile(scaled_err_percent, 0.95):.6f}")
print(f"max error (%)       : {np.max(scaled_err_percent):.6f}")
print(f"points > 1%         : {np.sum(scaled_err_percent > 1.0)}")
print(f"points > 5%         : {np.sum(scaled_err_percent > 5.0)}")
