#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


D = Path("assets/classic_supersonic/validated_modal_points/rebuilt_aggregates_latest")
OUT = D / "curated_strict"
OUT.mkdir(parents=True, exist_ok=True)


def as_bool(s):
    return str(s).strip().lower() in {"true", "1", "yes", "y"}


def point_key(df):
    return list(zip(df["Mach"].round(10), df["alpha"].round(10)))


spectral = pd.read_csv(D / "supersonic_reference_core_local_spectral_REBUILT.csv")
modal = pd.read_csv(D / "supersonic_reference_core_local_modal_REBUILT.csv")
fields = pd.read_csv(D / "supersonic_reference_core_local_modal_fields_REBUILT.csv", low_memory=False)

spectral["trusted_spectral_bool"] = spectral["trusted_spectral"].map(as_bool)
modal["trusted_modal_bool"] = modal["trusted_modal"].map(as_bool)

spectral_c = spectral[spectral["trusted_spectral_bool"]].copy()
modal_c = modal[modal["trusted_modal_bool"]].copy()

# Sécurité supplémentaire : rejeter les ci de grille manifestes s'ils restent.
bad_grid_ci = {0.015, 0.025, 0.04, 0.055, 0.07, 0.085}
for name, df in [("spectral", spectral_c), ("modal", modal_c)]:
    bad = df[df["reference_ci"].round(6).isin(bad_grid_ci)]
    if len(bad):
        print(f"[WARN] {name}: suspicious grid-ci rows still present:")
        print(bad[["Mach", "alpha", "reference_cr", "reference_ci", "source_label"]].to_string(index=False))

keys = set(point_key(modal_c))
fields_c = fields[
    list(zip(fields["Mach"].round(10), fields["alpha"].round(10))).count if False else
    fields.apply(lambda r: (round(float(r["Mach"]), 10), round(float(r["alpha"]), 10)) in keys, axis=1)
].copy()

spectral_c = spectral_c.sort_values(["Mach", "alpha"])
modal_c = modal_c.sort_values(["Mach", "alpha"])
fields_c = fields_c.sort_values(["Mach", "alpha", "y"])

spectral_c.to_csv(OUT / "supersonic_reference_core_local_spectral_CURATED.csv", index=False)
modal_c.to_csv(OUT / "supersonic_reference_core_local_modal_CURATED.csv", index=False)
fields_c.to_csv(OUT / "supersonic_reference_core_local_modal_fields_CURATED.csv", index=False)

print("[OK] curated spectral:", len(spectral_c))
print("[OK] curated modal:", len(modal_c))
print("[OK] curated field rows:", len(fields_c))

print("\nCurated modal points:")
print(modal_c[["Mach", "alpha", "reference_cr", "reference_ci", "source_label"]].to_string(index=False))

# ci curves
with PdfPages(OUT / "supersonic_ci_curated_strict.pdf") as pp:
    for M in sorted(set(spectral_c["Mach"].round(10)) | set(modal_c["Mach"].round(10))):
        fig, ax = plt.subplots(figsize=(7.6, 4.8))

        s = spectral_c[np.isclose(spectral_c["Mach"], M)].sort_values("alpha")
        m = modal_c[np.isclose(modal_c["Mach"], M)].sort_values("alpha")

        if not s.empty:
            ax.plot(s["alpha"], s["reference_ci"], "o-", label="spectral trusted", linewidth=1.4)

        if not m.empty:
            ax.plot(m["alpha"], m["reference_ci"], "s--", label="modal trusted", linewidth=1.2)

        ax.set_title(f"Curated strict ci(alpha), M={M:g}")
        ax.set_xlabel("alpha")
        ax.set_ylabel("ci")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        pp.savefig(fig)
        fig.savefig(OUT / f"ci_curated_M{int(round(M*100)):03d}.png", dpi=220)
        plt.close(fig)

print("[OK] wrote", OUT / "supersonic_ci_curated_strict.pdf")

# modes curated
with PdfPages(OUT / "supersonic_modes_curated_strict.pdf") as pp:
    groups = list(fields_c.groupby(["Mach", "alpha"], sort=True))

    for i in range(0, len(groups), 6):
        chunk = groups[i:i+6]
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        axes = axes.ravel()

        for ax in axes:
            ax.axis("off")

        for ax, ((M, a), sub) in zip(axes, chunk):
            ax.axis("on")
            sub = sub.sort_values("y")

            y = sub["y"].to_numpy(float)
            p = sub["p_real"].to_numpy(float) + 1j * sub["p_imag"].to_numpy(float)
            norm = np.nanmax(np.abs(p))

            if not np.isfinite(norm) or norm <= 0:
                continue

            ci = sub["reference_ci"].iloc[0]
            ax.plot(y, p.real / norm, linewidth=1.0, label="Re(p)/max|p|")
            ax.plot(y, np.abs(p) / norm, "--", linewidth=1.0, label="|p|/max|p|")
            ax.axhline(0.0, linewidth=0.6)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"M={M:.2f}, alpha={a:.5f}\nci={ci:.5g}, N={len(sub)}", fontsize=9)
            ax.set_xlabel("y")
            ax.set_ylabel("normalized mode")

        axes[0].legend(fontsize=8)
        fig.tight_layout()
        pp.savefig(fig)
        plt.close(fig)

print("[OK] wrote", OUT / "supersonic_modes_curated_strict.pdf")
