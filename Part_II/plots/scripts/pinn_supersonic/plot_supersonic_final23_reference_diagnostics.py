#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

D = Path("assets/classic_supersonic/validated_modal_points/rebuilt_aggregates_latest/curated_strict/final23")
OUT = D / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

spec = pd.read_csv(D / "supersonic_reference_core_local_spectral_FINAL23.csv")
mod = pd.read_csv(D / "supersonic_reference_core_local_modal_FINAL23.csv")
fld = pd.read_csv(D / "supersonic_reference_core_local_modal_fields_FINAL23.csv", low_memory=False)

metrics = [
    ("reference_ci", "ci"),
    ("reference_cr", "cr"),
    ("reference_omega_i", "omega_i = alpha ci"),
]

with PdfPages(OUT / "supersonic_final23_cr_ci_omega.pdf") as pp:
    for M in sorted(set(spec["Mach"].round(10)) | set(mod["Mach"].round(10))):
        for col, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(7.8, 4.8))
            s = spec[np.isclose(spec["Mach"], M)].sort_values("alpha")
            m = mod[np.isclose(mod["Mach"], M)].sort_values("alpha")
            ax.plot(s["alpha"], s[col], "o-", label="spectral final", linewidth=1.5)
            ax.plot(m["alpha"], m[col], "s--", label="modal final", linewidth=1.2)
            ax.set_title(f"Final23 {ylabel}(alpha), M={M:g}")
            ax.set_xlabel("alpha")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            pp.savefig(fig)
            fig.savefig(OUT / f"{col}_M{int(round(M*100)):03d}.png", dpi=220)
            plt.close(fig)

with PdfPages(OUT / "supersonic_final23_modes.pdf") as pp:
    groups = list(fld.groupby(["Mach", "alpha"], sort=True))

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

mod[["Mach", "alpha", "reference_cr", "reference_ci", "reference_omega_i", "line_id", "source_label"]].sort_values(["Mach", "alpha"]).to_csv(
    OUT / "final23_reference_table.csv",
    index=False,
)

print("[OK] wrote", OUT / "supersonic_final23_cr_ci_omega.pdf")
print("[OK] wrote", OUT / "supersonic_final23_modes.pdf")
print("[OK] wrote", OUT / "final23_reference_table.csv")
