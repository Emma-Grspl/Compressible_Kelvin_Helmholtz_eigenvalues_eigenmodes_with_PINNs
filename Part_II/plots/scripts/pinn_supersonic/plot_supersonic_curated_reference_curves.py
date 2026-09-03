#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

D = Path("assets/classic_supersonic/validated_modal_points/rebuilt_aggregates_latest/curated_strict")
OUT = D / "reference_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

spec = pd.read_csv(D / "supersonic_reference_core_local_spectral_CURATED.csv")
mod = pd.read_csv(D / "supersonic_reference_core_local_modal_CURATED.csv")

metrics = [
    ("reference_ci", "ci"),
    ("reference_cr", "cr"),
    ("reference_omega_i", "omega_i = alpha ci"),
]

pdf = OUT / "supersonic_curated_reference_cr_ci_omega.pdf"

with PdfPages(pdf) as pp:
    for M in sorted(set(spec["Mach"].round(10)) | set(mod["Mach"].round(10))):
        for col, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(7.8, 4.8))

            s = spec[np.isclose(spec["Mach"], M)].sort_values("alpha")
            m = mod[np.isclose(mod["Mach"], M)].sort_values("alpha")

            if not s.empty:
                ax.plot(s["alpha"], s[col], "o-", label="spectral trusted", linewidth=1.5)

            if not m.empty:
                ax.plot(m["alpha"], m[col], "s--", label="modal trusted", linewidth=1.2)

            ax.set_title(f"Curated strict {ylabel}(alpha), M={M:g}")
            ax.set_xlabel("alpha")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()

            pp.savefig(fig)
            fig.savefig(OUT / f"{col}_M{int(round(M*100)):03d}.png", dpi=220)
            plt.close(fig)

print("[OK] wrote", pdf)
print("[OK] wrote PNGs in", OUT)
