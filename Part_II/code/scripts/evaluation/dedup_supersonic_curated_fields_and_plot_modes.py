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

fields = pd.read_csv(D / "supersonic_reference_core_local_modal_fields_CURATED.csv", low_memory=False)

priority = [
    "supersonic_reference_core_local_modal_fields_M140_branch_guided",
    "supersonic_shooting_ci_alpha_spectral_continuation_core_fields",
    "supersonic_shooting_ci_alpha_continuation_core_fields",
]

rows = []

for (M, a), sub in fields.groupby(["Mach", "alpha"], sort=True):
    sub = sub.copy()

    chosen = None
    for token in priority:
        ss = sub[sub["source_fields_csv"].astype(str).str.contains(token, regex=False)]
        if not ss.empty:
            chosen = ss
            break

    if chosen is None:
        # fallback: source with most distinct y
        counts = (
            sub.groupby("source_fields_csv")["y"]
            .nunique()
            .sort_values(ascending=False)
        )
        chosen = sub[sub["source_fields_csv"] == counts.index[0]]

    chosen = chosen.sort_values("y").drop_duplicates("y", keep="first")
    rows.append(chosen)

out = pd.concat(rows, ignore_index=True).sort_values(["Mach", "alpha", "y"])
out_csv = D / "supersonic_reference_core_local_modal_fields_CURATED_SINGLE_SOURCE.csv"
out.to_csv(out_csv, index=False)

print("[OK] wrote", out_csv)
print("\nfield rows per point:")
print(out.groupby(["Mach","alpha"]).size().reset_index(name="n_y").to_string(index=False))
print("\nsource per point:")
print(out.groupby(["Mach","alpha"])["source_fields_csv"].first().reset_index().to_string(index=False))

pdf = OUT / "supersonic_modes_curated_single_source.pdf"

with PdfPages(pdf) as pp:
    groups = list(out.groupby(["Mach", "alpha"], sort=True))

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

            tag = f"M{int(round(M*100)):03d}_a{int(round(a*100000)):05d}"
            fig1, ax1 = plt.subplots(figsize=(8.5, 4.5))
            ax1.plot(y, p.real / norm, linewidth=1.1, label="Re(p)/max|p|")
            ax1.plot(y, np.abs(p) / norm, "--", linewidth=1.1, label="|p|/max|p|")
            ax1.axhline(0.0, linewidth=0.6)
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"M={M:.2f}, alpha={a:.5f}, ci={ci:.5g}, N={len(sub)}")
            ax1.set_xlabel("y")
            ax1.set_ylabel("normalized mode")
            ax1.legend()
            fig1.tight_layout()
            fig1.savefig(OUT / f"mode_single_source_{tag}.png", dpi=220)
            plt.close(fig1)

        axes[0].legend(fontsize=8)
        fig.tight_layout()
        pp.savefig(fig)
        plt.close(fig)

print("[OK] wrote", pdf)
