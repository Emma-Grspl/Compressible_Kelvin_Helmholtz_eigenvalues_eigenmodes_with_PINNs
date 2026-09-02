#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scripts.audits.audit_M18_M19_max_step_convergence_c6e9b8c068 import (
    load_candidates,
    representative_subset,
    exact_log_amplitude_ms,
    reconstruct_df,
    zfield,
    interp_complex,
    aligned_rel_l2,
    mode_xlim,
)

FIELD_SPECS = [
    ("p", "p_real", "p_imag"),
    ("rho", "rho_real", "rho_imag"),
    ("v", "v_real", "v_imag"),
    ("u", "u_real", "u_imag"),
]

ROOT = Path("assets/classic_supersonic/scan_M18_M19_alpha_0_03_strict/refined_near_valid_branch")
CAND = ROOT / "refined_near_valid_candidates.csv"
OUT = ROOT / "mapping_vs_box_audit"
FIELDS_DIR = OUT / "fields"
OUT.mkdir(parents=True, exist_ok=True)
FIELDS_DIR.mkdir(exist_ok=True)

cand = load_candidates(CAND)
cand = representative_subset(cand, n_each=5)

# Deux familles de tests :
# 1. mapping_scale varie, y_limit fixé.
# 2. y_limit varie, mapping_scale fixé.
SETTINGS = [
    ("map1p5_y2000", 1.5, 2000.0, "mapping_only"),
    ("map2_y2000",   2.0, 2000.0, "mapping_only"),
    ("map3_y2000",   3.0, 2000.0, "mapping_only"),
    ("map5_y2000",   5.0, 2000.0, "mapping_only"),

    ("map3_y1200",   3.0, 1200.0, "box_only"),
    ("map3_y1600",   3.0, 1600.0, "box_only"),
    ("map3_y2000b",  3.0, 2000.0, "box_only"),
    ("map3_y2400",   3.0, 2400.0, "box_only"),
]

REFS = {
    "mapping_only": "map3_y2000",
    "box_only": "map3_y2000b",
}

rows = []
cache = {}

print("n representative points:", len(cand))

for _, r in cand.iterrows():
    M = float(r["Mach"])
    a = float(r["alpha"])
    cr = float(r["cr"])
    ci = float(r["ci"])

    print(f"[point] M={M:.2f} alpha={a:.5f} cr={cr:.6f} ci={ci:.6f}", flush=True)

    for label, mapping_scale, y_limit, family in SETTINGS:
        ln_amp, stage2, actual_y = exact_log_amplitude_ms(
            alpha=a,
            mach=M,
            cr=cr,
            ci=ci,
            max_y_limit=y_limit,
            mapping_scale=mapping_scale,
            max_step=float("inf"),
        )

        df = reconstruct_df(
            alpha=a,
            mach=M,
            cr=cr,
            ci=ci,
            ln_p_start_right=ln_amp,
            max_y_limit=y_limit,
            mapping_scale=mapping_scale,
            max_step=float("inf"),
        )

        df["audit_setting"] = label
        df["mapping_scale"] = mapping_scale
        df["requested_y_limit"] = y_limit
        df["actual_y_limit"] = actual_y
        df["family"] = family

        fpath = FIELDS_DIR / f"M{M:.2f}_alpha{a:.5f}_{label}.csv"
        df.to_csv(fpath, index=False)

        cache[(M, a, label)] = df

        rows.append({
            "Mach": M,
            "alpha": a,
            "cr": cr,
            "ci": ci,
            "setting": label,
            "family": family,
            "mapping_scale": mapping_scale,
            "requested_y_limit": y_limit,
            "actual_y_limit": actual_y,
            "stage2": stage2,
            "fields_file": str(fpath),
        })

runs = pd.DataFrame(rows)
runs.to_csv(OUT / "runs.csv", index=False)

metric_rows = []

for _, r in cand.iterrows():
    M = float(r["Mach"])
    a = float(r["alpha"])

    for family in ["mapping_only", "box_only"]:
        ref_label = REFS[family]
        ref = cache[(M, a, ref_label)]

        y_ref = ref["y"].to_numpy(float)

        for label, mapping_scale, y_limit, fam in SETTINGS:
            if fam != family or label == ref_label:
                continue

            df = cache[(M, a, label)]
            y = df["y"].to_numpy(float)

            row = {
                "Mach": M,
                "alpha": a,
                "family": family,
                "setting": label,
                "ref_setting": ref_label,
                "mapping_scale": mapping_scale,
                "requested_y_limit": y_limit,
            }

            for fname, re_col, im_col in FIELD_SPECS:
                z_ref = zfield(ref, re_col, im_col)
                z = zfield(df, re_col, im_col)

                common_min = max(float(np.nanmin(y_ref)), float(np.nanmin(y)))
                common_max = min(float(np.nanmax(y_ref)), float(np.nanmax(y)))

                mask_common = (y_ref >= common_min) & (y_ref <= common_max)
                yy = y_ref[mask_common]
                zr = z_ref[mask_common]
                zz = interp_complex(y, z, yy)

                amp = np.abs(zr)
                peak = np.nanmax(amp)

                for thr in [0.10, 0.05, 0.02, 0.01]:
                    mask = amp >= thr * peak
                    if mask.sum() >= 20:
                        row[f"{fname}_rel_l2_amp_ge_{thr:g}"] = aligned_rel_l2(zr[mask], zz[mask])
                    else:
                        row[f"{fname}_rel_l2_amp_ge_{thr:g}"] = np.nan

                tail = (amp >= 1e-3 * peak) & (amp < 2e-2 * peak)
                if tail.sum() >= 20:
                    row[f"{fname}_tail_rel_l2_0p001_0p02"] = aligned_rel_l2(zr[tail], zz[tail])
                else:
                    row[f"{fname}_tail_rel_l2_0p001_0p02"] = np.nan

            metric_rows.append(row)

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(OUT / "mapping_vs_box_metrics.csv", index=False)

for family in ["mapping_only", "box_only"]:
    pdf_path = OUT / f"{family}_real_overlay.pdf"

    with PdfPages(pdf_path) as pdf:
        for _, r in cand.iterrows():
            M = float(r["Mach"])
            a = float(r["alpha"])

            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), dpi=180)
            axes = axes.ravel()

            family_settings = [s for s in SETTINGS if s[3] == family]
            ref = cache[(M, a, REFS[family])]

            for ax, (fname, re_col, im_col) in zip(axes, FIELD_SPECS):
                y_ref = ref["y"].to_numpy(float)
                z_ref = zfield(ref, re_col, im_col)
                amp_ref = np.abs(z_ref)
                ax.set_xlim(*mode_xlim(y_ref, amp_ref, threshold=0.05))

                for label, mapping_scale, y_limit, fam in family_settings:
                    df = cache[(M, a, label)]
                    y = df["y"].to_numpy(float)
                    z = zfield(df, re_col, im_col)

                    scale = np.nanmax(np.abs(z))
                    if not np.isfinite(scale) or scale <= 0:
                        scale = 1.0

                    ax.plot(y, np.real(z / scale), linewidth=0.75, label=label)

                ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
                ax.set_ylim(-1.05, 1.05)
                ax.grid(True, alpha=0.25, linestyle=":")
                ax.set_title(f"Re({fname}) / max|{fname}|")
                ax.set_xlabel("y")
                ax.legend(fontsize=6)

            fig.suptitle(f"{family} — M={M:.2f}, alpha={a:.5f}", fontsize=11)
            fig.tight_layout(rect=[0, 0.02, 1, 0.94])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("wrote:", pdf_path)

summary = {
    "status": "mapping_vs_box_audit_complete",
    "n_points": int(len(cand)),
    "runs": str(OUT / "runs.csv"),
    "metrics": str(OUT / "mapping_vs_box_metrics.csv"),
    "pdf_mapping_only": str(OUT / "mapping_only_real_overlay.pdf"),
    "pdf_box_only": str(OUT / "box_only_real_overlay.pdf"),
}

(OUT / "summary.json").write_text(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2))

print("\nCore amp >= 5% metrics by family/setting")
for field in ["p", "rho", "v", "u"]:
    col = f"{field}_rel_l2_amp_ge_0.05"
    print("\n", field)
    print(metrics.groupby(["family", "setting"])[col].describe()[["count", "mean", "50%", "max"]])
