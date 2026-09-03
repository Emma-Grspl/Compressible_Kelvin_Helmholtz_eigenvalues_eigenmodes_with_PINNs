#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from classic_supersonic_reference.solver.mstab17_supersonic_solver import Mstab17SupersonicSolver
from classic_supersonic_reference.validation.modal_reconstruction import reconstruct_from_solver


FIELD_SPECS = [
    ("p", "p_real", "p_imag"),
    ("rho", "rho_real", "rho_imag"),
    ("v", "v_real", "v_imag"),
    ("u", "u_real", "u_imag"),
]


MAX_STEPS = [
    ("inf", float("inf")),
    ("2", 2.0),
    ("1", 1.0),
    ("0p5", 0.5),
    ("0p25", 0.25),
]


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Mach" not in df.columns and "M" in df.columns:
        df = df.rename(columns={"M": "Mach"})
    for c in ["Mach", "alpha", "cr", "ci"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Mach", "alpha", "cr", "ci"]).sort_values(["Mach", "alpha"]).reset_index(drop=True)


def representative_subset(df: pd.DataFrame, n_each: int = 5) -> pd.DataFrame:
    rows = []
    for M, g in df.groupby("Mach"):
        g = g.sort_values("alpha").reset_index(drop=True)
        if len(g) <= n_each:
            rows.append(g)
            continue
        idx = np.linspace(0, len(g) - 1, n_each).round().astype(int)
        rows.append(g.iloc[idx].drop_duplicates(["Mach", "alpha"]))
    return pd.concat(rows, ignore_index=True).sort_values(["Mach", "alpha"]).reset_index(drop=True)


def exact_log_amplitude_ms(alpha, mach, cr, ci, max_y_limit, mapping_scale, max_step):
    solver = Mstab17SupersonicSolver(
        alpha=alpha,
        Mach=mach,
        match_y=1.0,
        use_mapping=True,
        mapping_scale=mapping_scale,
        min_y_limit=10.0,
        max_y_limit=max_y_limit,
        y_limit_factor=10.0,
        max_step=max_step,
    )

    sol_left, _, sol_right_full, y_limit = solver.get_trajectories(
        cr, ci, ln_p_start_right=0.0
    )

    if not (sol_left.success and sol_right_full.success):
        return np.nan, np.nan, float(y_limit)

    target_y = solver.amplitude_match_y
    ln_left = solver._interp_component(target_y, sol_left, 2)
    ln_right_zero = solver._interp_component(target_y, sol_right_full, 2)

    ln_required = float(ln_left - ln_right_zero)
    stage2 = float(solver.stage2_objective(ln_required, cr, ci))

    return ln_required, stage2, float(y_limit)


def reconstruct_df(alpha, mach, cr, ci, ln_p_start_right, max_y_limit, mapping_scale, max_step):
    solver = Mstab17SupersonicSolver(
        alpha=alpha,
        Mach=mach,
        match_y=1.0,
        use_mapping=True,
        mapping_scale=mapping_scale,
        min_y_limit=10.0,
        max_y_limit=max_y_limit,
        y_limit_factor=10.0,
        max_step=max_step,
    )
    fields = reconstruct_from_solver(
        solver,
        cr=cr,
        ci=ci,
        ln_p_start_right=ln_p_start_right,
    )

    y = np.asarray(fields["y"], dtype=float)

    out = pd.DataFrame({
        "Mach": mach,
        "alpha": alpha,
        "cr": cr,
        "ci": ci,
        "omega_i": alpha * ci,
        "y": y,
    })

    for name in ["p", "rho", "u", "v"]:
        z = np.asarray(fields[name])
        out[f"{name}_real"] = np.real(z)
        out[f"{name}_imag"] = np.imag(z)

    return out.sort_values("y").reset_index(drop=True)


def zfield(df, re_col, im_col):
    return (
        pd.to_numeric(df[re_col], errors="coerce").to_numpy(float)
        + 1j * pd.to_numeric(df[im_col], errors="coerce").to_numpy(float)
    )


def interp_complex(y_src, z_src, y_tgt):
    zr = np.interp(y_tgt, y_src, np.real(z_src))
    zi = np.interp(y_tgt, y_src, np.imag(z_src))
    return zr + 1j * zi


def aligned_rel_l2(z_ref, z):
    denom = np.linalg.norm(z_ref)
    if denom <= 0 or not np.isfinite(denom):
        return np.inf
    phase = np.vdot(z, z_ref)
    if abs(phase) > 0:
        z = z * phase / abs(phase)
    return float(np.linalg.norm(z_ref - z) / denom)


def compare_to_ref(ref_df, df, field_name, re_col, im_col):
    y_ref = ref_df["y"].to_numpy(float)
    y = df["y"].to_numpy(float)

    z_ref = zfield(ref_df, re_col, im_col)
    z = zfield(df, re_col, im_col)

    common_min = max(float(np.nanmin(y_ref)), float(np.nanmin(y)))
    common_max = min(float(np.nanmax(y_ref)), float(np.nanmax(y)))

    mask_common = (y_ref >= common_min) & (y_ref <= common_max)
    y_common = y_ref[mask_common]
    z_ref_common = z_ref[mask_common]
    z_on_ref = interp_complex(y, z, y_common)

    amp = np.abs(z_ref_common)
    peak = np.nanmax(amp)

    row = {}
    for thr in [0.10, 0.05, 0.02, 0.01]:
        mask = amp >= thr * peak
        if mask.sum() >= 20:
            row[f"{field_name}_rel_l2_amp_ge_{thr:g}"] = aligned_rel_l2(z_ref_common[mask], z_on_ref[mask])
        else:
            row[f"{field_name}_rel_l2_amp_ge_{thr:g}"] = np.nan

    tail = (amp >= 1e-3 * peak) & (amp < 2e-2 * peak)
    if tail.sum() >= 20:
        row[f"{field_name}_rel_l2_tail_0p001_0p02"] = aligned_rel_l2(z_ref_common[tail], z_on_ref[tail])
    else:
        row[f"{field_name}_rel_l2_tail_0p001_0p02"] = np.nan

    return row


def mode_xlim(y, amp, threshold=0.05, min_half=80):
    peak = np.nanmax(amp)
    if not np.isfinite(peak) or peak <= 0:
        return np.nanmin(y), np.nanmax(y)
    mask = amp >= threshold * peak
    if not np.any(mask):
        return np.nanmin(y), np.nanmax(y)
    half = max(float(np.nanmax(np.abs(y[mask]))), min_half)
    return -half, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path(
            "assets/classic_supersonic/scan_M18_M19_alpha_0_03_strict/"
            "refined_near_valid_branch/refined_near_valid_candidates.csv"
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "assets/classic_supersonic/scan_M18_M19_alpha_0_03_strict/"
            "refined_near_valid_branch/max_step_audit"
        ),
    )
    ap.add_argument("--representative", action="store_true")
    ap.add_argument("--n-each", type=int, default=5)
    ap.add_argument("--mapping-scale", type=float, default=3.0)
    ap.add_argument("--max-y-limit", type=float, default=2000.0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields_dir = args.output_dir / "fields_by_max_step"
    fields_dir.mkdir(exist_ok=True)

    cand = load_candidates(args.candidate_csv)
    if args.representative:
        cand = representative_subset(cand, n_each=args.n_each)

    rows = []
    field_cache = {}

    print("n points:", len(cand))

    for _, r in cand.iterrows():
        M = float(r["Mach"])
        a = float(r["alpha"])
        cr = float(r["cr"])
        ci = float(r["ci"])

        print(f"[point] M={M:.2f} alpha={a:.5f} cr={cr:.6f} ci={ci:.6f}", flush=True)

        for label, ms in MAX_STEPS:
            ln_amp, stage2, y_limit = exact_log_amplitude_ms(
                alpha=a,
                mach=M,
                cr=cr,
                ci=ci,
                max_y_limit=args.max_y_limit,
                mapping_scale=args.mapping_scale,
                max_step=ms,
            )

            df = reconstruct_df(
                alpha=a,
                mach=M,
                cr=cr,
                ci=ci,
                ln_p_start_right=ln_amp,
                max_y_limit=args.max_y_limit,
                mapping_scale=args.mapping_scale,
                max_step=ms,
            )

            df["max_step_label"] = label
            df["max_step"] = ms
            df["source"] = "max_step_audit_recomputed"
            df["validation_status"] = "max_step_audit_only_not_validated"

            fpath = fields_dir / f"M{M:.2f}_alpha{a:.5f}_maxstep_{label}_fields.csv"
            df.to_csv(fpath, index=False)

            field_cache[(M, a, label)] = df

            rows.append({
                "Mach": M,
                "alpha": a,
                "cr": cr,
                "ci": ci,
                "max_step_label": label,
                "max_step": ms,
                "ln_p_start_right": ln_amp,
                "stage2_mismatch": stage2,
                "y_limit": y_limit,
                "fields_file": str(fpath),
            })

            print(f"  max_step={label}: stage2={stage2:.2e}, y_limit={y_limit:.1f}", flush=True)

    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(args.output_dir / "max_step_runs.csv", index=False)

    # Comparaison au max_step le plus fin.
    ref_label = "0p25"
    comp_rows = []

    for _, r in cand.iterrows():
        M = float(r["Mach"])
        a = float(r["alpha"])

        ref = field_cache[(M, a, ref_label)]

        for label, _ in MAX_STEPS:
            if label == ref_label:
                continue

            df = field_cache[(M, a, label)]
            row = {
                "Mach": M,
                "alpha": a,
                "setting": f"maxstep_{label}",
                "ref_setting": f"maxstep_{ref_label}",
            }

            for fname, re_col, im_col in FIELD_SPECS:
                row.update(compare_to_ref(ref, df, fname, re_col, im_col))

            comp_rows.append(row)

    comp = pd.DataFrame(comp_rows)
    comp.to_csv(args.output_dir / "max_step_convergence_metrics.csv", index=False)

    # PDF overlay Re.
    pdf_path = args.output_dir / "max_step_real_oscillation_overlay.pdf"

    with PdfPages(pdf_path) as pdf:
        for _, r in cand.iterrows():
            M = float(r["Mach"])
            a = float(r["alpha"])

            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), dpi=180)
            axes = axes.ravel()

            for ax, (fname, re_col, im_col) in zip(axes, FIELD_SPECS):
                ref = field_cache[(M, a, ref_label)]
                y_ref = ref["y"].to_numpy(float)
                z_ref = zfield(ref, re_col, im_col)
                amp_ref = np.abs(z_ref)
                xlim = mode_xlim(y_ref, amp_ref, threshold=0.05)

                for label, _ in MAX_STEPS:
                    df = field_cache[(M, a, label)]
                    y = df["y"].to_numpy(float)
                    z = zfield(df, re_col, im_col)
                    scale = np.nanmax(np.abs(z))
                    if not np.isfinite(scale) or scale <= 0:
                        scale = 1.0
                    ax.plot(y, np.real(z / scale), linewidth=0.75, label=f"max_step={label}")

                ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
                ax.set_xlim(*xlim)
                ax.set_ylim(-1.05, 1.05)
                ax.grid(True, alpha=0.25, linestyle=":")
                ax.set_title(f"Re({fname}) / max|{fname}|")
                ax.set_xlabel("y")
                ax.legend(fontsize=6)

            fig.suptitle(f"MAX_STEP OVERLAY — M={M:.2f}, alpha={a:.5f}", fontsize=11)
            fig.tight_layout(rect=[0, 0.02, 1, 0.94])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    summary = {
        "status": "max_step_audit_complete",
        "n_points": int(len(cand)),
        "mapping_scale": args.mapping_scale,
        "max_y_limit": args.max_y_limit,
        "reference_max_step": ref_label,
        "max_steps": [x[0] for x in MAX_STEPS],
        "metrics_file": str(args.output_dir / "max_step_convergence_metrics.csv"),
        "pdf_file": str(pdf_path),
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

    print("\nQuantiles core amp >= 5%, compared to max_step=0.25")
    for fname, _, _ in FIELD_SPECS:
        col = f"{fname}_rel_l2_amp_ge_0.05"
        print(fname)
        print(comp[col].quantile([0.5, 0.9, 0.95, 0.99, 1.0]).to_string())

    print("\nPDF:", pdf_path)


if __name__ == "__main__":
    main()
