#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classic_supersonic_reference.solver.mstab17_supersonic_solver import Mstab17SupersonicSolver
import sys as _sys
from pathlib import Path as _Path

_CAMPAIGNS_DIR = (
    _Path(__file__).resolve().parents[1]
    / "campaigns"
)

if str(_CAMPAIGNS_DIR) not in _sys.path:
    _sys.path.insert(
        0,
        str(_CAMPAIGNS_DIR),
    )

from scripts.audits.audit_scan_supersonic_M18_M19_strict_modal_validation import (

    exact_log_amplitude,
    reconstruct_dataframe,
    field_shape_metrics,
)


FIELD_SPECS = [
    ("p", "p_real", "p_imag"),
    ("rho", "rho_real", "rho_imag"),
    ("u", "u_real", "u_imag"),
    ("v", "v_real", "v_imag"),
]


SETTINGS = [
    # name, mapping_scale, search_y_limit, final_y_limit
    ("ms1p5_y1200", 1.5, 900.0, 1200.0),
    ("ms2_y1600", 2.0, 900.0, 1600.0),
    ("ms3_y2000", 3.0, 1200.0, 2000.0),
    ("ms5_y1600", 5.0, 900.0, 1600.0),
]


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["Mach", "alpha", "cr", "ci", "stage1_mismatch"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Mach", "alpha", "cr", "ci"])
    df = df.sort_values(["Mach", "alpha"]).reset_index(drop=True)
    return df


def solve_local(
    *,
    mach: float,
    alpha: float,
    cr0: float,
    ci0: float,
    mapping_scale: float,
    search_y_limit: float,
    cr_width: float,
    ci_width: float,
    grid_size: int,
    max_iter: int,
):
    cr_min = max(0.0, cr0 - cr_width)
    cr_max = cr0 + cr_width
    ci_min = max(1e-7, ci0 - ci_width)
    ci_max = max(ci_min * 1.01, ci0 + ci_width)

    solver = Mstab17SupersonicSolver(
        alpha=alpha,
        Mach=mach,
        match_y=1.0,
        use_mapping=True,
        mapping_scale=mapping_scale,
        min_y_limit=10.0,
        max_y_limit=search_y_limit,
        y_limit_factor=10.0,
    )

    cr, ci, stage1 = solver.solve_eigenvalue(
        cr_min=cr_min,
        cr_max=cr_max,
        ci_min=ci_min,
        ci_max=ci_max,
        max_iter=max_iter,
        grid_size=grid_size,
        constrain_to_initial_box=True,
        tol=1e-11,
    )

    return float(cr), float(ci), float(stage1)


def complex_field(df: pd.DataFrame, re_col: str, im_col: str) -> np.ndarray:
    return (
        pd.to_numeric(df[re_col], errors="coerce").to_numpy(dtype=float)
        + 1j * pd.to_numeric(df[im_col], errors="coerce").to_numpy(dtype=float)
    )


def interp_complex(y_src, z_src, y_tgt):
    zr = np.interp(y_tgt, y_src, np.real(z_src))
    zi = np.interp(y_tgt, y_src, np.imag(z_src))
    return zr + 1j * zi


def rel_l2_to_reference(ref_df: pd.DataFrame, df: pd.DataFrame) -> dict:
    out = {}

    y_ref = ref_df["y"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    common_min = max(float(np.nanmin(y_ref)), float(np.nanmin(y)))
    common_max = min(float(np.nanmax(y_ref)), float(np.nanmax(y)))

    for name, re_col, im_col in FIELD_SPECS:
        z_ref = complex_field(ref_df, re_col, im_col)
        z = complex_field(df, re_col, im_col)

        amp_ref = np.abs(z_ref)
        peak = float(np.nanmax(amp_ref))

        if not np.isfinite(peak) or peak <= 0:
            out[f"{name}_rel_l2_to_ref"] = np.inf
            continue

        mask = (
            (y_ref >= common_min)
            & (y_ref <= common_max)
            & (amp_ref >= 1e-3 * peak)
        )

        if mask.sum() < 30:
            mask = (y_ref >= common_min) & (y_ref <= common_max)

        yy = y_ref[mask]
        zr = z_ref[mask]
        zz = interp_complex(y, z, yy)

        denom = np.linalg.norm(zr)
        if denom <= 0 or not np.isfinite(denom):
            rel = np.inf
        else:
            # alignement de phase globale
            phase = np.vdot(zz, zr)
            if abs(phase) > 0:
                zz = zz * phase / abs(phase)
            rel = float(np.linalg.norm(zr - zz) / denom)

        out[f"{name}_rel_l2_to_ref"] = rel

    out["max_rel_l2_to_ref"] = float(
        np.nanmax([out[f"{name}_rel_l2_to_ref"] for name, _, _ in FIELD_SPECS])
    )
    return out


def run_one_setting(
    *,
    candidate,
    setting_name: str,
    mapping_scale: float,
    search_y_limit: float,
    final_y_limit: float,
    args,
):
    mach = float(candidate["Mach"])
    alpha = float(candidate["alpha"])
    cr0 = float(candidate["cr"])
    ci0 = float(candidate["ci"])

    cr, ci, stage1 = solve_local(
        mach=mach,
        alpha=alpha,
        cr0=cr0,
        ci0=ci0,
        mapping_scale=mapping_scale,
        search_y_limit=search_y_limit,
        cr_width=args.cr_width,
        ci_width=args.ci_width,
        grid_size=args.grid_size,
        max_iter=args.max_iter,
    )

    ln_amp, stage2, y_limit, amp_status = exact_log_amplitude(
        alpha=alpha,
        mach=mach,
        cr=cr,
        ci=ci,
        max_y_limit=final_y_limit,
        mapping_scale=mapping_scale,
    )

    fields = reconstruct_dataframe(
        alpha=alpha,
        mach=mach,
        cr=cr,
        ci=ci,
        ln_p_start_right=ln_amp,
        max_y_limit=final_y_limit,
        mapping_scale=mapping_scale,
    )

    metrics = field_shape_metrics(fields)

    row = {
        "Mach": mach,
        "alpha": alpha,
        "setting": setting_name,
        "mapping_scale": mapping_scale,
        "search_y_limit": search_y_limit,
        "final_y_limit": final_y_limit,
        "cr": cr,
        "ci": ci,
        "omega_i": alpha * ci,
        "stage1_mismatch": stage1,
        "stage2_mismatch_exact": stage2,
        "ln_p_start_right_exact": ln_amp,
        "actual_y_limit": y_limit,
        "amplitude_status": amp_status,
    }
    row.update(metrics)

    return row, fields


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
            "refined_near_valid_branch/convergence_audit"
        ),
    )
    ap.add_argument("--cr-width", type=float, default=0.008)
    ap.add_argument("--ci-width", type=float, default=0.003)
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=7)
    ap.add_argument("--max-points", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fields_by_setting").mkdir(exist_ok=True)

    candidates = load_candidates(args.candidate_csv)

    if args.max_points and args.max_points > 0:
        candidates = candidates.head(args.max_points)

    all_rows = []
    point_summary = []

    for idx, cand in candidates.iterrows():
        mach = float(cand["Mach"])
        alpha = float(cand["alpha"])

        print(f"[point {idx+1}/{len(candidates)}] M={mach:.2f} alpha={alpha:.5f}", flush=True)

        setting_rows = []
        setting_fields = {}

        for setting_name, mapping_scale, search_y_limit, final_y_limit in SETTINGS:
            try:
                row, fields = run_one_setting(
                    candidate=cand,
                    setting_name=setting_name,
                    mapping_scale=mapping_scale,
                    search_y_limit=search_y_limit,
                    final_y_limit=final_y_limit,
                    args=args,
                )

                fields["setting"] = setting_name
                fields["source"] = "convergence_audit_recomputed"
                fields["validation_status"] = "convergence_audit_only_not_validated"

                fpath = (
                    args.output_dir / "fields_by_setting"
                    / f"M{mach:.2f}_alpha{alpha:.5f}_{setting_name}_fields.csv"
                )
                fields.to_csv(fpath, index=False)
                row["fields_file"] = str(fpath)

                setting_rows.append(row)
                setting_fields[setting_name] = fields

                print(
                    f"  {setting_name}: cr={row['cr']:.6f} ci={row['ci']:.6f} "
                    f"stage1={row['stage1_mismatch']:.2e} "
                    f"edge={row['max_edge_frac']:.2e} "
                    f"center={row['max_center_jump']:.2e}",
                    flush=True,
                )

            except Exception as exc:
                row = {
                    "Mach": mach,
                    "alpha": alpha,
                    "setting": setting_name,
                    "status": "exception",
                    "exception": repr(exc),
                }
                setting_rows.append(row)
                print(f"  {setting_name}: EXCEPTION {repr(exc)}", flush=True)

        if not setting_rows:
            continue

        # Compare tous les settings à la référence ms2_y1600 si disponible.
        ref_name = "ms2_y1600"
        if ref_name in setting_fields:
            ref = setting_fields[ref_name]
            for row in setting_rows:
                sname = row.get("setting")
                if sname in setting_fields:
                    comp = rel_l2_to_reference(ref, setting_fields[sname])
                    row.update(comp)

        all_rows.extend(setting_rows)

        valid_rows = [r for r in setting_rows if "cr" in r and "ci" in r]
        if valid_rows:
            crs = np.array([r["cr"] for r in valid_rows], dtype=float)
            cis = np.array([r["ci"] for r in valid_rows], dtype=float)
            stage1s = np.array([r.get("stage1_mismatch", np.inf) for r in valid_rows], dtype=float)
            edges = np.array([r.get("max_edge_frac", np.inf) for r in valid_rows], dtype=float)
            centers = np.array([r.get("max_center_jump", np.inf) for r in valid_rows], dtype=float)
            rels = np.array([r.get("max_rel_l2_to_ref", np.nan) for r in valid_rows], dtype=float)

            summary = {
                "Mach": mach,
                "alpha": alpha,
                "n_success_settings": int(len(valid_rows)),
                "cr_min": float(np.nanmin(crs)),
                "cr_max": float(np.nanmax(crs)),
                "cr_range": float(np.nanmax(crs) - np.nanmin(crs)),
                "ci_min": float(np.nanmin(cis)),
                "ci_max": float(np.nanmax(cis)),
                "ci_range": float(np.nanmax(cis) - np.nanmin(cis)),
                "max_stage1": float(np.nanmax(stage1s)),
                "max_edge_frac": float(np.nanmax(edges)),
                "max_center_jump": float(np.nanmax(centers)),
                "max_modal_rel_l2": float(np.nanmax(rels)) if np.isfinite(rels).any() else np.inf,
            }

            reasons = []
            if summary["n_success_settings"] < len(SETTINGS):
                reasons.append("some_settings_failed")
            if summary["max_stage1"] > 1e-4:
                reasons.append("stage1_not_converged")
            if summary["cr_range"] > 3e-3:
                reasons.append("cr_not_stable")
            if summary["ci_range"] > 3e-3:
                reasons.append("ci_not_stable")
            if summary["max_edge_frac"] > 2.5e-2:
                reasons.append("edge_not_decayed")
            if summary["max_center_jump"] > 5e-2:
                reasons.append("center_jump")
            if summary["max_modal_rel_l2"] > 0.15:
                reasons.append("modal_shape_not_stable")

            summary["convergence_status"] = (
                "converged_requires_visual_confirmation" if not reasons
                else "not_converged"
            )
            summary["reject_reasons"] = ";".join(reasons)
            point_summary.append(summary)

    rows_df = pd.DataFrame(all_rows)
    rows_df.to_csv(args.output_dir / "convergence_audit_by_setting.csv", index=False)

    summary_df = pd.DataFrame(point_summary)
    summary_df.to_csv(args.output_dir / "convergence_audit_by_point.csv", index=False)

    converged = summary_df[
        summary_df["convergence_status"].eq("converged_requires_visual_confirmation")
    ].copy()
    converged.to_csv(args.output_dir / "converged_candidates_requires_visual_confirmation.csv", index=False)

    summary_json = {
        "status": "convergence_audit_complete",
        "n_points_tested": int(len(summary_df)),
        "n_converged_requires_visual_confirmation": int(len(converged)),
        "status_counts": summary_df["convergence_status"].value_counts(dropna=False).to_dict()
        if len(summary_df) else {},
        "settings": [
            {
                "name": name,
                "mapping_scale": ms,
                "search_y_limit": sy,
                "final_y_limit": fy,
            }
            for name, ms, sy, fy in SETTINGS
        ],
        "acceptance_thresholds": {
            "max_stage1": 1e-4,
            "cr_range": 3e-3,
            "ci_range": 3e-3,
            "max_edge_frac": 2.5e-2,
            "max_center_jump": 5e-2,
            "max_modal_rel_l2": 0.15,
        },
        "important_note": (
            "This is a numerical convergence audit over mapping_scale and y_limit. "
            "It is still not a final validation. Visual confirmation is required."
        ),
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
