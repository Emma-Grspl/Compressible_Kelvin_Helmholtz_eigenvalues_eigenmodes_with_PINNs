#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver
from scripts.audits.audit_supersonic_shooting_visual_validation_6969b4f1bf import reconstruct_shooting_fields


FIELDS = {
    "p": ("p_real", "p_imag"),
    "rho": ("rho_real", "rho_imag"),
    "u": ("u_real", "u_imag"),
    "v": ("v_real", "v_imag"),
}


def exact_log_amplitude(
    *,
    alpha: float,
    mach: float,
    cr: float,
    ci: float,
    max_y_limit: float,
    match_y: float = 1.0,
    mapping_scale: float = 5.0,
):
    solver = Mstab17SupersonicSolver(
        alpha=alpha,
        Mach=mach,
        match_y=match_y,
        use_mapping=True,
        mapping_scale=mapping_scale,
        min_y_limit=10.0,
        max_y_limit=max_y_limit,
        y_limit_factor=10.0,
    )

    sol_left, _, sol_right_full, y_limit = solver.get_trajectories(
        cr, ci, ln_p_start_right=0.0
    )

    if not (sol_left.success and sol_right_full.success):
        return np.nan, np.nan, float(y_limit), "trajectory_failure"

    target_y = solver.amplitude_match_y
    ln_left = solver._interp_component(target_y, sol_left, 2)
    ln_right_zero = solver._interp_component(target_y, sol_right_full, 2)

    ln_required = float(ln_left - ln_right_zero)
    stage2_exact = float(solver.stage2_objective(ln_required, cr, ci))

    return ln_required, stage2_exact, float(y_limit), "exact_log_amplitude_match"


def to_complex(fields: pd.DataFrame, name: str) -> np.ndarray:
    re_col, im_col = FIELDS[name]
    return (
        pd.to_numeric(fields[re_col], errors="coerce").to_numpy(dtype=float)
        + 1j * pd.to_numeric(fields[im_col], errors="coerce").to_numpy(dtype=float)
    )


def reconstruct_dataframe(
    *,
    alpha: float,
    mach: float,
    cr: float,
    ci: float,
    ln_p_start_right: float,
    max_y_limit: float,
    match_y: float = 1.0,
    mapping_scale: float = 5.0,
) -> pd.DataFrame:
    fields = reconstruct_shooting_fields(
        alpha=alpha,
        mach=mach,
        cr=cr,
        ci=ci,
        ln_p_start_right=ln_p_start_right,
        match_y=match_y,
        use_mapping=True,
        mapping_scale=mapping_scale,
        min_y_limit=10.0,
        max_y_limit=max_y_limit,
        y_limit_factor=10.0,
    )

    y = np.asarray(fields["y"], dtype=float)

    out = pd.DataFrame({
        "Mach": mach,
        "alpha": alpha,
        "cr": cr,
        "ci": ci,
        "omega_i": alpha * ci,
        "ln_p_start_right_exact": ln_p_start_right,
        "max_y_limit_used": max_y_limit,
        "y": y,
    })

    for name in ["p", "rho", "u", "v"]:
        z = np.asarray(fields[name])
        out[f"{name}_real"] = np.real(z)
        out[f"{name}_imag"] = np.imag(z)

    return out.sort_values("y").reset_index(drop=True)


def field_shape_metrics(df: pd.DataFrame) -> dict:
    y = df["y"].to_numpy(dtype=float)
    ymax = float(np.nanmax(np.abs(y)))

    metrics = {}

    all_finite = True
    edge_fracs = []
    center_jumps = []
    adjacent_jumps = []
    zero_side_ratios = []

    center_idx = int(np.argmin(np.abs(y)))

    for name in ["p", "rho", "u", "v"]:
        z = to_complex(df, name)
        amp = np.abs(z)

        finite = np.isfinite(np.real(z)).all() and np.isfinite(np.imag(z)).all()
        all_finite = all_finite and bool(finite)

        peak = float(np.nanmax(amp)) if len(amp) else np.nan
        if not np.isfinite(peak) or peak <= 0:
            edge_frac = np.inf
            center_jump = np.inf
            adjacent_jump = np.inf
            side_ratio = np.inf
        else:
            edge_mask = np.abs(y) > 0.90 * ymax
            edge_frac = float(np.nanmax(amp[edge_mask]) / peak) if np.any(edge_mask) else np.inf

            if 1 <= center_idx < len(z) - 1:
                center_jump = float(abs(z[center_idx + 1] - z[center_idx - 1]) / peak)
            else:
                center_jump = np.inf

            dz = np.diff(z)
            adjacent_jump = float(np.nanmax(np.abs(dz)) / peak) if len(dz) else np.inf

            left_energy = float(np.trapz(amp[y < 0] ** 2, y[y < 0])) if np.any(y < 0) else 0.0
            right_energy = float(np.trapz(amp[y > 0] ** 2, y[y > 0])) if np.any(y > 0) else 0.0
            mn = max(min(left_energy, right_energy), 1e-300)
            mx = max(left_energy, right_energy, 1e-300)
            side_ratio = float(mx / mn)

        metrics[f"{name}_edge_frac"] = edge_frac
        metrics[f"{name}_center_jump"] = center_jump
        metrics[f"{name}_adjacent_jump"] = adjacent_jump
        metrics[f"{name}_left_right_energy_ratio"] = side_ratio

        edge_fracs.append(edge_frac)
        center_jumps.append(center_jump)
        adjacent_jumps.append(adjacent_jump)
        zero_side_ratios.append(side_ratio)

    metrics["all_finite"] = bool(all_finite)
    metrics["max_edge_frac"] = float(np.nanmax(edge_fracs))
    metrics["max_center_jump"] = float(np.nanmax(center_jumps))
    metrics["max_adjacent_jump"] = float(np.nanmax(adjacent_jumps))
    metrics["max_left_right_energy_ratio"] = float(np.nanmax(zero_side_ratios))

    return metrics


def interp_complex(y_src, z_src, y_tgt):
    zr = np.interp(y_tgt, y_src, np.real(z_src))
    zi = np.interp(y_tgt, y_src, np.imag(z_src))
    return zr + 1j * zi


def ylimit_stability(df_low: pd.DataFrame, df_high: pd.DataFrame) -> dict:
    y_low = df_low["y"].to_numpy(dtype=float)
    y_high = df_high["y"].to_numpy(dtype=float)

    common_min = max(float(np.nanmin(y_low)), float(np.nanmin(y_high)))
    common_max = min(float(np.nanmax(y_low)), float(np.nanmax(y_high)))

    out = {}

    rels = []

    for name in ["p", "rho", "u", "v"]:
        z_low = to_complex(df_low, name)
        z_high = to_complex(df_high, name)

        amp_high = np.abs(z_high)
        peak = float(np.nanmax(amp_high))

        if not np.isfinite(peak) or peak <= 0:
            rel = np.inf
            out[f"{name}_ylimit_rel_l2"] = rel
            rels.append(rel)
            continue

        mask_high = (
            (y_high >= common_min)
            & (y_high <= common_max)
            & (amp_high >= 1e-3 * peak)
        )

        if mask_high.sum() < 20:
            mask_high = (y_high >= common_min) & (y_high <= common_max)

        y_grid = y_high[mask_high]
        zh = z_high[mask_high]
        zl = interp_complex(y_low, z_low, y_grid)

        denom = np.linalg.norm(zh)
        if denom <= 0 or not np.isfinite(denom):
            rel = np.inf
        else:
            phase = np.vdot(zl, zh)
            if abs(phase) > 0:
                zl = zl * phase / abs(phase)
            rel = float(np.linalg.norm(zh - zl) / denom)

        out[f"{name}_ylimit_rel_l2"] = rel
        rels.append(rel)

    out["max_ylimit_rel_l2"] = float(np.nanmax(rels))
    return out


def strict_accept(metrics: dict, args) -> tuple[bool, list[str]]:
    reasons = []

    if not metrics.get("all_finite", False):
        reasons.append("non_finite_fields")

    if metrics["stage1_mismatch"] > args.stage1_tol:
        reasons.append("stage1_too_large")

    if metrics["stage2_mismatch_exact_1200"] > args.stage2_tol:
        reasons.append("stage2_exact_too_large")

    if metrics["max_edge_frac"] > args.max_edge_frac:
        reasons.append("edge_not_decayed")

    if metrics["max_center_jump"] > args.max_center_jump:
        reasons.append("center_jump_too_large")

    if metrics["max_adjacent_jump"] > args.max_adjacent_jump:
        reasons.append("adjacent_jump_too_large")

    if metrics["max_ylimit_rel_l2"] > args.max_ylimit_rel_l2:
        reasons.append("ylimit_unstable")

    return len(reasons) == 0, reasons


def solve_candidates_for_point(alpha: float, mach: float, args) -> list[dict]:
    cr_boxes = [
        (0.00, 0.25),
        (0.20, 0.45),
        (0.40, 0.65),
        (0.60, 0.85),
        (0.80, 1.05),
    ]

    ci_boxes = [
        (1e-5, 2e-3),
        (1e-3, 8e-3),
        (5e-3, 2e-2),
        (1.5e-2, 6e-2),
        (5e-2, 1.2e-1),
    ]

    rows = []

    for cr_min, cr_max in cr_boxes:
        for ci_min, ci_max in ci_boxes:
            solver = Mstab17SupersonicSolver(
                alpha=alpha,
                Mach=mach,
                match_y=1.0,
                use_mapping=True,
                mapping_scale=5.0,
                min_y_limit=10.0,
                max_y_limit=args.search_y_limit,
                y_limit_factor=10.0,
            )

            try:
                cr, ci, err = solver.solve_eigenvalue(
                    cr_min=cr_min,
                    cr_max=cr_max,
                    ci_min=ci_min,
                    ci_max=ci_max,
                    max_iter=args.max_iter,
                    grid_size=args.grid_size,
                    constrain_to_initial_box=True,
                    tol=1e-10,
                )
            except Exception as exc:
                rows.append({
                    "Mach": mach,
                    "alpha": alpha,
                    "cr_box_min": cr_min,
                    "cr_box_max": cr_max,
                    "ci_box_min": ci_min,
                    "ci_box_max": ci_max,
                    "status": "solve_exception",
                    "exception": repr(exc),
                })
                continue

            rows.append({
                "Mach": mach,
                "alpha": alpha,
                "cr": float(cr),
                "ci": float(ci),
                "omega_i": float(alpha * ci),
                "stage1_mismatch": float(err),
                "cr_box_min": cr_min,
                "cr_box_max": cr_max,
                "ci_box_min": ci_min,
                "ci_box_max": ci_max,
                "status": "raw_candidate",
                "exception": "",
            })

    # Dedup close candidates.
    good = [r for r in rows if "cr" in r and np.isfinite(r["stage1_mismatch"])]
    good = sorted(good, key=lambda r: r["stage1_mismatch"])

    dedup = []
    seen = set()
    for r in good:
        key = (round(r["cr"], 4), round(r["ci"], 5))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)

    return dedup[: args.max_validate_per_point] + [r for r in rows if "cr" not in r]


def validate_candidate(row: dict, args) -> tuple[dict, pd.DataFrame | None]:
    mach = float(row["Mach"])
    alpha = float(row["alpha"])
    cr = float(row["cr"])
    ci = float(row["ci"])

    out = dict(row)

    try:
        ln600, stage2_600, y600, amp_status_600 = exact_log_amplitude(
            alpha=alpha,
            mach=mach,
            cr=cr,
            ci=ci,
            max_y_limit=600.0,
        )

        ln1200, stage2_1200, y1200, amp_status_1200 = exact_log_amplitude(
            alpha=alpha,
            mach=mach,
            cr=cr,
            ci=ci,
            max_y_limit=1200.0,
        )

        out.update({
            "ln_p_start_right_exact_600": ln600,
            "stage2_mismatch_exact_600": stage2_600,
            "y_limit_600": y600,
            "amplitude_status_600": amp_status_600,
            "ln_p_start_right_exact_1200": ln1200,
            "stage2_mismatch_exact_1200": stage2_1200,
            "y_limit_1200": y1200,
            "amplitude_status_1200": amp_status_1200,
        })

        if not np.isfinite(ln600) or not np.isfinite(ln1200):
            out["status"] = "rejected"
            out["reject_reasons"] = "amplitude_match_failure"
            return out, None

        df600 = reconstruct_dataframe(
            alpha=alpha,
            mach=mach,
            cr=cr,
            ci=ci,
            ln_p_start_right=ln600,
            max_y_limit=600.0,
        )

        df1200 = reconstruct_dataframe(
            alpha=alpha,
            mach=mach,
            cr=cr,
            ci=ci,
            ln_p_start_right=ln1200,
            max_y_limit=1200.0,
        )

        shape = field_shape_metrics(df1200)
        stab = ylimit_stability(df600, df1200)

        out.update(shape)
        out.update(stab)

        ok, reasons = strict_accept(out, args)

        out["status"] = "strict_auto_validated" if ok else "rejected"
        out["reject_reasons"] = "" if ok else ";".join(reasons)

        if ok:
            df1200["validation_status"] = "strict_auto_validated_requires_visual_confirmation"
            df1200["source"] = "M18_M19_strict_scan"
            df1200["stage1_mismatch"] = out["stage1_mismatch"]
            df1200["stage2_mismatch_exact"] = out["stage2_mismatch_exact_1200"]
            df1200["max_edge_frac"] = out["max_edge_frac"]
            df1200["max_center_jump"] = out["max_center_jump"]
            df1200["max_adjacent_jump"] = out["max_adjacent_jump"]
            df1200["max_ylimit_rel_l2"] = out["max_ylimit_rel_l2"]
            return out, df1200

        return out, None

    except Exception as exc:
        out["status"] = "validation_exception"
        out["reject_reasons"] = repr(exc)
        return out, None


def task_grid(args):
    alphas = np.arange(args.alpha_min, args.alpha_max + 0.5 * args.alpha_step, args.alpha_step)
    alphas = [float(round(a, 10)) for a in alphas if a > 0]
    machs = [float(m) for m in args.machs]
    return [(m, a) for m in machs for a in alphas]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machs", nargs="+", type=float, default=[1.8, 1.9])
    parser.add_argument("--alpha-min", type=float, default=0.01)
    parser.add_argument("--alpha-max", type=float, default=0.30)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/classic_supersonic/scan_M18_M19_alpha_0_03_strict"))
    parser.add_argument("--search-y-limit", type=float, default=600.0)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=7)
    parser.add_argument("--max-validate-per-point", type=int, default=6)

    parser.add_argument("--stage1-tol", type=float, default=1e-3)
    parser.add_argument("--stage2-tol", type=float, default=1e-8)
    parser.add_argument("--max-edge-frac", type=float, default=2e-2)
    parser.add_argument("--max-center-jump", type=float, default=0.25)
    parser.add_argument("--max-adjacent-jump", type=float, default=0.35)
    parser.add_argument("--max-ylimit-rel-l2", type=float, default=0.20)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "point_results").mkdir(exist_ok=True)
    (args.output_dir / "accepted_fields").mkdir(exist_ok=True)

    jobs = task_grid(args)

    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(jobs):
            raise SystemExit(f"task-id {args.task_id} outside 0..{len(jobs)-1}")
        jobs = [jobs[args.task_id]]

    all_rows = []

    for mach, alpha in jobs:
        print(f"[point] M={mach} alpha={alpha}")

        candidates = solve_candidates_for_point(alpha, mach, args)

        point_rows = []
        accepted_fields = []

        for cand in candidates:
            if "cr" not in cand:
                point_rows.append(cand)
                continue

            result, fields_df = validate_candidate(cand, args)
            point_rows.append(result)

            print(
                f"[candidate] M={mach} alpha={alpha} "
                f"cr={result.get('cr')} ci={result.get('ci')} "
                f"stage1={result.get('stage1_mismatch')} "
                f"status={result.get('status')} "
                f"reasons={result.get('reject_reasons')}"
            )

            if fields_df is not None:
                accepted_fields.append(fields_df)

        point_df = pd.DataFrame(point_rows)
        point_path = args.output_dir / "point_results" / f"M{mach:.2f}_alpha{alpha:.5f}_candidates.csv"
        point_df.to_csv(point_path, index=False)

        if accepted_fields:
            fields_out = pd.concat(accepted_fields, ignore_index=True)
            fields_path = args.output_dir / "accepted_fields" / f"M{mach:.2f}_alpha{alpha:.5f}_accepted_fields.csv"
            fields_out.to_csv(fields_path, index=False)

        all_rows.extend(point_rows)

    if args.task_id is None:
        df = pd.DataFrame(all_rows)
        df.to_csv(args.output_dir / "all_candidates.csv", index=False)

        accepted = df[df["status"].eq("strict_auto_validated")].copy()
        accepted.to_csv(args.output_dir / "strict_auto_validated_candidates.csv", index=False)

        summary = {
            "n_jobs": len(jobs),
            "n_rows": int(len(df)),
            "n_strict_auto_validated": int(len(accepted)),
            "criteria": {
                "stage1_tol": args.stage1_tol,
                "stage2_tol": args.stage2_tol,
                "max_edge_frac": args.max_edge_frac,
                "max_center_jump": args.max_center_jump,
                "max_adjacent_jump": args.max_adjacent_jump,
                "max_ylimit_rel_l2": args.max_ylimit_rel_l2,
            },
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
