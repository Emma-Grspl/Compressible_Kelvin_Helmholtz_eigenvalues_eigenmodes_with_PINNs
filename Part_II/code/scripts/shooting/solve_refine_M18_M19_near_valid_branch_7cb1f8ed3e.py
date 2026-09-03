#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver
from scripts.audits.audit_scan_supersonic_M18_M19_strict_modal_validation_14e1027f3b import (
    exact_log_amplitude,
    reconstruct_dataframe,
    field_shape_metrics,
    ylimit_stability,
)


def load_seeds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    for c in ["Mach", "alpha", "cr", "ci", "stage1_mismatch"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Mach", "alpha", "cr", "ci", "stage1_mismatch"])
    df = (
        df.sort_values(["Mach", "alpha", "stage1_mismatch"])
          .groupby(["Mach", "alpha"], as_index=False)
          .first()
          .sort_values(["Mach", "alpha"])
          .reset_index(drop=True)
    )
    return df


def build_refined_grid(seeds: pd.DataFrame, alpha_step: float) -> pd.DataFrame:
    rows = []

    for mach, g in seeds.groupby("Mach"):
        g = g.sort_values("alpha")
        amin = float(g["alpha"].min())
        amax = float(g["alpha"].max())

        alphas = np.arange(amin, amax + 0.5 * alpha_step, alpha_step)
        alphas = np.round(alphas, 10)

        cr_interp = np.interp(alphas, g["alpha"], g["cr"])
        ci_interp = np.interp(alphas, g["alpha"], g["ci"])

        for a, cr0, ci0 in zip(alphas, cr_interp, ci_interp):
            rows.append({
                "Mach": float(mach),
                "alpha": float(a),
                "cr_seed": float(cr0),
                "ci_seed": float(ci0),
            })

    return pd.DataFrame(rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)


def solve_local_candidate(
    *,
    mach: float,
    alpha: float,
    cr_seed: float,
    ci_seed: float,
    cr_width: float,
    ci_width: float,
    mapping_scale: float,
    search_y_limit: float,
    grid_size: int,
    max_iter: int,
):
    cr_min = max(0.0, cr_seed - cr_width)
    cr_max = cr_seed + cr_width
    ci_min = max(1e-6, ci_seed - ci_width)
    ci_max = max(ci_min * 1.01, ci_seed + ci_width)

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

    return float(cr), float(ci), float(stage1), {
        "cr_min": cr_min,
        "cr_max": cr_max,
        "ci_min": ci_min,
        "ci_max": ci_max,
    }


def validate_refined(
    *,
    mach: float,
    alpha: float,
    cr: float,
    ci: float,
    stage1: float,
    mapping_scale: float,
    final_y_limit: float,
):
    # Deux reconstructions pour tester la stabilité y_limit.
    ln1200, stage2_1200, ylimit1200, amp_status1200 = exact_log_amplitude(
        alpha=alpha,
        mach=mach,
        cr=cr,
        ci=ci,
        max_y_limit=1200.0,
        mapping_scale=mapping_scale,
    )

    ln_final, stage2_final, ylimit_final, amp_status_final = exact_log_amplitude(
        alpha=alpha,
        mach=mach,
        cr=cr,
        ci=ci,
        max_y_limit=final_y_limit,
        mapping_scale=mapping_scale,
    )

    df1200 = reconstruct_dataframe(
        alpha=alpha,
        mach=mach,
        cr=cr,
        ci=ci,
        ln_p_start_right=ln1200,
        max_y_limit=1200.0,
        mapping_scale=mapping_scale,
    )

    df_final = reconstruct_dataframe(
        alpha=alpha,
        mach=mach,
        cr=cr,
        ci=ci,
        ln_p_start_right=ln_final,
        max_y_limit=final_y_limit,
        mapping_scale=mapping_scale,
    )

    shape = field_shape_metrics(df_final)
    stab = ylimit_stability(df1200, df_final)

    metrics = {
        "stage1_mismatch": float(stage1),
        "stage2_mismatch_exact_1200": float(stage2_1200),
        "stage2_mismatch_exact_final": float(stage2_final),
        "ln_p_start_right_exact_1200": float(ln1200),
        "ln_p_start_right_exact_final": float(ln_final),
        "y_limit_1200": float(ylimit1200),
        "y_limit_final": float(ylimit_final),
        "amplitude_status_1200": amp_status1200,
        "amplitude_status_final": amp_status_final,
    }
    metrics.update(shape)
    metrics.update(stab)

    # Critères d'acceptation raffinée.
    # adjacent_jump est reporté mais non bloquant, car il capte le pic local de u / couche critique.
    reasons = []

    if metrics["stage1_mismatch"] > 1e-4:
        reasons.append("stage1_too_large")
    if metrics["stage2_mismatch_exact_final"] > 1e-8:
        reasons.append("stage2_too_large")
    if not metrics["all_finite"]:
        reasons.append("non_finite_fields")
    if metrics["max_edge_frac"] > 2.5e-2:
        reasons.append("edge_not_decayed")
    if metrics["max_center_jump"] > 5e-2:
        reasons.append("center_jump_too_large")
    if metrics["max_ylimit_rel_l2"] > 8e-2:
        reasons.append("ylimit_unstable")

    if reasons:
        status = "refined_rejected"
    else:
        status = "refined_near_valid_requires_visual_confirmation"

    metrics["validation_status"] = status
    metrics["reject_reasons"] = ";".join(reasons)

    return metrics, df_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-csv", type=Path, default=Path(
        "assets/classic_supersonic/scan_M18_M19_alpha_0_03_strict/"
        "near_valid_except_adjacent_jump_candidates.csv"
    ))
    ap.add_argument("--output-dir", type=Path, default=Path(
        "assets/classic_supersonic/scan_M18_M19_alpha_0_03_strict/"
        "refined_near_valid_branch"
    ))
    ap.add_argument("--alpha-step", type=float, default=0.0025)
    ap.add_argument("--mapping-scale", type=float, default=2.0)
    ap.add_argument("--search-y-limit", type=float, default=900.0)
    ap.add_argument("--final-y-limit", type=float, default=1600.0)
    ap.add_argument("--cr-width", type=float, default=0.012)
    ap.add_argument("--ci-width", type=float, default=0.004)
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=7)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fields").mkdir(exist_ok=True)

    seeds = load_seeds(args.seed_csv)
    task_grid = build_refined_grid(seeds, args.alpha_step)

    task_grid.to_csv(args.output_dir / "refined_task_grid.csv", index=False)

    rows = []
    accepted_fields = []

    print("n refined tasks:", len(task_grid))
    print(task_grid.groupby("Mach")["alpha"].agg(["min", "max", "count"]))

    for i, r in task_grid.iterrows():
        mach = float(r["Mach"])
        alpha = float(r["alpha"])
        cr_seed = float(r["cr_seed"])
        ci_seed = float(r["ci_seed"])

        print(f"[{i+1}/{len(task_grid)}] M={mach:.2f} alpha={alpha:.5f} seed cr={cr_seed:.6f} ci={ci_seed:.6f}", flush=True)

        row = {
            "Mach": mach,
            "alpha": alpha,
            "cr_seed": cr_seed,
            "ci_seed": ci_seed,
            "mapping_scale": args.mapping_scale,
            "final_y_limit_requested": args.final_y_limit,
        }

        try:
            cr, ci, stage1, box = solve_local_candidate(
                mach=mach,
                alpha=alpha,
                cr_seed=cr_seed,
                ci_seed=ci_seed,
                cr_width=args.cr_width,
                ci_width=args.ci_width,
                mapping_scale=args.mapping_scale,
                search_y_limit=args.search_y_limit,
                grid_size=args.grid_size,
                max_iter=args.max_iter,
            )

            row.update(box)
            row.update({
                "cr": cr,
                "ci": ci,
                "omega_i": alpha * ci,
                "stage1_mismatch": stage1,
            })

            metrics, fields = validate_refined(
                mach=mach,
                alpha=alpha,
                cr=cr,
                ci=ci,
                stage1=stage1,
                mapping_scale=args.mapping_scale,
                final_y_limit=args.final_y_limit,
            )

            row.update(metrics)

            if metrics["validation_status"] == "refined_near_valid_requires_visual_confirmation":
                fields["source"] = "refined_near_valid_branch_mapping_scale_2"
                fields["validation_status"] = metrics["validation_status"]
                fields["stage1_mismatch"] = stage1
                fields["stage2_mismatch_exact_final"] = metrics["stage2_mismatch_exact_final"]
                fields["max_edge_frac"] = metrics["max_edge_frac"]
                fields["max_center_jump"] = metrics["max_center_jump"]
                fields["max_adjacent_jump"] = metrics["max_adjacent_jump"]
                fields["max_ylimit_rel_l2"] = metrics["max_ylimit_rel_l2"]

                fpath = args.output_dir / "fields" / f"M{mach:.2f}_alpha{alpha:.5f}_fields.csv"
                fields.to_csv(fpath, index=False)
                row["fields_file"] = str(fpath)
                accepted_fields.append(fields)

            print(
                f"    -> cr={cr:.6f} ci={ci:.6f} stage1={stage1:.3e} "
                f"edge={row.get('max_edge_frac', np.nan):.2e} "
                f"center={row.get('max_center_jump', np.nan):.2e} "
                f"adj={row.get('max_adjacent_jump', np.nan):.2e} "
                f"ylimit={row.get('max_ylimit_rel_l2', np.nan):.2e} "
                f"status={row.get('validation_status')}",
                flush=True,
            )

        except Exception as exc:
            row["validation_status"] = "refined_exception"
            row["reject_reasons"] = repr(exc)
            print(f"    -> EXCEPTION {repr(exc)}", flush=True)

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)
    out.to_csv(args.output_dir / "refined_candidates.csv", index=False)

    accepted = out[out["validation_status"].eq("refined_near_valid_requires_visual_confirmation")].copy()
    accepted.to_csv(args.output_dir / "refined_near_valid_candidates.csv", index=False)

    if accepted_fields:
        fields_all = pd.concat(accepted_fields, ignore_index=True)
        fields_all = fields_all.sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)
        fields_all.to_csv(args.output_dir / "refined_near_valid_modal_fields.csv", index=False)

    summary = {
        "status": "refined_near_valid_branch_requires_visual_confirmation",
        "alpha_step": args.alpha_step,
        "mapping_scale": args.mapping_scale,
        "final_y_limit": args.final_y_limit,
        "n_tasks": int(len(task_grid)),
        "n_refined_near_valid": int(len(accepted)),
        "status_counts": out["validation_status"].value_counts(dropna=False).to_dict(),
        "by_Mach": accepted.groupby("Mach").size().to_dict() if len(accepted) else {},
        "note": (
            "These are refined near-valid candidates, not final validated points. "
            "They require visual confirmation after higher-resolution reconstruction."
        ),
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
