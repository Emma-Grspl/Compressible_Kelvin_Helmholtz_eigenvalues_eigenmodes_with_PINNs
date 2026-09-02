#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from classic_supersonic_reference.solver.mstab17_supersonic_solver import Mstab17SupersonicSolver
from scripts.data_preparation.prepare_build_supersonic_neutral_M180_M190_gep_shooting_extension import build_candidate_table


OUTDIR = Path("assets/classic_supersonic/extension_neutral_M180_M190_gep_shooting")


def finite_float(x, default=np.nan):
    try:
        y = float(x)
    except Exception:
        return default
    return y if np.isfinite(y) else default


def find_cr_at_fixed_ci(
    *,
    mach: float,
    alpha: float,
    ci_fixed: float,
    cr_min: float,
    cr_max: float,
    n_grid: int,
):
    solver = Mstab17SupersonicSolver(
        alpha=alpha,
        Mach=mach,
        match_y=1.0,
        use_mapping=True,
        mapping_scale=5.0,
        min_y_limit=10.0,
        max_y_limit=1200.0,
        y_limit_factor=10.0,
        ln_p_right_min=-30.0,
        ln_p_right_max=5.0,
    )

    cr_grid = np.linspace(cr_min, cr_max, n_grid)
    vals = []

    for cr in cr_grid:
        try:
            mismatch = float(solver.stage1_mismatch(float(cr), float(ci_fixed)))
        except Exception:
            mismatch = np.nan
        vals.append(mismatch)

    vals = np.asarray(vals, dtype=float)
    finite = np.isfinite(vals)

    if not finite.any():
        return {
            "shooting_cr": np.nan,
            "shooting_ci": ci_fixed,
            "stage1_mismatch": np.nan,
            "spectral_success": False,
            "reason": "no_finite_stage1_mismatch",
            "grid_best_cr": np.nan,
            "grid_best_mismatch": np.nan,
        }

    finite_indices = np.where(finite)[0]
    best_idx = finite_indices[np.argmin(vals[finite])]
    grid_best_cr = float(cr_grid[best_idx])
    grid_best_mismatch = float(vals[best_idx])

    if best_idx == 0:
        bracket_min = float(cr_grid[0])
        bracket_max = float(cr_grid[1])
        edge_flag = "left_edge"
    elif best_idx == len(cr_grid) - 1:
        bracket_min = float(cr_grid[-2])
        bracket_max = float(cr_grid[-1])
        edge_flag = "right_edge"
    else:
        bracket_min = float(cr_grid[best_idx - 1])
        bracket_max = float(cr_grid[best_idx + 1])
        edge_flag = "interior"

    def objective(cr):
        try:
            return float(solver.stage1_mismatch(float(cr), float(ci_fixed)))
        except Exception:
            return 1e99

    opt = minimize_scalar(
        objective,
        bounds=(bracket_min, bracket_max),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 80},
    )

    cr_opt = float(opt.x)
    mismatch_opt = float(opt.fun)

    return {
        "shooting_cr": cr_opt,
        "shooting_ci": ci_fixed,
        "omega_i": float(alpha * ci_fixed),
        "stage1_mismatch": mismatch_opt,
        "spectral_success": bool(mismatch_opt < 5e-2),
        "mode_success": False,
        "success": False,
        "reason": "fixed_ci_spectral_anchor",
        "grid_best_cr": grid_best_cr,
        "grid_best_mismatch": grid_best_mismatch,
        "grid_edge_flag": edge_flag,
        "minimize_success": bool(opt.success),
        "minimize_message": str(opt.message),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci-datasets", type=Path, required=True)
    parser.add_argument("--cr-datasets", type=Path, required=True)
    parser.add_argument("--cr-min", type=float, default=0.55)
    parser.add_argument("--cr-max", type=float, default=1.05)
    parser.add_argument("--n-grid", type=int, default=101)
    parser.add_argument("--output", type=Path, default=OUTDIR / "supersonic_neutral_M180_M190_fixed_ci_shooting_anchors.csv")
    args = parser.parse_args()

    candidates = build_candidate_table(args.ci_datasets, args.cr_datasets, max_points=None)

    rows = []
    for _, row in candidates.iterrows():
        mach = float(row["Mach"])
        alpha = float(row["alpha"])
        ci_seed = finite_float(row["ci_seed_blumen"])

        print(f"[fixed-ci] M={mach} alpha={alpha} ci_seed={ci_seed}")

        if not np.isfinite(ci_seed) or ci_seed <= 0:
            rows.append(
                {
                    "Mach": mach,
                    "alpha": alpha,
                    "ci_seed_blumen": ci_seed,
                    "status": "failed",
                    "reason": "missing_ci_seed_blumen",
                }
            )
            continue

        res = find_cr_at_fixed_ci(
            mach=mach,
            alpha=alpha,
            ci_fixed=ci_seed,
            cr_min=args.cr_min,
            cr_max=args.cr_max,
            n_grid=args.n_grid,
        )

        status = "spectral_validated" if res["spectral_success"] else "requires_review"

        out = {
            "Mach": mach,
            "alpha": alpha,
            "ci_seed_blumen": ci_seed,
            "cr_search_min": args.cr_min,
            "cr_search_max": args.cr_max,
            "status": status,
        }
        out.update(res)
        rows.append(out)

        print(
            f"[fixed-ci-result] M={mach} alpha={alpha} "
            f"cr={out.get('shooting_cr')} ci={ci_seed} "
            f"stage1={out.get('stage1_mismatch')} "
            f"spectral_success={out.get('spectral_success')}"
        )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[fixed-ci] wrote {args.output}")


if __name__ == "__main__":
    main()
