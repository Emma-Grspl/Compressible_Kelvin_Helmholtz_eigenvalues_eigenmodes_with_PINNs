#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver
from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from scripts.data_preparation.prepare_build_supersonic_neutral_M180_M190_gep_shooting_extension import build_candidate_table


OUTDIR = Path("assets/classic_supersonic/extension_neutral_M180_M190_gep_shooting")


def finite_float(x, default=np.nan):
    try:
        y = float(x)
    except Exception:
        return default
    return y if np.isfinite(y) else default


def parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def nearest_gep_mode(alpha: float, mach: float, target_cr: float, target_ci: float, ci_weight: float = 4.0):
    solver = NotebookStyleDenseGEPSolver(
        alpha=alpha,
        Mach=mach,
        n_points=301,
        mapping_kind="pin",
        mapping_scale=5.0,
        xi_max=0.98,
    )
    modes = solver.finite_modes()
    if not modes:
        return None, np.nan, 0

    def dist(mode):
        return math.sqrt(
            (float(mode["cr"]) - target_cr) ** 2
            + ci_weight * (float(mode["ci"]) - target_ci) ** 2
        )

    best = min(modes, key=dist)
    return best, float(dist(best)), len(modes)


def run_one_probe(row, cr_guess: float, ci_guess: float, args):
    alpha = float(row["alpha"])
    mach = float(row["Mach"])

    cr_min = max(0.0, cr_guess - args.cr_half)
    cr_max = cr_guess + args.cr_half

    ci_min = max(args.ci_floor, ci_guess / args.ci_box_factor)
    ci_max = max(ci_guess * args.ci_box_factor, ci_guess + args.ci_extra)

    rec = {
        "Mach": mach,
        "alpha": alpha,
        "ci_seed_blumen": finite_float(row["ci_seed_blumen"]),
        "cr_seed_blumen": finite_float(row.get("cr_seed_blumen", np.nan)),
        "cr_guess": cr_guess,
        "ci_guess": ci_guess,
        "cr_min": cr_min,
        "cr_max": cr_max,
        "ci_min": ci_min,
        "ci_max": ci_max,
        "shooting_exception": "",
    }

    try:
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

        result = solver.solve(
            cr_min=cr_min,
            cr_max=cr_max,
            ci_min=ci_min,
            ci_max=ci_max,
            max_iter=args.max_iter,
            grid_size=args.grid_size,
        )

        shoot_cr = finite_float(getattr(result, "cr", np.nan))
        shoot_ci = finite_float(getattr(result, "ci", np.nan))

        rec.update(
            {
                "shooting_success": bool(getattr(result, "success", False)),
                "shooting_spectral_success": bool(getattr(result, "spectral_success", False)),
                "shooting_cr": shoot_cr,
                "shooting_ci": shoot_ci,
                "shooting_omega_i": finite_float(getattr(result, "omega_i", np.nan)),
                "shooting_residual": finite_float(getattr(result, "residual", np.nan)),
                "shooting_mismatch": finite_float(getattr(result, "mismatch", np.nan)),
                "shooting_y_limit": finite_float(getattr(result, "y_limit", np.nan)),
            }
        )

        if np.isfinite(shoot_cr) and np.isfinite(shoot_ci):
            mode, d, n_modes = nearest_gep_mode(alpha, mach, shoot_cr, shoot_ci, ci_weight=args.gep_ci_weight)
            rec["n_finite_gep_modes"] = n_modes
            rec["nearest_gep_distance"] = d

            if mode is not None:
                rec.update(
                    {
                        "nearest_gep_cr": finite_float(mode.get("cr", np.nan)),
                        "nearest_gep_ci": finite_float(mode.get("ci", np.nan)),
                        "nearest_gep_omega_i": finite_float(mode.get("omega_i", np.nan)),
                    }
                )
            else:
                rec.update(
                    {
                        "nearest_gep_cr": np.nan,
                        "nearest_gep_ci": np.nan,
                        "nearest_gep_omega_i": np.nan,
                    }
                )
        else:
            rec.update(
                {
                    "n_finite_gep_modes": np.nan,
                    "nearest_gep_distance": np.nan,
                    "nearest_gep_cr": np.nan,
                    "nearest_gep_ci": np.nan,
                    "nearest_gep_omega_i": np.nan,
                }
            )

    except Exception as exc:
        rec.update(
            {
                "shooting_success": False,
                "shooting_spectral_success": False,
                "shooting_cr": np.nan,
                "shooting_ci": np.nan,
                "shooting_omega_i": np.nan,
                "shooting_residual": np.nan,
                "shooting_mismatch": np.nan,
                "shooting_y_limit": np.nan,
                "n_finite_gep_modes": np.nan,
                "nearest_gep_distance": np.nan,
                "nearest_gep_cr": np.nan,
                "nearest_gep_ci": np.nan,
                "nearest_gep_omega_i": np.nan,
                "shooting_exception": repr(exc),
            }
        )

    shoot_ci = finite_float(rec.get("shooting_ci"))
    shoot_cr = finite_float(rec.get("shooting_cr"))
    nearest_d = finite_float(rec.get("nearest_gep_distance"))

    rec["accepted_probe"] = bool(
        np.isfinite(shoot_cr)
        and np.isfinite(shoot_ci)
        and shoot_cr > 0.0
        and shoot_ci > args.accept_ci_min
        and np.isfinite(nearest_d)
        and nearest_d <= args.accept_gep_distance
    )

    if rec["accepted_probe"]:
        rec["reason"] = "accepted_shooting_first_and_nearest_gep"
    elif not np.isfinite(shoot_cr) or not np.isfinite(shoot_ci):
        rec["reason"] = "rejected_no_finite_shooting_root"
    elif shoot_ci <= args.accept_ci_min:
        rec["reason"] = "rejected_shooting_ci_too_small"
    elif not np.isfinite(nearest_d):
        rec["reason"] = "rejected_no_nearest_gep"
    elif nearest_d > args.accept_gep_distance:
        rec["reason"] = "rejected_nearest_gep_too_far"
    else:
        rec["reason"] = "rejected_other"

    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci-datasets", type=Path, required=True)
    parser.add_argument("--cr-datasets", type=Path, required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--cr-guesses", type=str, default="")
    parser.add_argument("--ci-factors", type=str, default="1.0")
    parser.add_argument("--cr-half", type=float, default=0.08)
    parser.add_argument("--ci-box-factor", type=float, default=3.0)
    parser.add_argument("--ci-extra", type=float, default=0.002)
    parser.add_argument("--ci-floor", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--gep-ci-weight", type=float, default=4.0)
    parser.add_argument("--accept-ci-min", type=float, default=1e-6)
    parser.add_argument("--accept-gep-distance", type=float, default=0.08)
    parser.add_argument("--output", type=Path, default=OUTDIR / "shooting_first_probe.csv")
    args = parser.parse_args()

    candidates = build_candidate_table(args.ci_datasets, args.cr_datasets, max_points=None)

    sub = candidates[
        np.isclose(candidates["Mach"].astype(float), args.mach)
        & np.isclose(candidates["alpha"].astype(float), args.alpha)
    ]

    if sub.empty:
        raise SystemExit(f"No candidate row found for M={args.mach}, alpha={args.alpha}")

    row = sub.iloc[0]

    mach = float(row["Mach"])
    ci_seed = finite_float(row["ci_seed_blumen"], default=1e-4)
    cr_seed_blumen = finite_float(row.get("cr_seed_blumen", np.nan))
    cr_seed = cr_seed_blumen if np.isfinite(cr_seed_blumen) else 1.0 - 1.0 / mach

    if args.cr_guesses.strip():
        cr_guesses = parse_csv_floats(args.cr_guesses)
    else:
        cr_guesses = [
            cr_seed - 0.12,
            cr_seed - 0.06,
            cr_seed,
            cr_seed + 0.06,
            cr_seed + 0.12,
            0.80,
            0.90,
            0.98,
        ]

    cr_guesses = sorted({round(x, 12) for x in cr_guesses if x > 0.0})

    ci_factors = parse_csv_floats(args.ci_factors)
    ci_guesses = [max(args.ci_floor, ci_seed * f) for f in ci_factors]

    rows = []
    print(f"[probe] M={mach} alpha={float(row['alpha'])} ci_seed={ci_seed} cr_seed={cr_seed}")
    print(f"[probe] cr_guesses={cr_guesses}")
    print(f"[probe] ci_guesses={ci_guesses}")

    for cr_guess in cr_guesses:
        for ci_guess in ci_guesses:
            print(f"[probe] shooting box around cr={cr_guess:.6g}, ci={ci_guess:.6g}")
            rec = run_one_probe(row, cr_guess, ci_guess, args)
            rows.append(rec)
            print(
                "[probe-result]",
                "accepted=", rec["accepted_probe"],
                "shoot=(", rec.get("shooting_cr"), rec.get("shooting_ci"), ")",
                "gep=(", rec.get("nearest_gep_cr"), rec.get("nearest_gep_ci"), ")",
                "d=", rec.get("nearest_gep_distance"),
                "reason=", rec["reason"],
            )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[probe] wrote {args.output}")


if __name__ == "__main__":
    main()
