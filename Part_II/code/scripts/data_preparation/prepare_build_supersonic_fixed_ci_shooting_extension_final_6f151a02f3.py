#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from classic_supersonic_reference.solver.mstab17_supersonic_solver import Mstab17SupersonicSolver
from scripts.audits.audit_supersonic_shooting_visual_validation_6969b4f1bf import reconstruct_shooting_fields


OUTDIR = Path("assets/classic_supersonic/extension_neutral_M180_M190_gep_shooting")


def parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return bool(x)


def enrich_exact_amplitude(
    anchors: pd.DataFrame,
    *,
    match_y: float,
    mapping_scale: float,
    min_y_limit: float,
    max_y_limit: float,
    y_limit_factor: float,
) -> pd.DataFrame:
    rows = []

    for _, row in anchors.iterrows():
        mach = float(row["Mach"])
        alpha = float(row["alpha"])
        cr = float(row["shooting_cr"])
        ci = float(row["shooting_ci"])

        print(f"[exact-amplitude] M={mach} alpha={alpha} c=({cr}, {ci})")

        solver = Mstab17SupersonicSolver(
            alpha=alpha,
            Mach=mach,
            match_y=match_y,
            use_mapping=True,
            mapping_scale=mapping_scale,
            min_y_limit=min_y_limit,
            max_y_limit=max_y_limit,
            y_limit_factor=y_limit_factor,
        )

        sol_left, _, sol_right_full, y_limit = solver.get_trajectories(
            cr, ci, ln_p_start_right=0.0
        )

        out = row.to_dict()

        if not (sol_left.success and sol_right_full.success):
            out.update({
                "ln_p_start_right_exact": np.nan,
                "ln_p_left_target": np.nan,
                "ln_p_right_target_if_start_0": np.nan,
                "stage2_mismatch_exact": np.nan,
                "mode_success_exact": False,
                "full_success_exact": False,
                "y_limit_exact": float(y_limit),
                "amplitude_status": "trajectory_failure",
                "final_status": "trajectory_failure",
            })
            rows.append(out)
            continue

        target_y = solver.amplitude_match_y
        ln_left = solver._interp_component(target_y, sol_left, 2)
        ln_right_zero = solver._interp_component(target_y, sol_right_full, 2)

        ln_required = float(ln_left - ln_right_zero)
        stage2_exact = float(solver.stage2_objective(ln_required, cr, ci))

        spectral_success = parse_bool(row.get("spectral_success", False))
        mode_success_exact = bool(stage2_exact < 1e-2)
        full_success_exact = bool(spectral_success and mode_success_exact)

        out.update({
            "ln_p_start_right_exact": ln_required,
            "ln_p_left_target": float(ln_left),
            "ln_p_right_target_if_start_0": float(ln_right_zero),
            "stage2_mismatch_exact": stage2_exact,
            "mode_success_exact": mode_success_exact,
            "full_success_exact": full_success_exact,
            "y_limit_exact": float(y_limit),
            "amplitude_status": "exact_log_amplitude_match",
            "final_status": (
                "full_shooting_validated_exact_amplitude"
                if full_success_exact
                else "requires_review"
            ),
        })

        rows.append(out)

        print(
            f"[exact-amplitude-result] ln={ln_required} "
            f"stage2={stage2_exact} full_success={full_success_exact}"
        )

    return pd.DataFrame(rows)


def build_modal_fields(
    enriched: pd.DataFrame,
    *,
    match_y: float,
    mapping_scale: float,
    min_y_limit: float,
    max_y_limit: float,
    y_limit_factor: float,
) -> pd.DataFrame:
    rows = []

    for _, row in enriched.iterrows():
        mach = float(row["Mach"])
        alpha = float(row["alpha"])
        cr = float(row["shooting_cr"])
        ci = float(row["shooting_ci"])
        ln_p = float(row["ln_p_start_right_exact"])

        print(f"[modal-fields] M={mach} alpha={alpha} c=({cr}, {ci}) ln_p={ln_p}")

        fields = reconstruct_shooting_fields(
            alpha=alpha,
            mach=mach,
            cr=cr,
            ci=ci,
            ln_p_start_right=ln_p,
            match_y=match_y,
            use_mapping=True,
            mapping_scale=mapping_scale,
            min_y_limit=min_y_limit,
            max_y_limit=max_y_limit,
            y_limit_factor=y_limit_factor,
        )

        y = fields["y"]
        for i in range(len(y)):
            rows.append({
                "Mach": mach,
                "alpha": alpha,
                "cr": cr,
                "ci": ci,
                "omega_i": alpha * ci,
                "ln_p_start_right_exact": ln_p,
                "y": float(y[i]),
                "rho_real": float(np.real(fields["rho"][i])),
                "rho_imag": float(np.imag(fields["rho"][i])),
                "u_real": float(np.real(fields["u"][i])),
                "u_imag": float(np.imag(fields["u"][i])),
                "v_real": float(np.real(fields["v"][i])),
                "v_imag": float(np.imag(fields["v"][i])),
                "p_real": float(np.real(fields["p"][i])),
                "p_imag": float(np.imag(fields["p"][i])),
                "source": "fixed_ci_shooting_exact_amplitude",
                "validation_status": str(row["final_status"]),
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchors",
        type=Path,
        default=OUTDIR / "supersonic_neutral_M180_M190_fixed_ci_shooting_anchors.csv",
    )
    parser.add_argument(
        "--eigen-output",
        type=Path,
        default=OUTDIR / "supersonic_neutral_M180_M190_fixed_ci_shooting_anchors_exact_amplitude.csv",
    )
    parser.add_argument(
        "--fields-output",
        type=Path,
        default=OUTDIR / "supersonic_neutral_M180_M190_fixed_ci_shooting_modal_fields.csv",
    )
    parser.add_argument("--match-y", type=float, default=1.0)
    parser.add_argument("--mapping-scale", type=float, default=5.0)
    parser.add_argument("--min-y-limit", type=float, default=10.0)
    parser.add_argument("--max-y-limit", type=float, default=1200.0)
    parser.add_argument("--y-limit-factor", type=float, default=10.0)
    args = parser.parse_args()

    anchors = pd.read_csv(args.anchors)

    enriched = enrich_exact_amplitude(
        anchors,
        match_y=args.match_y,
        mapping_scale=args.mapping_scale,
        min_y_limit=args.min_y_limit,
        max_y_limit=args.max_y_limit,
        y_limit_factor=args.y_limit_factor,
    )
    args.eigen_output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.eigen_output, index=False)
    print(f"[final] wrote {args.eigen_output}")

    fields = build_modal_fields(
        enriched,
        match_y=args.match_y,
        mapping_scale=args.mapping_scale,
        min_y_limit=args.min_y_limit,
        max_y_limit=args.max_y_limit,
        y_limit_factor=args.y_limit_factor,
    )
    args.fields_output.parent.mkdir(parents=True, exist_ok=True)
    fields.to_csv(args.fields_output, index=False)
    print(f"[final] wrote {args.fields_output}")
    print("[final] rows =", len(fields))
    print(fields.groupby(["Mach", "alpha"]).size())


if __name__ == "__main__":
    main()
