#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from classical_solver.supersonic.mstab17_supersonic_solver import (
    Mstab17SupersonicSolver,
)
from scripts.audits.audit_supersonic_shooting_visual_validation_6969b4f1bf import (
    reconstruct_shooting_fields,
)
from scripts.evaluation.evaluate_validate_pinn_supersonic_offanchor_gep_shooting import (
    PINNChart,
    complex_fit_metrics,
    interpolate_complex,
    spectral_distance,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.07, 0.10, 0.13, 0.16, 0.19],
    )

    parser.add_argument("--comparison-ymax", type=float, default=20.0)
    parser.add_argument("--match-y", type=float, default=1.0)

    parser.add_argument("--use-mapping", action="store_true")
    parser.add_argument("--mapping-scale", type=float, default=3.0)
    parser.add_argument("--min-y-limit", type=float, default=20.0)
    parser.add_argument("--max-y-limit", type=float, default=2000.0)
    parser.add_argument("--y-limit-factor", type=float, default=6.0)

    parser.add_argument("--cr-half-width", type=float, default=0.005)
    parser.add_argument("--ci-half-width", type=float, default=0.002)
    parser.add_argument(
        "--ci-relative-half-width",
        type=float,
        default=0.25,
    )

    parser.add_argument("--shooting-max-iter", type=int, default=10)
    parser.add_argument("--shooting-grid-size", type=int, default=5)

    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fields_dir = args.output_dir / "fields"
    fields_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    chart = PINNChart(args.checkpoint, device)

    dataset_path = Path(chart.config["dataset"])
    data = np.load(dataset_path, allow_pickle=True)

    anchor_alpha = np.asarray(data["alpha_anchors"], dtype=float)
    cr_ref_all = np.asarray(data["cr_ref"], dtype=float)
    ci_ref_all = np.asarray(data["ci_ref"], dtype=float)

    row_alpha = np.asarray(data["row_alpha"], dtype=float)
    row_y = np.asarray(data["y"], dtype=float)

    row_p = (
        np.asarray(data["p_real"], dtype=float)
        + 1j * np.asarray(data["p_imag"], dtype=float)
    )

    rows: list[dict] = []

    for requested_alpha in args.alphas:
        anchor_index = int(
            np.argmin(np.abs(anchor_alpha - requested_alpha))
        )

        alpha = float(anchor_alpha[anchor_index])

        if abs(alpha - requested_alpha) > 1.0e-10:
            raise RuntimeError(
                f"Requested alpha={requested_alpha} is not an exact anchor. "
                f"Nearest anchor={alpha}."
            )

        cr_ref = float(cr_ref_all[anchor_index])
        ci_ref = float(ci_ref_all[anchor_index])

        modal_mask = np.isclose(
            row_alpha,
            alpha,
            atol=1.0e-12,
            rtol=0.0,
        )

        if np.count_nonzero(modal_mask) < 3:
            raise RuntimeError(
                f"Not enough modal reference rows for alpha={alpha}."
            )

        y_ref = row_y[modal_mask]
        p_ref = row_p[modal_mask]

        order = np.argsort(y_ref)
        y_ref = y_ref[order]
        p_ref = p_ref[order]

        core_mask = np.abs(y_ref) <= args.comparison_ymax

        pinn = chart.predict(alpha, y_ref)

        pinn_fit = complex_fit_metrics(
            p_ref,
            pinn["p"],
            mask=core_mask,
        )

        ci_width = max(
            args.ci_half_width,
            args.ci_relative_half_width * abs(ci_ref),
        )

        solver = Mstab17SupersonicSolver(
            alpha=alpha,
            Mach=chart.Mach,
            match_y=args.match_y,
            use_mapping=args.use_mapping,
            mapping_scale=args.mapping_scale,
            min_y_limit=args.min_y_limit,
            max_y_limit=args.max_y_limit,
            y_limit_factor=args.y_limit_factor,
        )

        shooting = solver.solve(
            cr_min=max(0.0, cr_ref - args.cr_half_width),
            cr_max=cr_ref + args.cr_half_width,
            ci_min=max(1.0e-6, ci_ref - ci_width),
            ci_max=ci_ref + ci_width,
            max_iter=args.shooting_max_iter,
            grid_size=args.shooting_grid_size,
        )

        fields = reconstruct_shooting_fields(
            alpha=alpha,
            mach=chart.Mach,
            cr=float(shooting.cr),
            ci=float(shooting.ci),
            ln_p_start_right=float(shooting.ln_p_start_right),
            match_y=args.match_y,
            use_mapping=args.use_mapping,
            mapping_scale=args.mapping_scale,
            min_y_limit=args.min_y_limit,
            max_y_limit=args.max_y_limit,
            y_limit_factor=args.y_limit_factor,
        )

        shooting_y = np.asarray(fields["y"], dtype=float)
        shooting_p = np.asarray(fields["p"], dtype=np.complex128)

        finite = (
            np.isfinite(shooting_y)
            & np.isfinite(shooting_p.real)
            & np.isfinite(shooting_p.imag)
        )

        shooting_y = shooting_y[finite]
        shooting_p = shooting_p[finite]

        order = np.argsort(shooting_y)
        shooting_y = shooting_y[order]
        shooting_p = shooting_p[order]

        unique_y, inverse, counts = np.unique(
            shooting_y,
            return_inverse=True,
            return_counts=True,
        )

        if unique_y.size != shooting_y.size:
            pressure_sum = np.zeros(
                unique_y.shape,
                dtype=np.complex128,
            )
            np.add.at(pressure_sum, inverse, shooting_p)
            shooting_p = pressure_sum / counts
            shooting_y = unique_y

        shooting_p_on_ref = interpolate_complex(
            y_ref,
            shooting_y,
            shooting_p,
        )

        shooting_fit = complex_fit_metrics(
            p_ref,
            shooting_p_on_ref,
            mask=core_mask,
        )

        shoot_ref_distance = spectral_distance(
            float(shooting.cr),
            float(shooting.ci),
            cr_ref,
            ci_ref,
            ci_weight=2.0,
        )

        result = {
            "Mach": chart.Mach,
            "alpha": alpha,
            "cr_ref": cr_ref,
            "ci_ref": ci_ref,
            "pinn_cr": pinn["cr"],
            "pinn_ci": pinn["ci"],
            "shooting_cr": float(shooting.cr),
            "shooting_ci": float(shooting.ci),
            "shooting_stage1_mismatch": float(
                shooting.stage1_mismatch
            ),
            "shooting_stage2_mismatch": float(
                shooting.stage2_mismatch
            ),
            "shooting_spectral_distance_to_reference": (
                shoot_ref_distance
            ),
            "pinn_reference_p_overlap": (
                pinn_fit["p_overlap"]
            ),
            "pinn_reference_p_rel_after_fit": (
                pinn_fit["p_rel_after_fit"]
            ),
            "shooting_reference_p_overlap": (
                shooting_fit["p_overlap"]
            ),
            "shooting_reference_p_rel_after_fit": (
                shooting_fit["p_rel_after_fit"]
            ),
            "pinn_reference_modal_valid": bool(
                pinn_fit["p_overlap"] >= 0.99
                and pinn_fit["p_rel_after_fit"] <= 0.02
            ),
            "shooting_reference_modal_valid": bool(
                shooting_fit["p_overlap"] >= 0.90
                and shooting_fit["p_rel_after_fit"] <= 0.30
            ),
        }

        rows.append(result)

        print(
            f"\nalpha={alpha:.5f}"
            f"\n  spectral distance shooting/ref = "
            f"{shoot_ref_distance:.6e}"
            f"\n  PINN/ref: overlap={pinn_fit['p_overlap']:.6f}, "
            f"p_rel={pinn_fit['p_rel_after_fit']:.6e}"
            f"\n  shooting/ref: overlap="
            f"{shooting_fit['p_overlap']:.6f}, "
            f"p_rel={shooting_fit['p_rel_after_fit']:.6e}"
        )

        np.savez_compressed(
            fields_dir / f"anchor_alpha_{alpha:.5f}.npz",
            Mach=np.array(chart.Mach),
            alpha=np.array(alpha),
            cr_ref=np.array(cr_ref),
            ci_ref=np.array(ci_ref),
            shooting_cr=np.array(shooting.cr),
            shooting_ci=np.array(shooting.ci),
            y_ref=y_ref,
            p_ref_real=p_ref.real,
            p_ref_imag=p_ref.imag,
            pinn_p_real=pinn["p"].real,
            pinn_p_imag=pinn["p"].imag,
            shooting_p_real=shooting_p_on_ref.real,
            shooting_p_imag=shooting_p_on_ref.imag,
        )

        pd.DataFrame(rows).to_csv(
            args.output_dir / "anchor_modal_control.csv",
            index=False,
        )

    frame = pd.DataFrame(rows)

    report = {
        "n_cases": int(len(frame)),
        "n_pinn_reference_modal_valid": int(
            frame["pinn_reference_modal_valid"].sum()
        ),
        "n_shooting_reference_modal_valid": int(
            frame["shooting_reference_modal_valid"].sum()
        ),
        "median_pinn_reference_p_rel": float(
            frame["pinn_reference_p_rel_after_fit"].median()
        ),
        "median_shooting_reference_p_rel": float(
            frame["shooting_reference_p_rel_after_fit"].median()
        ),
        "interpretation": {
            "shooting_bad_at_anchors": (
                "The shooting modal reconstruction is incompatible "
                "with the classical modal dataset."
            ),
            "shooting_good_at_anchors_only": (
                "The shooting reconstruction is valid, but the PINN "
                "modal interpolation must be investigated off-anchor."
            ),
        },
    }

    (
        args.output_dir / "anchor_modal_control_report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print(frame.to_string(index=False))

    print("\nREPORT")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
