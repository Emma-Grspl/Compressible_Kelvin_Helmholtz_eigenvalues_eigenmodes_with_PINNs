#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.evaluation.evaluate_validate_pinn_supersonic_offanchor_gep_shooting import (
    PINNChart,
    complex_fit_metrics,
    interpolate_complex,
)
from scripts.audits.audit_supersonic_shooting_visual_validation_6969b4f1bf import (
    reconstruct_shooting_fields,
)


EPS = 1.0e-30


def deduplicate_and_sort(
    y: np.ndarray,
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    field = np.asarray(field, dtype=np.complex128)

    finite = (
        np.isfinite(y)
        & np.isfinite(field.real)
        & np.isfinite(field.imag)
    )
    y = y[finite]
    field = field[finite]

    order = np.argsort(y)
    y = y[order]
    field = field[order]

    unique_y, inverse, counts = np.unique(
        y,
        return_inverse=True,
        return_counts=True,
    )

    if unique_y.size != y.size:
        field_sum = np.zeros(
            unique_y.shape,
            dtype=np.complex128,
        )
        np.add.at(field_sum, inverse, field)
        field = field_sum / counts
        y = unique_y

    if y.size < 3:
        raise RuntimeError(
            f"Only {y.size} usable points in shooting branch."
        )

    if np.any(np.diff(y) <= 0.0):
        raise RuntimeError(
            "Non-increasing y grid after sorting and deduplication."
        )

    return y, field


def split_shooting_branches(
    y_raw: np.ndarray,
    p_raw: np.ndarray,
    match_y: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
]:
    """
    Recover the two concatenated shooting branches.

    Preferred case:
      the raw array has a restart or duplicate at the matching point.

    Fallback:
      split geometrically around match_y.
    """
    y_raw = np.asarray(y_raw, dtype=float)
    p_raw = np.asarray(p_raw, dtype=np.complex128)

    finite = (
        np.isfinite(y_raw)
        & np.isfinite(p_raw.real)
        & np.isfinite(p_raw.imag)
    )
    y_raw = y_raw[finite]
    p_raw = p_raw[finite]

    if y_raw.size < 6:
        raise RuntimeError("Too few raw shooting points.")

    dy_raw = np.diff(y_raw)
    restart_indices = np.flatnonzero(dy_raw <= 0.0)

    if restart_indices.size:
        scores = (
            np.abs(y_raw[restart_indices] - match_y)
            + np.abs(y_raw[restart_indices + 1] - match_y)
        )
        split_index = int(
            restart_indices[np.argmin(scores)]
        )

        y_a = y_raw[: split_index + 1]
        p_a = p_raw[: split_index + 1]

        y_b = y_raw[split_index + 1 :]
        p_b = p_raw[split_index + 1 :]

        y_a, p_a = deduplicate_and_sort(y_a, p_a)
        y_b, p_b = deduplicate_and_sort(y_b, p_b)

        # Assign the branch with the lower median y to the left.
        if np.median(y_a) <= np.median(y_b):
            y_left, p_left = y_a, p_a
            y_right, p_right = y_b, p_b
        else:
            y_left, p_left = y_b, p_b
            y_right, p_right = y_a, p_a

        method = "raw_restart_split"

    else:
        left_mask = y_raw <= match_y
        right_mask = y_raw >= match_y

        if np.count_nonzero(left_mask) < 3:
            raise RuntimeError(
                "Too few points on the left of match_y."
            )

        if np.count_nonzero(right_mask) < 3:
            raise RuntimeError(
                "Too few points on the right of match_y."
            )

        y_left, p_left = deduplicate_and_sort(
            y_raw[left_mask],
            p_raw[left_mask],
        )
        y_right, p_right = deduplicate_and_sort(
            y_raw[right_mask],
            p_raw[right_mask],
        )

        method = "geometric_split"

    # Remove any large crossing of the matching point.
    tolerance = max(
        1.0e-10,
        1.0e-8 * max(1.0, abs(match_y)),
    )

    left_keep = y_left <= match_y + tolerance
    right_keep = y_right >= match_y - tolerance

    y_left = y_left[left_keep]
    p_left = p_left[left_keep]

    y_right = y_right[right_keep]
    p_right = p_right[right_keep]

    if y_left.size < 3 or y_right.size < 3:
        raise RuntimeError(
            "Branch split produced fewer than three points "
            "on one side."
        )

    return (
        y_left,
        p_left,
        y_right,
        p_right,
        method,
    )


def extrapolate_to_match(
    y: np.ndarray,
    field: np.ndarray,
    match_y: float,
    *,
    side: str,
    n_fit: int = 6,
) -> complex:
    """
    Estimate the one-sided value at match_y using an exact point
    when available, otherwise a local linear complex fit.
    """
    tolerance = max(
        1.0e-10,
        1.0e-8 * max(1.0, abs(match_y)),
    )

    exact = np.abs(y - match_y) <= tolerance

    if np.any(exact):
        return complex(np.mean(field[exact]))

    count = min(n_fit, y.size)

    if side == "left":
        indices = np.arange(y.size - count, y.size)
    elif side == "right":
        indices = np.arange(count)
    else:
        raise ValueError(side)

    x_local = y[indices] - match_y
    z_local = field[indices]

    degree = 1 if count >= 2 else 0

    real_coefficients = np.polyfit(
        x_local,
        z_local.real,
        deg=degree,
    )
    imag_coefficients = np.polyfit(
        x_local,
        z_local.imag,
        deg=degree,
    )

    return complex(
        np.polyval(real_coefficients, 0.0),
        np.polyval(imag_coefficients, 0.0),
    )


def branch_derivative(
    y: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    return np.gradient(
        p,
        y,
        edge_order=2,
    )


def complex_relative_jump(
    left: complex,
    right: complex,
) -> float:
    return float(
        abs(left - right)
        / max(
            abs(left),
            abs(right),
            EPS,
        )
    )


def build_stitched_field(
    y_left: np.ndarray,
    p_left: np.ndarray,
    q_left: np.ndarray,
    y_right: np.ndarray,
    p_right_scaled: np.ndarray,
    q_right_scaled: np.ndarray,
    *,
    match_y: float,
    p_left_match: complex,
    p_right_match_scaled: complex,
    q_left_match: complex,
    q_right_match_scaled: complex,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    left_keep = y_left < match_y
    right_keep = y_right > match_y

    p_match = 0.5 * (
        p_left_match
        + p_right_match_scaled
    )
    q_match = 0.5 * (
        q_left_match
        + q_right_match_scaled
    )

    y_global = np.concatenate(
        [
            y_left[left_keep],
            np.array([match_y]),
            y_right[right_keep],
        ]
    )

    p_global = np.concatenate(
        [
            p_left[left_keep],
            np.array([p_match]),
            p_right_scaled[right_keep],
        ]
    )

    q_global = np.concatenate(
        [
            q_left[left_keep],
            np.array([q_match]),
            q_right_scaled[right_keep],
        ]
    )

    return deduplicate_and_sort_three(
        y_global,
        p_global,
        q_global,
    )


def deduplicate_and_sort_three(
    y: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    order = np.argsort(y)
    y = np.asarray(y, dtype=float)[order]
    p = np.asarray(p, dtype=np.complex128)[order]
    q = np.asarray(q, dtype=np.complex128)[order]

    unique_y, inverse, counts = np.unique(
        y,
        return_inverse=True,
        return_counts=True,
    )

    if unique_y.size != y.size:
        p_sum = np.zeros(
            unique_y.shape,
            dtype=np.complex128,
        )
        q_sum = np.zeros(
            unique_y.shape,
            dtype=np.complex128,
        )

        np.add.at(p_sum, inverse, p)
        np.add.at(q_sum, inverse, q)

        p = p_sum / counts
        q = q_sum / counts
        y = unique_y

    return y, p, q


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--validation-summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--match-y",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--comparison-ymax",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--n-y",
        type=int,
        default=2001,
    )

    parser.add_argument(
        "--use-mapping",
        action="store_true",
    )
    parser.add_argument(
        "--mapping-scale",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--min-y-limit",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--max-y-limit",
        type=float,
        default=2000.0,
    )
    parser.add_argument(
        "--y-limit-factor",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--modal-overlap-min",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--p-rel-max",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--q-rel-max",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--q-match-jump-max",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    fields_dir = args.output_dir / "fields"
    fields_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if (
            args.device == "cuda"
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    chart = PINNChart(
        args.checkpoint,
        device,
    )

    summary = pd.read_csv(
        args.validation_summary
    )

    required = [
        "alpha",
        "shooting_cr",
        "shooting_ci",
        "shooting_ln_p_start_right",
    ]

    missing = [
        column
        for column in required
        if column not in summary.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns in validation summary: {missing}\n"
            f"Available columns: {summary.columns.tolist()}"
        )

    y_compare = np.linspace(
        -args.comparison_ymax,
        args.comparison_ymax,
        args.n_y,
    )

    rows: list[dict] = []

    for case_index, source_row in summary.iterrows():
        alpha = float(source_row["alpha"])
        cr = float(source_row["shooting_cr"])
        ci = float(source_row["shooting_ci"])
        ln_p_start_right = float(
            source_row["shooting_ln_p_start_right"]
        )

        print(
            f"\n[case {case_index}] "
            f"alpha={alpha:.8f} "
            f"c={cr:.8f}+i{ci:.8f}"
        )

        shooting_fields = reconstruct_shooting_fields(
            alpha=alpha,
            mach=chart.Mach,
            cr=cr,
            ci=ci,
            ln_p_start_right=ln_p_start_right,
            match_y=args.match_y,
            use_mapping=args.use_mapping,
            mapping_scale=args.mapping_scale,
            min_y_limit=args.min_y_limit,
            max_y_limit=args.max_y_limit,
            y_limit_factor=args.y_limit_factor,
        )

        y_raw = np.asarray(
            shooting_fields["y"],
            dtype=float,
        )
        p_raw = np.asarray(
            shooting_fields["p"],
            dtype=np.complex128,
        )

        (
            y_left,
            p_left,
            y_right,
            p_right,
            split_method,
        ) = split_shooting_branches(
            y_raw,
            p_raw,
            args.match_y,
        )

        q_left = branch_derivative(
            y_left,
            p_left,
        )
        q_right = branch_derivative(
            y_right,
            p_right,
        )

        p_left_match = extrapolate_to_match(
            y_left,
            p_left,
            args.match_y,
            side="left",
        )
        p_right_match = extrapolate_to_match(
            y_right,
            p_right,
            args.match_y,
            side="right",
        )

        q_left_match = extrapolate_to_match(
            y_left,
            q_left,
            args.match_y,
            side="left",
        )
        q_right_match = extrapolate_to_match(
            y_right,
            q_right,
            args.match_y,
            side="right",
        )

        if abs(p_right_match) <= EPS:
            raise RuntimeError(
                f"Right pressure is zero at match point "
                f"for alpha={alpha}."
            )

        pressure_scale = (
            p_left_match
            / p_right_match
        )

        p_right_scaled = (
            pressure_scale * p_right
        )
        q_right_scaled = (
            pressure_scale * q_right
        )

        p_right_match_scaled = (
            pressure_scale
            * p_right_match
        )
        q_right_match_scaled = (
            pressure_scale
            * q_right_match
        )

        pressure_jump_before = (
            complex_relative_jump(
                p_left_match,
                p_right_match,
            )
        )
        pressure_jump_after = (
            complex_relative_jump(
                p_left_match,
                p_right_match_scaled,
            )
        )
        q_jump_after = (
            complex_relative_jump(
                q_left_match,
                q_right_match_scaled,
            )
        )

        derivative_scale = (
            q_left_match / q_right_match
            if abs(q_right_match) > EPS
            else complex(np.nan, np.nan)
        )

        scale_disagreement = float(
            abs(
                pressure_scale
                - derivative_scale
            )
            / max(
                abs(pressure_scale),
                abs(derivative_scale),
                EPS,
            )
        )

        (
            y_stitched,
            p_stitched,
            q_stitched,
        ) = build_stitched_field(
            y_left,
            p_left,
            q_left,
            y_right,
            p_right_scaled,
            q_right_scaled,
            match_y=args.match_y,
            p_left_match=p_left_match,
            p_right_match_scaled=(
                p_right_match_scaled
            ),
            q_left_match=q_left_match,
            q_right_match_scaled=(
                q_right_match_scaled
            ),
        )

        p_shoot_on_compare = interpolate_complex(
            y_compare,
            y_stitched,
            p_stitched,
        )
        q_shoot_on_compare = interpolate_complex(
            y_compare,
            y_stitched,
            q_stitched,
        )

        pinn = chart.predict(
            alpha,
            y_compare,
        )

        fit = complex_fit_metrics(
            pinn["p"],
            p_shoot_on_compare,
            target_q=pinn["q"],
            candidate_q=q_shoot_on_compare,
        )

        # Independent consistency of the final stitched pressure field.
        q_from_global_gradient = np.gradient(
            p_shoot_on_compare,
            y_compare,
            edge_order=2,
        )

        # Exclude a small matching neighbourhood for this diagnostic:
        # a remaining q jump would otherwise dominate one grid cell.
        exclude_width = max(
            5.0 * np.median(np.diff(y_compare)),
            1.0e-2,
        )
        away_from_match = (
            np.abs(y_compare - args.match_y)
            > exclude_width
        )

        q_global_consistency = float(
            np.linalg.norm(
                (
                    q_from_global_gradient
                    - q_shoot_on_compare
                )[away_from_match]
            )
            / max(
                np.linalg.norm(
                    q_shoot_on_compare[
                        away_from_match
                    ]
                ),
                EPS,
            )
        )

        modal_validated = bool(
            fit["p_overlap"]
            >= args.modal_overlap_min
            and fit["p_rel_after_fit"]
            <= args.p_rel_max
            and fit["q_rel_after_fit"]
            <= args.q_rel_max
            and q_jump_after
            <= args.q_match_jump_max
        )

        improved_p = bool(
            fit["p_rel_after_fit"]
            < float(
                source_row.get(
                    "shooting_pinn_p_rel_after_fit",
                    np.inf,
                )
            )
        )
        improved_q = bool(
            fit["q_rel_after_fit"]
            < float(
                source_row.get(
                    "shooting_pinn_q_rel_after_fit",
                    np.inf,
                )
            )
        )

        if modal_validated:
            status = "modal_stitch_validated"
        elif (
            pressure_jump_after <= 1.0e-8
            and q_jump_after > args.q_match_jump_max
        ):
            status = (
                "pressure_stitched_but_derivative_mismatch_remains"
            )
        elif improved_p or improved_q:
            status = "stitch_improves_mode_but_not_validated"
        else:
            status = "stitch_does_not_improve_mode"

        result = {
            "case_index": int(case_index),
            "Mach": chart.Mach,
            "alpha": alpha,
            "shooting_cr": cr,
            "shooting_ci": ci,
            "split_method": split_method,
            "n_left": int(y_left.size),
            "n_right": int(y_right.size),
            "match_y": args.match_y,
            "p_left_match_real": float(
                p_left_match.real
            ),
            "p_left_match_imag": float(
                p_left_match.imag
            ),
            "p_right_match_real": float(
                p_right_match.real
            ),
            "p_right_match_imag": float(
                p_right_match.imag
            ),
            "q_left_match_real": float(
                q_left_match.real
            ),
            "q_left_match_imag": float(
                q_left_match.imag
            ),
            "q_right_match_real": float(
                q_right_match.real
            ),
            "q_right_match_imag": float(
                q_right_match.imag
            ),
            "pressure_scale_real": float(
                pressure_scale.real
            ),
            "pressure_scale_imag": float(
                pressure_scale.imag
            ),
            "derivative_scale_real": float(
                derivative_scale.real
            ),
            "derivative_scale_imag": float(
                derivative_scale.imag
            ),
            "scale_disagreement": (
                scale_disagreement
            ),
            "pressure_jump_before": (
                pressure_jump_before
            ),
            "pressure_jump_after": (
                pressure_jump_after
            ),
            "q_jump_after": q_jump_after,
            "stitched_p_overlap": (
                fit["p_overlap"]
            ),
            "stitched_p_rel_after_fit": (
                fit["p_rel_after_fit"]
            ),
            "stitched_q_rel_after_fit": (
                fit["q_rel_after_fit"]
            ),
            "stitched_q_global_consistency": (
                q_global_consistency
            ),
            "original_p_overlap": float(
                source_row.get(
                    "shooting_pinn_p_overlap",
                    np.nan,
                )
            ),
            "original_p_rel_after_fit": float(
                source_row.get(
                    "shooting_pinn_p_rel_after_fit",
                    np.nan,
                )
            ),
            "original_q_rel_after_fit": float(
                source_row.get(
                    "shooting_pinn_q_rel_after_fit",
                    np.nan,
                )
            ),
            "improved_p": improved_p,
            "improved_q": improved_q,
            "modal_stitch_validated": (
                modal_validated
            ),
            "status": status,
        }

        rows.append(result)

        print(
            "  jump p: "
            f"{pressure_jump_before:.3e} -> "
            f"{pressure_jump_after:.3e}"
        )
        print(
            "  jump q after p scaling: "
            f"{q_jump_after:.3e}"
        )
        print(
            "  scale disagreement: "
            f"{scale_disagreement:.3e}"
        )
        print(
            "  PINN fit: "
            f"overlap={fit['p_overlap']:.5f} "
            f"p_rel={fit['p_rel_after_fit']:.3e} "
            f"q_rel={fit['q_rel_after_fit']:.3e}"
        )
        print("  status:", status)

        np.savez_compressed(
            fields_dir
            / (
                f"stitch_case_{case_index:03d}_"
                f"alpha_{alpha:.8f}.npz"
            ),
            Mach=np.array(chart.Mach),
            alpha=np.array(alpha),
            cr=np.array(cr),
            ci=np.array(ci),
            match_y=np.array(args.match_y),
            y_left=y_left,
            p_left_real=p_left.real,
            p_left_imag=p_left.imag,
            q_left_real=q_left.real,
            q_left_imag=q_left.imag,
            y_right=y_right,
            p_right_raw_real=p_right.real,
            p_right_raw_imag=p_right.imag,
            q_right_raw_real=q_right.real,
            q_right_raw_imag=q_right.imag,
            p_right_scaled_real=(
                p_right_scaled.real
            ),
            p_right_scaled_imag=(
                p_right_scaled.imag
            ),
            q_right_scaled_real=(
                q_right_scaled.real
            ),
            q_right_scaled_imag=(
                q_right_scaled.imag
            ),
            pressure_scale=np.array(
                pressure_scale
            ),
            derivative_scale=np.array(
                derivative_scale
            ),
            y_stitched=y_stitched,
            p_stitched_real=p_stitched.real,
            p_stitched_imag=p_stitched.imag,
            q_stitched_real=q_stitched.real,
            q_stitched_imag=q_stitched.imag,
            y_compare=y_compare,
            pinn_p_real=pinn["p"].real,
            pinn_p_imag=pinn["p"].imag,
            pinn_q_real=pinn["q"].real,
            pinn_q_imag=pinn["q"].imag,
            shooting_p_compare_real=(
                p_shoot_on_compare.real
            ),
            shooting_p_compare_imag=(
                p_shoot_on_compare.imag
            ),
            shooting_q_compare_real=(
                q_shoot_on_compare.real
            ),
            shooting_q_compare_imag=(
                q_shoot_on_compare.imag
            ),
        )

        pd.DataFrame(rows).to_csv(
            args.output_dir
            / "shooting_complex_stitch_summary.csv",
            index=False,
        )

    result_frame = pd.DataFrame(rows)

    report = {
        "n_cases": int(len(result_frame)),
        "n_modal_stitch_validated": int(
            result_frame[
                "modal_stitch_validated"
            ].sum()
        ),
        "n_improved_p": int(
            result_frame["improved_p"].sum()
        ),
        "n_improved_q": int(
            result_frame["improved_q"].sum()
        ),
        "median_pressure_jump_before": float(
            result_frame[
                "pressure_jump_before"
            ].median()
        ),
        "median_pressure_jump_after": float(
            result_frame[
                "pressure_jump_after"
            ].median()
        ),
        "median_q_jump_after": float(
            result_frame[
                "q_jump_after"
            ].median()
        ),
        "median_p_overlap": float(
            result_frame[
                "stitched_p_overlap"
            ].median()
        ),
        "median_p_rel_after_fit": float(
            result_frame[
                "stitched_p_rel_after_fit"
            ].median()
        ),
        "median_q_rel_after_fit": float(
            result_frame[
                "stitched_q_rel_after_fit"
            ].median()
        ),
        "interpretation": (
            "The right shooting half is multiplied by the "
            "complex pressure matching factor p_left/p_right. "
            "Pressure continuity is imposed by construction; "
            "q continuity remains an independent diagnostic."
        ),
    }

    (
        args.output_dir
        / "shooting_complex_stitch_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[summary]")
    print(
        result_frame[
            [
                "alpha",
                "pressure_jump_before",
                "pressure_jump_after",
                "q_jump_after",
                "scale_disagreement",
                "original_p_rel_after_fit",
                "stitched_p_rel_after_fit",
                "original_q_rel_after_fit",
                "stitched_q_rel_after_fit",
                "stitched_p_overlap",
                "modal_stitch_validated",
                "status",
            ]
        ].to_string(index=False)
    )

    print("\n[report]")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
