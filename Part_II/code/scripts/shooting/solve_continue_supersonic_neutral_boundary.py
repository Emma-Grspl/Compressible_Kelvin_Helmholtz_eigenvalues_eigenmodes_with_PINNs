#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from classical_solver.supersonic.mstab17_supersonic_solver import (
    Mstab17SupersonicSolver,
)


EPS = 1.0e-30


def spectral_distance(
    cr1: float,
    ci1: float,
    cr2: float,
    ci2: float,
    ci_weight: float = 2.0,
) -> float:
    return float(
        np.hypot(
            cr1 - cr2,
            ci_weight * (ci1 - ci2),
        )
    )


def load_seed(
    path: Path,
    alpha: float,
) -> tuple[float, float, float]:
    data = np.load(
        path,
        allow_pickle=True,
    )

    alpha_anchors = np.asarray(
        data["alpha_anchors"],
        dtype=float,
    )

    index = int(
        np.argmin(
            np.abs(
                alpha_anchors - alpha
            )
        )
    )

    if (
        abs(
            alpha_anchors[index]
            - alpha
        )
        > 1.0e-10
    ):
        raise ValueError(
            f"alpha={alpha} is not an exact anchor; "
            f"nearest={alpha_anchors[index]}"
        )

    return (
        float(
            np.asarray(
                data["Mach_fixed"]
            ).item()
        ),
        float(
            np.asarray(
                data["cr_ref"],
                dtype=float,
            )[index]
        ),
        float(
            np.asarray(
                data["ci_ref"],
                dtype=float,
            )[index]
        ),
    )


def make_solver(
    Mach: float,
    alpha: float,
    args: argparse.Namespace,
) -> Mstab17SupersonicSolver:
    return Mstab17SupersonicSolver(
        alpha=float(alpha),
        Mach=float(Mach),
        match_y=args.match_y,
        amplitude_match_y=args.match_y,
        use_mapping=args.use_mapping,
        mapping_scale=args.mapping_scale,
        min_y_limit=args.min_y_limit,
        max_y_limit=args.max_y_limit,
        y_limit_factor=args.y_limit_factor,
        rtol=args.rtol,
        atol=args.atol,
    )


def local_solve(
    Mach: float,
    alpha: float,
    cr_seed: float,
    ci_seed: float,
    args: argparse.Namespace,
):
    best_result = None
    best_meta = None
    best_score = np.inf

    for box_scale in args.retry_scales:
        cr_half_width = (
            args.cr_half_width
            * box_scale
        )

        ci_half_width = max(
            args.ci_half_width,
            args.ci_relative_half_width
            * max(
                ci_seed,
                args.ci_floor,
            ),
        ) * box_scale

        cr_min = max(
            0.0,
            cr_seed - cr_half_width,
        )
        cr_max = (
            cr_seed + cr_half_width
        )

        ci_min = max(
            args.ci_floor,
            ci_seed - ci_half_width,
        )
        ci_max = max(
            ci_min + args.ci_floor,
            ci_seed + ci_half_width,
        )

        metadata = {
            "box_scale": float(
                box_scale
            ),
            "cr_min": float(cr_min),
            "cr_max": float(cr_max),
            "ci_min": float(ci_min),
            "ci_max": float(ci_max),
            "error": "",
        }

        try:
            result = make_solver(
                Mach,
                alpha,
                args,
            ).solve(
                cr_min=cr_min,
                cr_max=cr_max,
                ci_min=ci_min,
                ci_max=ci_max,
                max_iter=args.max_iter,
                tol=args.solve_tol,
                grid_size=args.grid_size,
                constrain_to_initial_box=True,
            )

            seed_distance = (
                spectral_distance(
                    result.cr,
                    result.ci,
                    cr_seed,
                    ci_seed,
                    args.ci_weight,
                )
            )

            accepted = bool(
                result.success
                and result.ci > 0.0
                and result.stage1_mismatch
                <= args.stage1_max
                and result.stage2_mismatch
                <= args.stage2_max
                and seed_distance
                <= (
                    args.max_seed_distance
                    * box_scale
                )
            )

            metadata.update(
                {
                    "accepted": accepted,
                    "seed_distance": (
                        seed_distance
                    ),
                }
            )

            score = (
                result.stage1_mismatch
                + result.stage2_mismatch
                + 1.0e-2
                * seed_distance
            )

            if score < best_score:
                best_result = result
                best_meta = dict(
                    metadata
                )
                best_score = score

            if accepted:
                return result, metadata

        except Exception as error:
            metadata.update(
                {
                    "accepted": False,
                    "error": repr(error),
                }
            )

    if best_meta is None:
        best_meta = {
            "accepted": False,
            "box_scale": np.nan,
            "seed_distance": np.nan,
            "error": (
                "all local attempts failed"
            ),
        }

    return best_result, best_meta


def make_row(
    kind: str,
    Mach: float,
    alpha: float,
    cr_seed: float,
    ci_seed: float,
    result,
    metadata: dict,
    requested_target: bool = True,
) -> dict:
    output = {
        "kind": kind,
        "Mach": float(Mach),
        "alpha": float(alpha),
        "requested_target": bool(
            requested_target
        ),
        "cr_seed": float(cr_seed),
        "ci_seed": float(ci_seed),
        "accepted": bool(
            metadata.get(
                "accepted",
                False,
            )
        ),
        "box_scale": float(
            metadata.get(
                "box_scale",
                np.nan,
            )
        ),
        "seed_distance": float(
            metadata.get(
                "seed_distance",
                np.nan,
            )
        ),
        "error": str(
            metadata.get(
                "error",
                "",
            )
        ),
    }

    if result is None:
        for key in [
            "cr",
            "ci",
            "omega_i",
            "stage1_mismatch",
            "stage2_mismatch",
            "ln_p_start_right",
            "y_limit",
            "min_abs_U_minus_c",
        ]:
            output[key] = np.nan

        output["success"] = False
        return output

    y_probe = np.linspace(
        -20.0,
        20.0,
        4001,
    )

    output.update(
        {
            "cr": float(result.cr),
            "ci": float(result.ci),
            "omega_i": float(
                alpha * result.ci
            ),
            "stage1_mismatch": float(
                result.stage1_mismatch
            ),
            "stage2_mismatch": float(
                result.stage2_mismatch
            ),
            "ln_p_start_right": float(
                result.ln_p_start_right
            ),
            "y_limit": float(
                result.y_limit
            ),
            "min_abs_U_minus_c": float(
                np.min(
                    np.abs(
                        np.tanh(y_probe)
                        - complex(
                            result.cr,
                            result.ci,
                        )
                    )
                )
            ),
            "success": bool(
                result.success
            ),
        }
    )

    return output


def segment_points(
    start: float,
    end: float,
    max_step: float,
) -> np.ndarray:
    number = max(
        1,
        int(
            math.ceil(
                abs(end - start)
                / max_step
            )
        ),
    )

    return np.linspace(
        start,
        end,
        number + 1,
    )[1:]


def fit_neutral_boundary(
    frame: pd.DataFrame,
    number_fit_points: int,
    margin: float,
) -> dict:
    fit_frame = frame.loc[
        frame["accepted"].fillna(False)
        & (frame["ci"] > 0.0)
    ].sort_values(
        "alpha"
    ).tail(
        number_fit_points
    )

    output = {
        "n_fit_points": int(
            len(fit_frame)
        ),
        "alpha_last": np.nan,
        "ci_last": np.nan,
        "alpha_neutral_linear": np.nan,
        "linear_rmse": np.nan,
        "alpha_neutral_power": np.nan,
        "power_A": np.nan,
        "power_beta": np.nan,
        "power_rmse": np.nan,
        "alpha_neutral_selected": np.nan,
        "selected_method": "none",
    }

    if len(fit_frame) < 3:
        return output

    alpha = fit_frame[
        "alpha"
    ].to_numpy(
        dtype=float
    )

    ci = fit_frame[
        "ci"
    ].to_numpy(
        dtype=float
    )

    output["alpha_last"] = float(
        alpha[-1]
    )
    output["ci_last"] = float(
        ci[-1]
    )

    slope, intercept = np.polyfit(
        alpha,
        ci,
        deg=1,
    )

    linear_prediction = (
        slope * alpha + intercept
    )

    output["linear_rmse"] = float(
        np.sqrt(
            np.mean(
                (
                    linear_prediction
                    - ci
                )
                ** 2
            )
        )
    )

    if slope < 0.0:
        alpha_neutral_linear = float(
            -intercept / slope
        )

        if (
            alpha[-1]
            < alpha_neutral_linear
            <= alpha[-1] + margin
        ):
            output[
                "alpha_neutral_linear"
            ] = alpha_neutral_linear

    def power_model(
        alpha_value,
        amplitude,
        alpha_neutral,
        beta,
    ):
        return (
            amplitude
            * np.maximum(
                alpha_neutral
                - alpha_value,
                1.0e-14,
            )
            ** beta
        )

    alpha_neutral_initial = output[
        "alpha_neutral_linear"
    ]

    if not np.isfinite(
        alpha_neutral_initial
    ):
        alpha_neutral_initial = (
            alpha[-1]
            + max(
                0.002,
                2.0
                * (
                    alpha[-1]
                    - alpha[-2]
                ),
            )
        )

    alpha_neutral_initial = float(
        np.clip(
            alpha_neutral_initial,
            alpha[-1] + 1.0e-6,
            alpha[-1]
            + 0.8 * margin,
        )
    )

    amplitude_initial = max(
        ci[-1]
        / max(
            alpha_neutral_initial
            - alpha[-1],
            EPS,
        ),
        EPS,
    )

    try:
        parameters, _ = curve_fit(
            power_model,
            alpha,
            ci,
            p0=[
                amplitude_initial,
                alpha_neutral_initial,
                1.0,
            ],
            bounds=(
                [
                    0.0,
                    alpha[-1]
                    + 1.0e-8,
                    0.2,
                ],
                [
                    np.inf,
                    alpha[-1]
                    + margin,
                    3.0,
                ],
            ),
            maxfev=50000,
        )

        amplitude = float(
            parameters[0]
        )
        alpha_neutral = float(
            parameters[1]
        )
        beta = float(
            parameters[2]
        )

        prediction = power_model(
            alpha,
            *parameters,
        )

        output.update(
            {
                "alpha_neutral_power": (
                    alpha_neutral
                ),
                "power_A": amplitude,
                "power_beta": beta,
                "power_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                prediction
                                - ci
                            )
                            ** 2
                        )
                    )
                ),
            }
        )

    except Exception:
        pass

    if (
        np.isfinite(
            output[
                "alpha_neutral_power"
            ]
        )
        and (
            not np.isfinite(
                output[
                    "alpha_neutral_linear"
                ]
            )
            or output["power_rmse"]
            <= (
                1.25
                * output["linear_rmse"]
            )
        )
    ):
        output[
            "alpha_neutral_selected"
        ] = output[
            "alpha_neutral_power"
        ]
        output[
            "selected_method"
        ] = "power"

    elif np.isfinite(
        output[
            "alpha_neutral_linear"
        ]
    ):
        output[
            "alpha_neutral_selected"
        ] = output[
            "alpha_neutral_linear"
        ]
        output[
            "selected_method"
        ] = "linear"

    return output


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed-dataset",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed-mach",
        type=float,
        default=1.8,
    )
    parser.add_argument(
        "--seed-alpha",
        type=float,
        default=0.13,
    )
    parser.add_argument(
        "--mach-values",
        nargs="+",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--mach-step-max",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--alpha-max",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--alpha-step-min",
        type=float,
        default=5.0e-4,
    )
    parser.add_argument(
        "--alpha-step-max",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--ci-stop",
        type=float,
        default=5.0e-5,
    )

    parser.add_argument(
        "--neutral-fit-points",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--neutral-fit-margin",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--match-y",
        type=float,
        default=1.0,
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
        "--rtol",
        type=float,
        default=1.0e-10,
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-12,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--solve-tol",
        type=float,
        default=1.0e-7,
    )

    parser.add_argument(
        "--cr-half-width",
        type=float,
        default=0.008,
    )
    parser.add_argument(
        "--ci-half-width",
        type=float,
        default=0.003,
    )
    parser.add_argument(
        "--ci-relative-half-width",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--retry-scales",
        nargs="+",
        type=float,
        default=[
            1.0,
            2.0,
            4.0,
        ],
    )

    parser.add_argument(
        "--ci-floor",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "--ci-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--max-seed-distance",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--stage1-max",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--stage2-max",
        type=float,
        default=1.0e-8,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        dataset_mach,
        seed_cr,
        seed_ci,
    ) = load_seed(
        args.seed_dataset,
        args.seed_alpha,
    )

    if (
        abs(
            dataset_mach
            - args.seed_mach
        )
        > 1.0e-10
    ):
        raise ValueError(
            f"dataset M={dataset_mach}, "
            f"requested seed M="
            f"{args.seed_mach}"
        )

    requested_mach = sorted(
        set(
            map(
                float,
                args.mach_values,
            )
        )
        | {
            float(
                args.seed_mach
            )
        }
    )

    config = vars(args).copy()

    config.update(
        {
            "seed_dataset": str(
                args.seed_dataset
            ),
            "output_dir": str(
                args.output_dir
            ),
            "requested_mach": (
                requested_mach
            ),
            "seed_cr": seed_cr,
            "seed_ci": seed_ci,
        }
    )

    (
        args.output_dir
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    mach_rows: list[dict] = []
    alpha_rows: list[dict] = []
    neutral_rows: list[dict] = []

    base_states: dict[
        float,
        object,
    ] = {}

    def flush() -> None:
        pd.DataFrame(
            mach_rows
        ).to_csv(
            args.output_dir
            / "mach_continuation.csv",
            index=False,
        )

        pd.DataFrame(
            alpha_rows
        ).to_csv(
            args.output_dir
            / "alpha_traces.csv",
            index=False,
        )

        pd.DataFrame(
            neutral_rows
        ).to_csv(
            args.output_dir
            / "neutral_boundary.csv",
            index=False,
        )

    seed_result, seed_metadata = (
        local_solve(
            args.seed_mach,
            args.seed_alpha,
            seed_cr,
            seed_ci,
            args,
        )
    )

    seed_row = make_row(
        "mach_seed",
        args.seed_mach,
        args.seed_alpha,
        seed_cr,
        seed_ci,
        seed_result,
        seed_metadata,
    )

    mach_rows.append(
        seed_row
    )
    flush()

    if not seed_row["accepted"]:
        raise RuntimeError(
            "Initial M=1.8 seed "
            "refinement failed."
        )

    base_states[
        round(
            args.seed_mach,
            10,
        )
    ] = seed_result

    def walk_mach(
        targets: list[float],
    ) -> None:
        current_mach = (
            args.seed_mach
        )
        current_result = (
            seed_result
        )

        for target_mach in targets:
            direction_success = True

            for next_mach in segment_points(
                current_mach,
                target_mach,
                args.mach_step_max,
            ):
                result, metadata = (
                    local_solve(
                        next_mach,
                        args.seed_alpha,
                        current_result.cr,
                        current_result.ci,
                        args,
                    )
                )

                result_row = make_row(
                    "mach_continuation",
                    next_mach,
                    args.seed_alpha,
                    current_result.cr,
                    current_result.ci,
                    result,
                    metadata,
                    requested_target=bool(
                        abs(
                            next_mach
                            - target_mach
                        )
                        <= 1.0e-10
                    ),
                )

                mach_rows.append(
                    result_row
                )
                flush()

                print(
                    f"M {current_mach:.5f}"
                    f" -> {next_mach:.5f} "
                    f"accepted="
                    f"{result_row['accepted']} "
                    f"c={result_row['cr']:.8f}"
                    f"+i{result_row['ci']:.8f}"
                )

                if not result_row[
                    "accepted"
                ]:
                    direction_success = False
                    break

                current_mach = float(
                    next_mach
                )
                current_result = result

            if not direction_success:
                break

            base_states[
                round(
                    target_mach,
                    10,
                )
            ] = current_result

    walk_mach(
        sorted(
            [
                Mach
                for Mach
                in requested_mach
                if Mach
                < args.seed_mach
            ],
            reverse=True,
        )
    )

    walk_mach(
        sorted(
            [
                Mach
                for Mach
                in requested_mach
                if Mach
                > args.seed_mach
            ]
        )
    )

    for Mach in requested_mach:
        key = round(
            Mach,
            10,
        )

        if key not in base_states:
            neutral_rows.append(
                {
                    "Mach": Mach,
                    "status": (
                        "missing_base_seed"
                    ),
                    "n_trace_points": 0,
                }
            )
            flush()
            continue

        current_result = (
            base_states[key]
        )
        current_alpha = (
            args.seed_alpha
        )
        alpha_step = (
            args.alpha_step
        )

        history = [
            (
                current_alpha,
                current_result.cr,
                current_result.ci,
            )
        ]

        trace_rows = []

        base_metadata = {
            "accepted": True,
            "box_scale": 0.0,
            "seed_distance": 0.0,
            "error": "",
        }

        first_row = make_row(
            "alpha_trace",
            Mach,
            current_alpha,
            current_result.cr,
            current_result.ci,
            current_result,
            base_metadata,
        )

        first_row.update(
            {
                "point_index": 0,
                "alpha_step_used": 0.0,
            }
        )

        alpha_rows.append(
            first_row
        )
        trace_rows.append(
            first_row
        )
        flush()

        status = (
            "alpha_limit_reached"
        )
        point_index = 1

        print(
            f"\nalpha trace M={Mach:.5f}"
        )

        while (
            current_alpha
            < args.alpha_max
            - 1.0e-12
        ):
            trial_alpha = min(
                current_alpha
                + alpha_step,
                args.alpha_max,
            )

            delta_alpha = (
                trial_alpha
                - current_alpha
            )

            if len(history) >= 2:
                (
                    alpha0,
                    cr0,
                    ci0,
                ) = history[-2]

                (
                    alpha1,
                    cr1,
                    ci1,
                ) = history[-1]

                denominator = max(
                    alpha1 - alpha0,
                    EPS,
                )

                trial_cr_seed = (
                    cr1
                    + (
                        cr1 - cr0
                    )
                    * delta_alpha
                    / denominator
                )

                trial_ci_seed = max(
                    args.ci_floor,
                    ci1
                    + (
                        ci1 - ci0
                    )
                    * delta_alpha
                    / denominator,
                )

            else:
                trial_cr_seed = (
                    current_result.cr
                )
                trial_ci_seed = (
                    current_result.ci
                )

            result, metadata = (
                local_solve(
                    Mach,
                    trial_alpha,
                    trial_cr_seed,
                    trial_ci_seed,
                    args,
                )
            )

            result_row = make_row(
                "alpha_trace",
                Mach,
                trial_alpha,
                trial_cr_seed,
                trial_ci_seed,
                result,
                metadata,
            )

            result_row.update(
                {
                    "point_index": (
                        point_index
                    ),
                    "alpha_step_used": (
                        delta_alpha
                    ),
                }
            )

            if not result_row[
                "accepted"
            ]:
                new_step = (
                    0.5 * alpha_step
                )

                if (
                    new_step
                    >= args.alpha_step_min
                    - 1.0e-14
                ):
                    print(
                        "retry alpha="
                        f"{trial_alpha:.8f}: "
                        f"step {alpha_step:.6f}"
                        f" -> {new_step:.6f}"
                    )

                    alpha_step = max(
                        args.alpha_step_min,
                        new_step,
                    )
                    continue

                alpha_rows.append(
                    result_row
                )
                trace_rows.append(
                    result_row
                )
                flush()

                status = (
                    "local_continuation_failed"
                )
                break

            alpha_rows.append(
                result_row
            )
            trace_rows.append(
                result_row
            )
            flush()

            print(
                f"alpha={trial_alpha:.8f} "
                f"c={result_row['cr']:.8f}"
                f"+i{result_row['ci']:.8e} "
                f"stage1="
                f"{result_row['stage1_mismatch']:.2e} "
                f"stage2="
                f"{result_row['stage2_mismatch']:.2e}"
            )

            current_alpha = float(
                trial_alpha
            )
            current_result = result

            history.append(
                (
                    current_alpha,
                    result.cr,
                    result.ci,
                )
            )

            point_index += 1

            if result.ci <= args.ci_stop:
                status = (
                    "reached_ci_stop"
                )
                break

            if result.ci < 5.0e-4:
                alpha_step = max(
                    args.alpha_step_min,
                    min(
                        alpha_step,
                        1.0e-3,
                    ),
                )

            elif result.ci < 2.0e-3:
                alpha_step = max(
                    args.alpha_step_min,
                    min(
                        alpha_step,
                        2.5e-3,
                    ),
                )

            elif result.ci < 5.0e-3:
                alpha_step = max(
                    args.alpha_step_min,
                    min(
                        alpha_step,
                        5.0e-3,
                    ),
                )

            else:
                alpha_step = min(
                    args.alpha_step_max,
                    1.2 * alpha_step,
                )

        fit = fit_neutral_boundary(
            pd.DataFrame(
                trace_rows
            ),
            args.neutral_fit_points,
            args.neutral_fit_margin,
        )

        neutral_row = {
            "Mach": Mach,
            "status": status,
            "n_trace_points": int(
                sum(
                    bool(
                        item["accepted"]
                    )
                    for item
                    in trace_rows
                )
            ),
            **fit,
        }

        neutral_rows.append(
            neutral_row
        )
        flush()

        print(
            f"neutral M={Mach:.5f}: "
            f"alpha_n="
            f"{neutral_row['alpha_neutral_selected']} "
            f"method="
            f"{neutral_row['selected_method']} "
            f"status={status}"
        )

    neutral_frame = pd.DataFrame(
        neutral_rows
    )

    report = {
        "n_requested_mach": int(
            len(requested_mach)
        ),
        "n_base_seeds": int(
            len(base_states)
        ),
        "n_neutral_estimates": int(
            np.isfinite(
                neutral_frame.get(
                    "alpha_neutral_selected",
                    pd.Series(
                        dtype=float
                    ),
                )
            ).sum()
        ),
        "status_counts": {
            str(key): int(value)
            for key, value
            in neutral_frame[
                "status"
            ].value_counts().items()
        },
        "scientific_note": (
            "alpha_n(M) is extrapolated "
            "from positive-ci shooting "
            "solutions on the unstable "
            "side; ci=0 is not imposed "
            "in the solver."
        ),
    }

    (
        args.output_dir
        / "report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n",
        neutral_frame.to_string(
            index=False
        ),
    )

    print(
        "\n",
        json.dumps(
            report,
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
