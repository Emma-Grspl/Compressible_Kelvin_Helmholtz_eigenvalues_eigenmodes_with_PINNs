#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CONTINUATION_SCRIPT = (
    ROOT / 'code/scripts/shooting/solve_continue_supersonic_neutral_boundary.py'
)

VALIDATION_SCRIPT = (
    ROOT / 'code/scripts/evaluation/evaluate_validate_pinn_supersonic_offanchor_gep_shooting.py'
)


def load_module(
    path: Path,
    module_name: str,
):
    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            f"Cannot import {path}"
        )

    module = (
        importlib.util.module_from_spec(
            specification
        )
    )

    specification.loader.exec_module(
        module
    )

    return module


def parse_bool(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
    )


def complex_interp(
    x_new: np.ndarray,
    x_old: np.ndarray,
    values_old: np.ndarray,
) -> np.ndarray:
    return (
        np.interp(
            x_new,
            x_old,
            values_old.real,
        )
        + 1j
        * np.interp(
            x_new,
            x_old,
            values_old.imag,
        )
    )


def unique_sorted_fields(
    y: np.ndarray,
    fields: dict[str, np.ndarray],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
]:
    order = np.argsort(y)

    y = np.asarray(
        y[order],
        dtype=float,
    )

    ordered = {
        key: np.asarray(
            value[order],
            dtype=np.complex128,
        )
        for key, value in fields.items()
    }

    unique_y, inverse, counts = np.unique(
        y,
        return_inverse=True,
        return_counts=True,
    )

    if unique_y.size == y.size:
        return y, ordered

    result: dict[str, np.ndarray] = {}

    for key, value in ordered.items():
        total = np.zeros(
            unique_y.shape,
            dtype=np.complex128,
        )

        np.add.at(
            total,
            inverse,
            value,
        )

        result[key] = total / counts

    return unique_y, result


def call_reconstructor(
    function: Callable[..., Any],
    *,
    Mach: float,
    alpha: float,
    shooting_result: Any,
    solver_args: Any,
) -> dict[str, Any]:
    """
    Call the active modal reconstructor using its actual API.

    Active signature:
        reconstruct_shooting_fields(
            *,
            alpha,
            mach,
            cr,
            ci,
            ln_p_start_right,
            match_y,
            use_mapping,
            mapping_scale,
            min_y_limit,
            max_y_limit,
            y_limit_factor,
            max_step=inf,
        )
    """
    required_result_attributes = [
        "cr",
        "ci",
        "ln_p_start_right",
    ]

    missing = [
        name
        for name in required_result_attributes
        if not hasattr(
            shooting_result,
            name,
        )
    ]

    if missing:
        raise AttributeError(
            "Shooting result is missing "
            f"attributes: {missing}"
        )

    output = function(
        alpha=float(alpha),
        mach=float(Mach),
        cr=float(
            shooting_result.cr
        ),
        ci=float(
            shooting_result.ci
        ),
        ln_p_start_right=float(
            shooting_result.ln_p_start_right
        ),
        match_y=float(
            solver_args.match_y
        ),
        use_mapping=bool(
            solver_args.use_mapping
        ),
        mapping_scale=float(
            solver_args.mapping_scale
        ),
        min_y_limit=float(
            solver_args.min_y_limit
        ),
        max_y_limit=float(
            solver_args.max_y_limit
        ),
        y_limit_factor=float(
            solver_args.y_limit_factor
        ),
        max_step=float("inf"),
    )

    if not isinstance(
        output,
        dict,
    ):
        raise TypeError(
            "reconstruct_shooting_fields "
            "must return a dictionary, "
            f"got {type(output)!r}."
        )

    if "y" not in output:
        raise KeyError(
            "Modal reconstruction does not "
            "contain the y grid."
        )

    if "p" not in output:
        raise KeyError(
            "Modal reconstruction does not "
            "contain p."
        )

    if (
        "q" not in output
        and "gamma" not in output
    ):
        raise KeyError(
            "Modal reconstruction contains "
            "neither q nor gamma."
        )

    return output


def prepare_modal_fields(
    reconstructed: dict[str, Any],
    y_target: np.ndarray,
) -> dict[str, np.ndarray]:
    if "p" not in reconstructed:
        raise KeyError(
            "Reconstruction does not contain p."
        )

    raw_p = np.asarray(
        reconstructed["p"],
        dtype=np.complex128,
    )

    if "y" in reconstructed:
        raw_y = np.asarray(
            reconstructed["y"],
            dtype=float,
        )
    elif raw_p.shape == y_target.shape:
        raw_y = y_target.copy()
    else:
        raise KeyError(
            "Reconstruction does not contain y "
            "and p is not on the requested grid."
        )

    if "gamma" in reconstructed:
        raw_gamma = np.asarray(
            reconstructed["gamma"],
            dtype=np.complex128,
        )
    elif "q" in reconstructed:
        raw_q = np.asarray(
            reconstructed["q"],
            dtype=np.complex128,
        )

        raw_gamma = np.divide(
            raw_q,
            raw_p,
            out=np.zeros_like(raw_q),
            where=np.abs(raw_p) > 1.0e-14,
        )
    else:
        raise KeyError(
            "Reconstruction contains neither "
            "gamma nor q."
        )

    if "q" in reconstructed:
        raw_q = np.asarray(
            reconstructed["q"],
            dtype=np.complex128,
        )
    else:
        raw_q = (
            raw_gamma * raw_p
        )

    if not (
        raw_y.shape
        == raw_p.shape
        == raw_q.shape
        == raw_gamma.shape
    ):
        raise RuntimeError(
            "Inconsistent raw field shapes: "
            f"y={raw_y.shape}, "
            f"p={raw_p.shape}, "
            f"q={raw_q.shape}, "
            f"gamma={raw_gamma.shape}"
        )

    raw_y, ordered = (
        unique_sorted_fields(
            raw_y,
            {
                "p": raw_p,
                "q_returned": raw_q,
                "gamma": raw_gamma,
            },
        )
    )

    if (
        y_target[0]
        < raw_y[0] - 1.0e-8
        or y_target[-1]
        > raw_y[-1] + 1.0e-8
    ):
        raise RuntimeError(
            "Reconstructed grid does not cover "
            "the requested target grid."
        )

    p = complex_interp(
        y_target,
        raw_y,
        ordered["p"],
    )

    q_returned = complex_interp(
        y_target,
        raw_y,
        ordered["q_returned"],
    )

    gamma = complex_interp(
        y_target,
        raw_y,
        ordered["gamma"],
    )

    q_gamma = gamma * p

    denominator = max(
        float(
            np.linalg.norm(q_returned)
        ),
        1.0e-30,
    )

    q_consistency = float(
        np.linalg.norm(
            q_returned - q_gamma
        )
        / denominator
    )

    # The atlas convention is exactly q = gamma p.
    q = q_gamma

    return {
        "p": p,
        "q": q,
        "gamma": gamma,
        "q_returned": q_returned,
        "q_consistency_rel": np.array(
            q_consistency,
            dtype=float,
        ),
    }


def l2_norm(
    y: np.ndarray,
    values: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            max(
                np.trapz(
                    np.abs(values) ** 2,
                    y,
                ),
                0.0,
            )
        )
    )


def normalize_fields(
    y: np.ndarray,
    fields: dict[str, np.ndarray],
) -> tuple[
    dict[str, np.ndarray],
    float,
]:
    norm = l2_norm(
        y,
        fields["p"],
    )

    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(
            f"Invalid p norm: {norm}"
        )

    normalized = dict(fields)

    normalized["p"] = (
        fields["p"] / norm
    )

    normalized["q"] = (
        fields["q"] / norm
    )

    normalized["q_returned"] = (
        fields["q_returned"] / norm
    )

    return normalized, norm


def peak_positive_gauge(
    fields: dict[str, np.ndarray],
) -> tuple[
    dict[str, np.ndarray],
    complex,
]:
    index = int(
        np.argmax(
            np.abs(fields["p"])
        )
    )

    value = fields["p"][index]

    if abs(value) <= 0.0:
        phase = 1.0 + 0.0j
    else:
        phase = np.exp(
            -1j * np.angle(value)
        )

    aligned = dict(fields)

    for key in [
        "p",
        "q",
        "q_returned",
    ]:
        aligned[key] = (
            fields[key] * phase
        )

    return aligned, complex(phase)


def overlap_and_phase(
    reference_y: np.ndarray,
    reference_p: np.ndarray,
    current_y: np.ndarray,
    current_p: np.ndarray,
    window: float = 20.0,
) -> tuple[float, complex]:
    lower = max(
        float(reference_y[0]),
        float(current_y[0]),
        -window,
    )

    upper = min(
        float(reference_y[-1]),
        float(current_y[-1]),
        window,
    )

    mask = (
        (current_y >= lower)
        & (current_y <= upper)
    )

    if int(mask.sum()) < 32:
        return np.nan, 1.0 + 0.0j

    y_common = current_y[mask]

    reference_on_current = (
        complex_interp(
            y_common,
            reference_y,
            reference_p,
        )
    )

    current_on_common = (
        current_p[mask]
    )

    overlap = np.trapz(
        np.conjugate(
            reference_on_current
        )
        * current_on_common,
        y_common,
    )

    norm_product = (
        l2_norm(
            y_common,
            reference_on_current,
        )
        * l2_norm(
            y_common,
            current_on_common,
        )
    )

    normalized_overlap = float(
        abs(overlap)
        / max(
            norm_product,
            1.0e-30,
        )
    )

    if abs(overlap) <= 0.0:
        phase = 1.0 + 0.0j
    else:
        phase = np.exp(
            -1j * np.angle(overlap)
        )

    return (
        normalized_overlap,
        complex(phase),
    )


def apply_phase(
    fields: dict[str, np.ndarray],
    phase: complex,
) -> dict[str, np.ndarray]:
    aligned = dict(fields)

    for key in [
        "p",
        "q",
        "q_returned",
    ]:
        aligned[key] = (
            fields[key] * phase
        )

    return aligned


def finite_complex(
    values: np.ndarray,
) -> bool:
    return bool(
        np.all(
            np.isfinite(values.real)
        )
        and np.all(
            np.isfinite(values.imag)
        )
    )


def select_asymptotic_gamma(
    *,
    alpha: float,
    Mach: float,
    c: complex,
    U_infinity: float,
    side: str,
) -> complex:
    """
    Far-field roots of

        gamma^2 =
        alpha^2 [1 - M^2 (U_infinity - c)^2].

    The decaying root has Re(gamma)>0 on the left
    and Re(gamma)<0 on the right.
    """
    root = complex(
        alpha
        * np.sqrt(
            1.0
            - Mach**2
            * (
                U_infinity
                - c
            ) ** 2
            + 0.0j
        )
    )

    candidates = (
        root,
        -root,
    )

    if side == "left":
        return max(
            candidates,
            key=lambda value: (
                value.real,
                -abs(value.imag),
            ),
        )

    if side == "right":
        return min(
            candidates,
            key=lambda value: (
                value.real,
                abs(value.imag),
            ),
        )

    raise ValueError(
        f"Unknown side: {side}"
    )


def asymptotic_bc_relative_error(
    *,
    alpha: float,
    Mach: float,
    cr: float,
    ci: float,
    p: np.ndarray,
    q: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    complex,
    complex,
]:
    c = complex(
        cr,
        ci,
    )

    gamma_left = (
        select_asymptotic_gamma(
            alpha=alpha,
            Mach=Mach,
            c=c,
            U_infinity=-1.0,
            side="left",
        )
    )

    gamma_right = (
        select_asymptotic_gamma(
            alpha=alpha,
            Mach=Mach,
            c=c,
            U_infinity=1.0,
            side="right",
        )
    )

    residual_left = abs(
        q[0]
        - gamma_left
        * p[0]
    )

    scale_left = (
        abs(q[0])
        + abs(
            gamma_left
            * p[0]
        )
        + 1.0e-30
    )

    residual_right = abs(
        q[-1]
        - gamma_right
        * p[-1]
    )

    scale_right = (
        abs(q[-1])
        + abs(
            gamma_right
            * p[-1]
        )
        + 1.0e-30
    )

    relative_left = float(
        residual_left
        / scale_left
    )

    relative_right = float(
        residual_right
        / scale_right
    )

    relative_max = max(
        relative_left,
        relative_right,
    )

    return (
        relative_max,
        relative_left,
        relative_right,
        gamma_left,
        gamma_right,
    )


def safe_name(
    value: float,
    digits: int,
) -> str:
    return (
        f"{value:.{digits}f}"
        .replace("-", "m")
        .replace(".", "p")
    )


def save_catalog(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--seed-traces",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    selection = parser.add_mutually_exclusive_group(
        required=True
    )

    selection.add_argument(
        "--mach-index",
        type=int,
    )
    selection.add_argument(
        "--mach-value",
        type=float,
    )

    parser.add_argument(
        "--s-values",
        nargs="*",
        type=float,
    )

    parser.add_argument(
        "--base-alpha",
        type=float,
        default=0.13,
    )
    parser.add_argument(
        "--continuation-alpha-step-max",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--continuation-alpha-step-min",
        type=float,
        default=0.000625,
    )
    parser.add_argument(
        "--cert-asymptotic-bc",
        type=float,
        default=1.0e-6,
    )

    parser.add_argument(
        "--n-y",
        type=int,
        default=4097,
    )
    parser.add_argument(
        "--mapping-scale",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--tail-points",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--cert-stage1",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--cert-stage2",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "--cert-q-consistency",
        type=float,
        default=5.0e-3,
    )
    parser.add_argument(
        "--cert-tail-ratio",
        type=float,
        default=1.0e-3,
    )

    args = parser.parse_args()

    continuation = load_module(
        CONTINUATION_SCRIPT,
        "supersonic_continuation_module",
    )

    validation = load_module(
        VALIDATION_SCRIPT,
        "supersonic_validation_module",
    )

    if not hasattr(
        validation,
        "reconstruct_shooting_fields",
    ):
        raise AttributeError(
            "The validation script does not "
            "expose reconstruct_shooting_fields."
        )

    reconstruct = getattr(
        validation,
        "reconstruct_shooting_fields",
    )

    print(
        "reconstruct signature:",
        inspect.signature(reconstruct),
        flush=True,
    )

    plan = pd.read_csv(
        args.plan
    )

    mach_values = np.sort(
        plan["Mach"].unique()
    )

    if args.mach_value is not None:
        distances = np.abs(
            mach_values
            - args.mach_value
        )

        index = int(
            np.argmin(distances)
        )

        if distances[index] > 1.0e-8:
            raise ValueError(
                f"Mach {args.mach_value} "
                "is absent from the plan."
            )

        Mach = float(
            mach_values[index]
        )

    else:
        if not (
            0
            <= args.mach_index
            < len(mach_values)
        ):
            raise IndexError(
                f"mach-index={args.mach_index}; "
                f"available=0..{len(mach_values)-1}"
            )

        Mach = float(
            mach_values[
                args.mach_index
            ]
        )

    cases = plan[
        np.isclose(
            plan["Mach"],
            Mach,
            atol=1.0e-10,
        )
    ].copy()

    if args.s_values:
        requested_s = np.asarray(
            args.s_values,
            dtype=float,
        )

        cases = cases[
            cases["s"].apply(
                lambda value: bool(
                    np.any(
                        np.isclose(
                            requested_s,
                            float(value),
                            atol=1.0e-10,
                        )
                    )
                )
            )
        ].copy()

    cases = cases.sort_values("alpha")

    if cases.empty:
        raise RuntimeError(
            f"No cases selected for M={Mach}"
        )

    traces = pd.read_csv(
        args.seed_traces
    )

    accepted = parse_bool(
        traces["accepted"]
    )

    seed_candidates = traces[
        accepted
        & np.isclose(
            traces["Mach"],
            Mach,
            atol=1.0e-8,
        )
        & np.isfinite(
            pd.to_numeric(
                traces["cr"],
                errors="coerce",
            )
        )
        & np.isfinite(
            pd.to_numeric(
                traces["ci"],
                errors="coerce",
            )
        )
    ].copy()

    if seed_candidates.empty:
        raise RuntimeError(
            f"No accepted spectral seed "
            f"for M={Mach}"
        )

    seed_index = (
        seed_candidates["alpha"]
        .sub(args.base_alpha)
        .abs()
        .idxmin()
    )

    seed_row = (
        seed_candidates.loc[
            seed_index
        ]
    )

    base_alpha = float(
        seed_row["alpha"]
    )

    base_cr = float(
        seed_row["cr"]
    )

    base_ci = float(
        seed_row["ci"]
    )

    solver_args = SimpleNamespace(
        match_y=1.0,
        use_mapping=True,
        mapping_scale=3.0,
        min_y_limit=20.0,
        max_y_limit=5000.0,
        y_limit_factor=8.0,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_iter=12,
        solve_tol=1.0e-7,
        grid_size=5,
        cr_half_width=0.012,
        ci_half_width=0.005,
        ci_relative_half_width=0.50,
        retry_scales=[
            1.0,
            2.0,
            4.0,
            8.0,
        ],
        ci_floor=1.0e-9,
        ci_weight=2.0,
        max_seed_distance=0.08,
        stage1_max=2.0e-5,
        stage2_max=1.0e-8,
    )

    mach_name = safe_name(
        Mach,
        3,
    )

    mach_dir = (
        args.output_dir
        / f"M_{mach_name}"
    )

    fields_dir = (
        mach_dir
        / "fields"
    )

    fields_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog_path = (
        mach_dir
        / f"catalog_M_{mach_name}.csv"
    )

    config = {
        "Mach": Mach,
        "base_alpha": base_alpha,
        "base_cr": base_cr,
        "base_ci": base_ci,
        "n_requested_cases": int(
            len(cases)
        ),
        "s_values": (
            cases["s"]
            .astype(float)
            .tolist()
        ),
        "reconstruct_signature": str(
            inspect.signature(
                reconstruct
            )
        ),
    }

    (
        mach_dir
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    base_result, base_metadata = (
        continuation.local_solve(
            Mach,
            base_alpha,
            base_cr,
            base_ci,
            solver_args,
        )
    )

    if (
        base_result is None
        or not bool(
            base_metadata.get(
                "accepted",
                False,
            )
        )
    ):
        raise RuntimeError(
            f"Base solve failed for M={Mach}, "
            f"alpha={base_alpha}: "
            f"{base_metadata}"
        )

    base_limit = float(
        base_result.y_limit
    )

    xi = np.linspace(
        -1.0,
        1.0,
        args.n_y,
    )

    base_y = (
        args.mapping_scale
        * np.sinh(
            xi
            * np.arcsinh(
                base_limit
                / args.mapping_scale
            )
        )
    )

    base_reconstructed = (
        call_reconstructor(
            reconstruct,
            Mach=Mach,
            alpha=base_alpha,
            shooting_result=base_result,
            solver_args=solver_args,
        )
    )

    base_fields = (
        prepare_modal_fields(
            base_reconstructed,
            base_y,
        )
    )

    base_fields, _ = (
        normalize_fields(
            base_y,
            base_fields,
        )
    )

    base_fields, _ = (
        peak_positive_gauge(
            base_fields
        )
    )

    base_reference = {
        "y": base_y,
        "p": base_fields["p"],
    }

    def continue_to_target(
        *,
        start_alpha: float,
        start_result: Any,
        target_alpha: float,
    ) -> tuple[
        Any,
        dict[str, Any],
        dict[str, Any],
    ]:
        current_alpha = float(
            start_alpha
        )

        current_result = (
            start_result
        )

        n_substeps = 0
        n_retries = 0

        last_result = None
        last_metadata: dict[
            str,
            Any,
        ] = {
            "accepted": False,
        }

        while not math.isclose(
            current_alpha,
            target_alpha,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            remaining = (
                target_alpha
                - current_alpha
            )

            direction = (
                1.0
                if remaining > 0.0
                else -1.0
            )

            proposed_step = min(
                abs(remaining),
                args.continuation_alpha_step_max,
            )

            minimum_step = min(
                abs(remaining),
                args.continuation_alpha_step_min,
            )

            trial_step = (
                proposed_step
            )

            substep_accepted = False

            while (
                trial_step
                >= minimum_step
                * (
                    1.0
                    - 1.0e-12
                )
            ):
                trial_alpha = (
                    current_alpha
                    + direction
                    * trial_step
                )

                trial_result, trial_metadata = (
                    continuation.local_solve(
                        Mach,
                        trial_alpha,
                        float(
                            current_result.cr
                        ),
                        float(
                            current_result.ci
                        ),
                        solver_args,
                    )
                )

                n_substeps += 1

                last_result = (
                    trial_result
                )

                last_metadata = dict(
                    trial_metadata
                )

                accepted = bool(
                    trial_result
                    is not None
                    and trial_metadata.get(
                        "accepted",
                        False,
                    )
                )

                if accepted:
                    current_alpha = float(
                        trial_alpha
                    )

                    current_result = (
                        trial_result
                    )

                    substep_accepted = True
                    break

                trial_step *= 0.5
                n_retries += 1

            if not substep_accepted:
                failure_metadata = dict(
                    last_metadata
                )

                failure_metadata[
                    "accepted"
                ] = False

                failure_metadata[
                    "continuation_failure_alpha"
                ] = current_alpha

                return (
                    last_result,
                    failure_metadata,
                    {
                        "continuation_success": False,
                        "continuation_n_substeps": n_substeps,
                        "continuation_n_retries": n_retries,
                        "continuation_last_alpha": current_alpha,
                    },
                )

        success_metadata = dict(
            last_metadata
        )

        success_metadata[
            "accepted"
        ] = True

        return (
            current_result,
            success_metadata,
            {
                "continuation_success": True,
                "continuation_n_substeps": n_substeps,
                "continuation_n_retries": n_retries,
                "continuation_last_alpha": current_alpha,
            },
        )

    catalog_rows: list[
        dict[str, Any]
    ] = []

    def process_case(
        case: pd.Series,
        result: Any,
        metadata: dict[str, Any],
        reference: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, np.ndarray] | None,
    ]:
        alpha = float(
            case["alpha"]
        )

        s = float(
            case["s"]
        )

        row: dict[str, Any] = {
            "Mach": Mach,
            "s": s,
            "alpha": alpha,
            "alpha_neutral": float(
                case["alpha_neutral"]
            ),
            "accepted": bool(
                metadata.get(
                    "accepted",
                    False,
                )
            ),
            "quality_level": "rejected",
            "quality_reason": "",
            "field_file": "",
        }

        if result is None:
            row["quality_reason"] = (
                "no_shooting_result"
            )

            return row, None

        row.update(
            {
                "cr": float(result.cr),
                "ci": float(result.ci),
                "stage1_mismatch": float(
                    result.stage1_mismatch
                ),
                "stage2_mismatch": float(
                    result.stage2_mismatch
                ),
                "shooting_success": bool(
                    result.success
                ),
                "spectral_success": bool(
                    result.spectral_success
                ),
                "mode_success": bool(
                    result.mode_success
                ),
                "y_limit": float(
                    result.y_limit
                ),
                "ln_p_start_right": float(
                    result.ln_p_start_right
                ),
            }
        )

        try:
            y_limit = float(
                result.y_limit
            )

            y = (
                args.mapping_scale
                * np.sinh(
                    xi
                    * np.arcsinh(
                        y_limit
                        / args.mapping_scale
                    )
                )
            )

            reconstructed = (
                call_reconstructor(
                    reconstruct,
                    Mach=Mach,
                    alpha=alpha,
                    shooting_result=result,
                    solver_args=solver_args,
                )
            )

            fields = (
                prepare_modal_fields(
                    reconstructed,
                    y,
                )
            )

            q_consistency = float(
                fields[
                    "q_consistency_rel"
                ]
            )

            fields, original_norm = (
                normalize_fields(
                    y,
                    fields,
                )
            )

            overlap, phase = (
                overlap_and_phase(
                    reference["y"],
                    reference["p"],
                    y,
                    fields["p"],
                )
            )

            fields = apply_phase(
                fields,
                phase,
            )

            finite = bool(
                finite_complex(
                    fields["p"]
                )
                and finite_complex(
                    fields["q"]
                )
                and finite_complex(
                    fields["gamma"]
                )
            )

            tail_count = min(
                args.tail_points,
                max(
                    1,
                    args.n_y // 20,
                ),
            )

            peak_p = max(
                float(
                    np.max(
                        np.abs(
                            fields["p"]
                        )
                    )
                ),
                1.0e-30,
            )

            tail_p = max(
                float(
                    np.max(
                        np.abs(
                            fields["p"][
                                :tail_count
                            ]
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            fields["p"][
                                -tail_count:
                            ]
                        )
                    )
                ),
            )

            tail_ratio = (
                tail_p / peak_p
            )

            min_abs_u_minus_c = float(
                np.min(
                    np.abs(
                        np.tanh(y)
                        - complex(
                            result.cr,
                            result.ci,
                        )
                    )
                )
            )

            (
                asymptotic_bc_rel,
                asymptotic_bc_left,
                asymptotic_bc_right,
                gamma_asymptotic_left,
                gamma_asymptotic_right,
            ) = asymptotic_bc_relative_error(
                alpha=alpha,
                Mach=Mach,
                cr=float(result.cr),
                ci=float(result.ci),
                p=fields["p"],
                q=fields["q"],
            )

            spectral_certified = bool(
                result.spectral_success
                and result.ci > 0.0
                and result.stage1_mismatch
                <= args.cert_stage1
            )

            modal_certified = bool(
                spectral_certified
                and result.success
                and result.mode_success
                and result.stage2_mismatch
                <= args.cert_stage2
                and finite
                and q_consistency
                <= args.cert_q_consistency
                and asymptotic_bc_rel
                <= args.cert_asymptotic_bc
            )

            if modal_certified:
                quality_level = (
                    "modal_certified"
                )

                if (
                    tail_ratio
                    <= args.cert_tail_ratio
                ):
                    quality_reason = (
                        "all_checks_passed"
                    )
                else:
                    quality_reason = (
                        "all_operator_checks_passed;"
                        "slow_decay_tail"
                    )

            elif spectral_certified:
                quality_level = (
                    "spectral_only"
                )

                reasons = []

                if not result.success:
                    reasons.append(
                        "shooting_success_false"
                    )

                if not result.mode_success:
                    reasons.append(
                        "mode_success_false"
                    )

                if (
                    result.stage2_mismatch
                    > args.cert_stage2
                ):
                    reasons.append(
                        "stage2_too_large"
                    )

                if not finite:
                    reasons.append(
                        "non_finite_fields"
                    )

                if (
                    q_consistency
                    > args.cert_q_consistency
                ):
                    reasons.append(
                        "q_consistency_failed"
                    )

                if (
                    asymptotic_bc_rel
                    > args.cert_asymptotic_bc
                ):
                    reasons.append(
                        "asymptotic_bc_failed"
                    )

                if (
                    tail_ratio
                    > args.cert_tail_ratio
                ):
                    reasons.append(
                        "slow_decay_tail"
                    )

                quality_reason = ";".join(
                    reasons
                )

            else:
                quality_level = "rejected"
                quality_reason = (
                    "spectral_certification_failed"
                )

            filename = (
                f"M_{mach_name}"
                f"_s_{safe_name(s, 4)}"
                ".npz"
            )

            field_path = (
                fields_dir / filename
            )

            np.savez_compressed(
                field_path,
                Mach=np.array(Mach),
                s=np.array(s),
                alpha=np.array(alpha),
                alpha_neutral=np.array(
                    float(
                        case[
                            "alpha_neutral"
                        ]
                    )
                ),
                cr=np.array(
                    float(result.cr)
                ),
                ci=np.array(
                    float(result.ci)
                ),
                xi=xi,
                y=y,
                p=fields["p"],
                q=fields["q"],
                gamma=fields["gamma"],
                q_returned=(
                    fields["q_returned"]
                ),
                q_consistency_rel=np.array(
                    q_consistency
                ),
                original_p_l2_norm=np.array(
                    original_norm
                ),
                phase_to_previous_real=np.array(
                    phase.real
                ),
                phase_to_previous_imag=np.array(
                    phase.imag
                ),
                overlap_previous=np.array(
                    overlap
                ),
                tail_ratio_p=np.array(
                    tail_ratio
                ),
                tail_truncated=np.array(
                    bool(
                        tail_ratio
                        > args.cert_tail_ratio
                    )
                ),
                asymptotic_bc_rel=np.array(
                    asymptotic_bc_rel
                ),
                asymptotic_bc_left=np.array(
                    asymptotic_bc_left
                ),
                asymptotic_bc_right=np.array(
                    asymptotic_bc_right
                ),
                gamma_asymptotic_left=np.array(
                    gamma_asymptotic_left
                ),
                gamma_asymptotic_right=np.array(
                    gamma_asymptotic_right
                ),
                min_abs_U_minus_c=np.array(
                    min_abs_u_minus_c
                ),
                stage1_mismatch=np.array(
                    float(
                        result.stage1_mismatch
                    )
                ),
                stage2_mismatch=np.array(
                    float(
                        result.stage2_mismatch
                    )
                ),
                quality_level=np.array(
                    quality_level
                ),
                quality_reason=np.array(
                    quality_reason
                ),
            )

            row.update(
                {
                    "quality_level": (
                        quality_level
                    ),
                    "quality_reason": (
                        quality_reason
                    ),
                    "field_file": str(
                        field_path
                    ),
                    "q_consistency_rel": (
                        q_consistency
                    ),
                    "tail_ratio_p": (
                        tail_ratio
                    ),
                    "tail_truncated": bool(
                        tail_ratio
                        > args.cert_tail_ratio
                    ),
                    "asymptotic_bc_rel": (
                        asymptotic_bc_rel
                    ),
                    "asymptotic_bc_left": (
                        asymptotic_bc_left
                    ),
                    "asymptotic_bc_right": (
                        asymptotic_bc_right
                    ),
                    "overlap_previous": (
                        overlap
                    ),
                    "min_abs_U_minus_c": (
                        min_abs_u_minus_c
                    ),
                    "p_l2_norm_before": (
                        original_norm
                    ),
                }
            )

            new_reference = {
                "y": y,
                "p": fields["p"],
            }

            return row, new_reference

        except Exception as error:
            row["quality_level"] = (
                "spectral_only"
                if bool(
                    getattr(
                        result,
                        "spectral_success",
                        False,
                    )
                )
                else "rejected"
            )

            row["quality_reason"] = (
                "modal_reconstruction_exception:"
                + repr(error)
            )

            row["traceback"] = (
                traceback.format_exc()
            )

            return row, None

    target_rows = [
        row
        for _, row in cases.iterrows()
    ]

    equal_base = [
        row
        for row in target_rows
        if math.isclose(
            float(row["alpha"]),
            base_alpha,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        )
    ]

    lower = sorted(
        [
            row
            for row in target_rows
            if float(row["alpha"])
            < base_alpha - 1.0e-10
        ],
        key=lambda row: float(
            row["alpha"]
        ),
        reverse=True,
    )

    upper = sorted(
        [
            row
            for row in target_rows
            if float(row["alpha"])
            > base_alpha + 1.0e-10
        ],
        key=lambda row: float(
            row["alpha"]
        ),
    )

    for row in equal_base:
        catalog_row, _ = process_case(
            row,
            base_result,
            {
                "accepted": True,
                "seed_distance": 0.0,
                "box_scale": 0.0,
            },
            base_reference,
        )

        catalog_rows.append(
            catalog_row
        )

        save_catalog(
            catalog_rows,
            catalog_path,
        )

    for direction_rows in [
        lower,
        upper,
    ]:
        previous_result = (
            base_result
        )

        previous_alpha = float(
            base_alpha
        )

        previous_reference = (
            base_reference
        )

        for row in direction_rows:
            alpha = float(
                row["alpha"]
            )

            (
                result,
                metadata,
                continuation_info,
            ) = continue_to_target(
                start_alpha=previous_alpha,
                start_result=previous_result,
                target_alpha=alpha,
            )

            catalog_row, reference = (
                process_case(
                    row,
                    result,
                    metadata,
                    previous_reference,
                )
            )

            catalog_row.update(
                continuation_info
            )

            catalog_rows.append(
                catalog_row
            )

            save_catalog(
                catalog_rows,
                catalog_path,
            )

            print(
                f"M={Mach:.4f} "
                f"s={float(row['s']):.4f} "
                f"alpha={alpha:.8f} "
                f"quality="
                f"{catalog_row['quality_level']} "
                f"reason="
                f"{catalog_row['quality_reason']}",
                flush=True,
            )

            if (
                result is not None
                and bool(
                    metadata.get(
                        "accepted",
                        False,
                    )
                )
            ):
                previous_result = result
                previous_alpha = alpha

            if reference is not None:
                previous_reference = (
                    reference
                )

    catalog = pd.DataFrame(
        catalog_rows
    ).sort_values("s")

    catalog.to_csv(
        catalog_path,
        index=False,
    )

    counts = {
        str(key): int(value)
        for key, value
        in catalog[
            "quality_level"
        ].value_counts().items()
    }

    report = {
        "Mach": Mach,
        "n_requested": int(
            len(cases)
        ),
        "n_completed": int(
            len(catalog)
        ),
        "quality_counts": counts,
    }

    (
        mach_dir
        / "report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
