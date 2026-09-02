#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PACKAGE = Path(__file__).resolve().parents[2]
REPO = PACKAGE.parent
SRC = PACKAGE / "src"

sys.path.insert(0, str(SRC))

from classic_supersonic_reference.solver.mstab17_supersonic_solver import (  # noqa: E402
    Mstab17SupersonicSolver,
)
from classic_supersonic_reference.validation.modal_reconstruction import (  # noqa: E402
    FIELD_NAMES,
    exact_right_log_amplitude,
    reconstruct_from_solver,
)


DEFAULT_CONFIG = (
    PACKAGE
    / "configs"
    / "reproducibility"
    / "witness_M150_a01625.yaml"
)

DEFAULT_REFERENCE = (
    PACKAGE
    / "data"
    / "samples"
    / "witness_M150_a01625_raw.parquet"
)

DEFAULT_JSON = (
    PACKAGE
    / "reproducibility"
    / "results"
    / "witness_M150_a01625_modal_calibration.json"
)

DEFAULT_CSV = (
    PACKAGE
    / "reproducibility"
    / "results"
    / "witness_M150_a01625_modal_comparison.csv"
)


def finite(value: Any) -> float:
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"Non-finite value: {value!r}"
        )

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON,
    )

    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV,
    )

    return parser


def instantiate_solver(
    config: dict,
) -> Mstab17SupersonicSolver:
    case = config["case"]
    solver_config = config["solver"]

    max_step = solver_config.get(
        "max_step"
    )

    if max_step is None:
        max_step = float("inf")
    else:
        max_step = finite(max_step)

    return Mstab17SupersonicSolver(
        alpha=finite(case["alpha"]),
        Mach=finite(case["Mach"]),
        match_y=finite(
            solver_config["match_y"]
        ),
        ln_p_start_left=finite(
            solver_config["ln_p_start_left"]
        ),
        rtol=finite(
            solver_config["rtol"]
        ),
        atol=finite(
            solver_config["atol"]
        ),
        max_step=max_step,
        min_y_limit=finite(
            solver_config["min_y_limit"]
        ),
        max_y_limit=finite(
            solver_config["max_y_limit"]
        ),
        y_limit_factor=finite(
            solver_config["y_limit_factor"]
        ),
        use_mapping=bool(
            solver_config["use_mapping"]
        ),
        mapping_scale=finite(
            solver_config["mapping_scale"]
        ),
    )


def solve_witness(
    solver: Mstab17SupersonicSolver,
    config: dict,
):
    expected = config["expected_spectral"]
    search = config["search"]

    cr_reference = finite(expected["cr"])
    ci_reference = finite(expected["ci"])

    cr_half_width = finite(
        search["cr_half_width"]
    )

    ci_half_width = finite(
        search["ci_half_width"]
    )

    return solver.solve(
        cr_min=max(
            0.0,
            cr_reference - cr_half_width,
        ),
        cr_max=(
            cr_reference + cr_half_width
        ),
        ci_min=max(
            1.0e-8,
            ci_reference - ci_half_width,
        ),
        ci_max=(
            ci_reference + ci_half_width
        ),
        max_iter=int(search["max_iter"]),
        tol=finite(search["tol"]),
        grid_size=int(search["grid_size"]),
        constrain_to_initial_box=bool(
            search["constrain_to_initial_box"]
        ),
    )


def complex_reference(
    frame: pd.DataFrame,
    field: str,
) -> np.ndarray:
    real_column = f"{field}_real"
    imag_column = f"{field}_imag"

    missing = {
        real_column,
        imag_column,
    }.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing reference columns: "
            f"{sorted(missing)}"
        )

    return (
        frame[real_column].to_numpy(
            dtype=float
        )
        + 1j
        * frame[imag_column].to_numpy(
            dtype=float
        )
    )


def interpolate_complex(
    source_y: np.ndarray,
    source_values: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    order = np.argsort(source_y)

    source_y = np.asarray(
        source_y,
        dtype=float,
    )[order]

    source_values = np.asarray(
        source_values,
        dtype=complex,
    )[order]

    unique_y, unique_indices = np.unique(
        source_y,
        return_index=True,
    )

    source_values = source_values[
        unique_indices
    ]

    real = np.interp(
        target_y,
        unique_y,
        source_values.real,
    )

    imag = np.interp(
        target_y,
        unique_y,
        source_values.imag,
    )

    return real + 1j * imag


def relative_l2(
    predicted: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    if not np.any(mask):
        return None

    denominator = np.linalg.norm(
        reference[mask]
    )

    if denominator <= 1.0e-15:
        return None

    return float(
        np.linalg.norm(
            predicted[mask]
            - reference[mask]
        )
        / denominator
    )


def complex_correlation(
    predicted: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    if not np.any(mask):
        return None

    predicted_norm = np.linalg.norm(
        predicted[mask]
    )

    reference_norm = np.linalg.norm(
        reference[mask]
    )

    denominator = (
        predicted_norm * reference_norm
    )

    if denominator <= 1.0e-15:
        return None

    return float(
        abs(
            np.vdot(
                predicted[mask],
                reference[mask],
            )
        )
        / denominator
    )


def global_alignment_factor(
    predicted: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> complex:
    predicted_parts = []
    reference_parts = []

    for field in FIELD_NAMES:
        reference_values = reference[field]
        predicted_values = predicted[field]

        peak = float(
            np.max(np.abs(reference_values))
        )

        mask = (
            np.abs(reference_values)
            >= 0.05 * peak
        )

        if np.any(mask):
            predicted_parts.append(
                predicted_values[mask]
            )
            reference_parts.append(
                reference_values[mask]
            )

    predicted_stack = np.concatenate(
        predicted_parts
    )

    reference_stack = np.concatenate(
        reference_parts
    )

    denominator = np.vdot(
        predicted_stack,
        predicted_stack,
    )

    if abs(denominator) <= 1.0e-30:
        raise ValueError(
            "Degenerate predicted modal fields."
        )

    return complex(
        np.vdot(
            predicted_stack,
            reference_stack,
        )
        / denominator
    )


def field_metrics(
    aligned: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    amplitude = np.abs(reference)
    peak = float(np.max(amplitude))

    full = np.ones(
        len(reference),
        dtype=bool,
    )

    core = amplitude >= 0.05 * peak

    tail = (
        (amplitude >= 0.001 * peak)
        & (amplitude <= 0.02 * peak)
    )

    return {
        "reference_peak": peak,
        "n_full": int(full.sum()),
        "n_core_amp_ge_5pct": int(
            core.sum()
        ),
        "n_tail_amp_0p1pct_to_2pct": int(
            tail.sum()
        ),
        "relative_l2_full_overlap": (
            relative_l2(
                aligned,
                reference,
                full,
            )
        ),
        "relative_l2_core_amp_ge_5pct": (
            relative_l2(
                aligned,
                reference,
                core,
            )
        ),
        "relative_l2_tail_amp_0p1pct_to_2pct": (
            relative_l2(
                aligned,
                reference,
                tail,
            )
        ),
        "complex_correlation_full": (
            complex_correlation(
                aligned,
                reference,
                full,
            )
        ),
        "complex_correlation_core": (
            complex_correlation(
                aligned,
                reference,
                core,
            )
        ),
        "max_abs_error_over_peak": float(
            np.max(
                np.abs(
                    aligned - reference
                )
            )
            / max(peak, 1.0e-15)
        ),
    }


def compare_variant(
    *,
    name: str,
    reconstructed: dict[str, Any],
    reference_frame: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    reconstructed_y = np.asarray(
        reconstructed["y"],
        dtype=float,
    )

    reference_y_all = (
        reference_frame["y"]
        .to_numpy(dtype=float)
    )

    y_min = float(
        np.min(reconstructed_y)
    )

    y_max = float(
        np.max(reconstructed_y)
    )

    overlap_mask = (
        (reference_y_all >= y_min)
        & (reference_y_all <= y_max)
    )

    reference_frame = (
        reference_frame.loc[overlap_mask]
        .sort_values("y")
        .reset_index(drop=True)
    )

    reference_y = (
        reference_frame["y"]
        .to_numpy(dtype=float)
    )

    if len(reference_y) < 100:
        raise ValueError(
            f"Too few overlap points: "
            f"{len(reference_y)}"
        )

    reference_fields = {
        field: complex_reference(
            reference_frame,
            field,
        )
        for field in FIELD_NAMES
    }

    predicted_fields = {
        field: interpolate_complex(
            reconstructed_y,
            reconstructed[field],
            reference_y,
        )
        for field in FIELD_NAMES
    }

    factor = global_alignment_factor(
        predicted_fields,
        reference_fields,
    )

    aligned_fields = {
        field: factor
        * predicted_fields[field]
        for field in FIELD_NAMES
    }

    metrics = {
        "variant": name,
        "reconstructed_y_min": y_min,
        "reconstructed_y_max": y_max,
        "reference_y_min": float(
            reference_y_all.min()
        ),
        "reference_y_max": float(
            reference_y_all.max()
        ),
        "overlap_y_min": float(
            reference_y.min()
        ),
        "overlap_y_max": float(
            reference_y.max()
        ),
        "n_reference_total": int(
            len(reference_y_all)
        ),
        "n_overlap": int(
            len(reference_y)
        ),
        "alignment": {
            "real": float(factor.real),
            "imag": float(factor.imag),
            "absolute": float(abs(factor)),
            "phase_radians": float(
                np.angle(factor)
            ),
        },
        "fields": {
            field: field_metrics(
                aligned_fields[field],
                reference_fields[field],
            )
            for field in FIELD_NAMES
        },
    }

    pointwise = pd.DataFrame(
        {
            "variant": name,
            "y": reference_y,
        }
    )

    for field in FIELD_NAMES:
        reference_values = (
            reference_fields[field]
        )

        predicted_values = (
            predicted_fields[field]
        )

        aligned_values = (
            aligned_fields[field]
        )

        pointwise[
            f"{field}_reference_real"
        ] = reference_values.real

        pointwise[
            f"{field}_reference_imag"
        ] = reference_values.imag

        pointwise[
            f"{field}_predicted_real"
        ] = predicted_values.real

        pointwise[
            f"{field}_predicted_imag"
        ] = predicted_values.imag

        pointwise[
            f"{field}_aligned_real"
        ] = aligned_values.real

        pointwise[
            f"{field}_aligned_imag"
        ] = aligned_values.imag

        pointwise[
            f"{field}_abs_error"
        ] = np.abs(
            aligned_values
            - reference_values
        )

    return metrics, pointwise


def main() -> None:
    args = build_parser().parse_args()

    config = yaml.safe_load(
        args.config.read_text()
    )

    reference_frame = pd.read_parquet(
        args.reference
    )

    reference_frame = (
        reference_frame
        .sort_values("y")
        .reset_index(drop=True)
    )

    solver = instantiate_solver(config)

    started = time.perf_counter()

    result = solve_witness(
        solver,
        config,
    )

    solved_fields = reconstruct_from_solver(
        solver,
        cr=float(result.cr),
        ci=float(result.ci),
        ln_p_start_right=float(
            result.ln_p_start_right
        ),
    )

    expected = config["expected_spectral"]

    frozen_cr = finite(expected["cr"])
    frozen_ci = finite(expected["ci"])

    frozen_ln_p = exact_right_log_amplitude(
        solver,
        cr=frozen_cr,
        ci=frozen_ci,
    )

    frozen_fields = reconstruct_from_solver(
        solver,
        cr=frozen_cr,
        ci=frozen_ci,
        ln_p_start_right=frozen_ln_p,
    )

    solved_metrics, solved_pointwise = (
        compare_variant(
            name="solved_spectral_root",
            reconstructed=solved_fields,
            reference_frame=reference_frame,
        )
    )

    frozen_metrics, frozen_pointwise = (
        compare_variant(
            name="frozen_spectral_values",
            reconstructed=frozen_fields,
            reference_frame=reference_frame,
        )
    )

    pointwise = pd.concat(
        [
            solved_pointwise,
            frozen_pointwise,
        ],
        ignore_index=True,
    )

    document = {
        "case": config["case"],
        "status": "calibration_only",
        "reference_modal_variant": (
            "raw_confirmed"
        ),
        "reference_path": str(
            args.reference.relative_to(REPO)
        ),
        "solve_result": {
            "cr": float(result.cr),
            "ci": float(result.ci),
            "omega_i": float(result.omega_i),
            "stage1_mismatch": float(
                result.stage1_mismatch
            ),
            "stage2_mismatch": float(
                result.stage2_mismatch
            ),
            "y_limit": float(
                result.y_limit
            ),
            "ln_p_start_right": float(
                result.ln_p_start_right
            ),
            "success": bool(result.success),
        },
        "frozen_spectral_reconstruction": {
            "cr": frozen_cr,
            "ci": frozen_ci,
            "omega_i": float(
                config[
                    "expected_spectral"
                ]["omega_i"]
            ),
            "ln_p_start_right": (
                frozen_ln_p
            ),
        },
        "comparison": {
            "alignment": (
                "single global complex "
                "least-squares factor"
            ),
            "core_definition": (
                "field amplitude >= 5% "
                "of its reference peak"
            ),
            "tail_definition": (
                "0.1% <= field amplitude "
                "<= 2% of its reference peak"
            ),
            "outside_overlap": (
                "not evaluated"
            ),
            "variants": [
                solved_metrics,
                frozen_metrics,
            ],
        },
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
    }

    args.json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.json_output.write_text(
        json.dumps(
            document,
            indent=2,
        )
        + "\n"
    )

    pointwise.to_csv(
        args.csv_output,
        index=False,
    )

    print(
        json.dumps(
            document,
            indent=2,
        )
    )

    print()
    print("Wrote:", args.json_output)
    print("Wrote:", args.csv_output)


if __name__ == "__main__":
    main()
