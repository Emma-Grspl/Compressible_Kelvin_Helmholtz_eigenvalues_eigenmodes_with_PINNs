#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd


PACKAGE = Path(__file__).resolve().parents[2]
REPO = PACKAGE.parent
SRC = PACKAGE / "src"

sys.path.insert(0, str(SRC))

from classic_supersonic_reference.solver.mstab17_supersonic_solver import (  # noqa: E402
    Mstab17SupersonicSolver,
)


DEFAULT_EXPECTED = (
    PACKAGE
    / "data"
    / "samples"
    / "witness_M150_a01625_expected.json"
)

DEFAULT_OUTPUT = (
    PACKAGE
    / "reproducibility"
    / "results"
    / "witness_M150_a01625_spectral_calibration.csv"
)

DEFAULT_JSON = (
    PACKAGE
    / "reproducibility"
    / "results"
    / "witness_M150_a01625_spectral_calibration.json"
)


CONFIGURATIONS = [
    {
        "name": "unmapped_y80",
        "use_mapping": False,
        "mapping_scale": 5.0,
        "min_y_limit": 10.0,
        "max_y_limit": 80.0,
        "y_limit_factor": 4.0,
    },
    {
        "name": "unmapped_y500",
        "use_mapping": False,
        "mapping_scale": 5.0,
        "min_y_limit": 10.0,
        "max_y_limit": 500.0,
        "y_limit_factor": 4.0,
    },
    {
        "name": "mapped_scale5_y500",
        "use_mapping": True,
        "mapping_scale": 5.0,
        "min_y_limit": 10.0,
        "max_y_limit": 500.0,
        "y_limit_factor": 4.0,
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_EXPECTED,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON,
    )
    parser.add_argument(
        "--cr-half-width",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--ci-half-width",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--solver-tol",
        type=float,
        default=1e-9,
    )

    return parser


def finite_float(value: object) -> float:
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"Non-finite value: {value!r}")

    return result


def run_configuration(
    *,
    configuration: dict,
    Mach: float,
    alpha: float,
    cr_reference: float,
    ci_reference: float,
    cr_half_width: float,
    ci_half_width: float,
    grid_size: int,
    max_iter: int,
    solver_tol: float,
) -> dict:
    solver = Mstab17SupersonicSolver(
        alpha=alpha,
        Mach=Mach,
        match_y=1.0,
        ln_p_start_left=-5.0,
        rtol=1e-10,
        atol=1e-12,
        max_step=float("inf"),
        min_y_limit=configuration["min_y_limit"],
        max_y_limit=configuration["max_y_limit"],
        y_limit_factor=configuration["y_limit_factor"],
        use_mapping=configuration["use_mapping"],
        mapping_scale=configuration["mapping_scale"],
    )

    row = {
        "configuration": configuration["name"],
        "Mach": Mach,
        "alpha": alpha,
        "use_mapping": configuration["use_mapping"],
        "mapping_scale": configuration["mapping_scale"],
        "min_y_limit": configuration["min_y_limit"],
        "max_y_limit": configuration["max_y_limit"],
        "y_limit_factor": configuration["y_limit_factor"],
        "cr_reference": cr_reference,
        "ci_reference": ci_reference,
        "status": "pending",
        "exception": "",
    }

    started = time.perf_counter()

    try:
        mismatch_reference = finite_float(
            solver.stage1_mismatch(
                cr_reference,
                ci_reference,
            )
        )

        cr_min = max(
            0.0,
            cr_reference - cr_half_width,
        )
        cr_max = cr_reference + cr_half_width
        ci_min = max(
            1e-8,
            ci_reference - ci_half_width,
        )
        ci_max = ci_reference + ci_half_width

        cr_solved, ci_solved, mismatch_solved = (
            solver.solve_eigenvalue(
                cr_min=cr_min,
                cr_max=cr_max,
                ci_min=ci_min,
                ci_max=ci_max,
                max_iter=max_iter,
                tol=solver_tol,
                grid_size=grid_size,
                constrain_to_initial_box=True,
            )
        )

        cr_solved = finite_float(cr_solved)
        ci_solved = finite_float(ci_solved)
        mismatch_solved = finite_float(mismatch_solved)

        row.update(
            {
                "status": "completed",
                "stage1_mismatch_at_reference": mismatch_reference,
                "cr_solved": cr_solved,
                "ci_solved": ci_solved,
                "omega_i_solved": alpha * ci_solved,
                "stage1_mismatch_solved": mismatch_solved,
                "abs_cr_error": abs(cr_solved - cr_reference),
                "abs_ci_error": abs(ci_solved - ci_reference),
                "abs_omega_i_error": abs(
                    alpha * ci_solved
                    - alpha * ci_reference
                ),
            }
        )

    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "exception": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        )

    row["elapsed_seconds"] = (
        time.perf_counter() - started
    )

    return row


def main() -> None:
    args = build_parser().parse_args()

    expected = json.loads(
        args.expected.read_text()
    )

    Mach = finite_float(
        expected["point"]["Mach"]
    )
    alpha = finite_float(
        expected["point"]["alpha"]
    )
    cr_reference = finite_float(
        expected["spectral"]["cr"]
    )
    ci_reference = finite_float(
        expected["spectral"]["ci"]
    )

    rows = []

    for configuration in CONFIGURATIONS:
        print()
        print(
            "Running:",
            configuration["name"],
        )

        row = run_configuration(
            configuration=configuration,
            Mach=Mach,
            alpha=alpha,
            cr_reference=cr_reference,
            ci_reference=ci_reference,
            cr_half_width=args.cr_half_width,
            ci_half_width=args.ci_half_width,
            grid_size=args.grid_size,
            max_iter=args.max_iter,
            solver_tol=args.solver_tol,
        )

        rows.append(row)

        print(
            json.dumps(
                row,
                indent=2,
                default=str,
            )
        )

    frame = pd.DataFrame(rows)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        args.output,
        index=False,
    )

    document = {
        "case": expected["name"],
        "point": expected["point"],
        "reference_spectral": expected["spectral"],
        "search": {
            "cr_half_width": args.cr_half_width,
            "ci_half_width": args.ci_half_width,
            "grid_size": args.grid_size,
            "max_iter": args.max_iter,
            "solver_tol": args.solver_tol,
            "constrain_to_initial_box": True,
        },
        "runs": rows,
    }

    args.json_output.write_text(
        json.dumps(
            document,
            indent=2,
            default=str,
        )
        + "\n"
    )

    print()
    print("Wrote:", args.output)
    print("Wrote:", args.json_output)

    completed = frame[
        frame["status"] == "completed"
    ].copy()

    if completed.empty:
        raise SystemExit(
            "No calibration configuration completed."
        )

    completed = completed.sort_values(
        [
            "abs_cr_error",
            "abs_ci_error",
            "stage1_mismatch_solved",
        ]
    )

    columns = [
        "configuration",
        "cr_solved",
        "ci_solved",
        "abs_cr_error",
        "abs_ci_error",
        "stage1_mismatch_solved",
        "elapsed_seconds",
    ]

    print()
    print("Completed configurations:")
    print(
        completed[columns].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
