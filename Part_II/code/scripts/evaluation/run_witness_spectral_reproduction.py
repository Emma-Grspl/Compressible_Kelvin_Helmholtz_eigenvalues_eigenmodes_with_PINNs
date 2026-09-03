#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import yaml


PACKAGE = Path(__file__).resolve().parents[2]
SRC = PACKAGE / "src"

sys.path.insert(0, str(SRC))

from classic_supersonic_reference.solver.mstab17_supersonic_solver import (  # noqa: E402
    Mstab17SupersonicSolver,
)


DEFAULT_CONFIG = (
    PACKAGE
    / "configs"
    / "reproducibility"
    / "witness_M150_a01625.yaml"
)

DEFAULT_OUTPUT = (
    PACKAGE
    / "reproducibility"
    / "results"
    / "witness_M150_a01625_spectral_reproduction.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def finite(value: object) -> float:
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"Non-finite value: {value!r}")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = yaml.safe_load(
        args.config.read_text()
    )

    case = config["case"]
    expected = config["expected_spectral"]
    solver_config = config["solver"]
    search = config["search"]
    acceptance = config["acceptance"]

    Mach = finite(case["Mach"])
    alpha = finite(case["alpha"])

    cr_reference = finite(expected["cr"])
    ci_reference = finite(expected["ci"])
    omega_i_reference = finite(expected["omega_i"])

    max_step = solver_config.get("max_step")

    if max_step is None:
        max_step = float("inf")
    else:
        max_step = finite(max_step)

    solver = Mstab17SupersonicSolver(
        alpha=alpha,
        Mach=Mach,
        match_y=finite(solver_config["match_y"]),
        ln_p_start_left=finite(
            solver_config["ln_p_start_left"]
        ),
        rtol=finite(solver_config["rtol"]),
        atol=finite(solver_config["atol"]),
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

    cr_half_width = finite(
        search["cr_half_width"]
    )
    ci_half_width = finite(
        search["ci_half_width"]
    )

    started = time.perf_counter()

    mismatch_reference = finite(
        solver.stage1_mismatch(
            cr_reference,
            ci_reference,
        )
    )

    cr_solved, ci_solved, mismatch_solved = (
        solver.solve_eigenvalue(
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
    )

    cr_solved = finite(cr_solved)
    ci_solved = finite(ci_solved)
    mismatch_solved = finite(mismatch_solved)
    omega_i_solved = alpha * ci_solved

    metrics = {
        "abs_cr_error": abs(
            cr_solved - cr_reference
        ),
        "abs_ci_error": abs(
            ci_solved - ci_reference
        ),
        "abs_omega_i_error": abs(
            omega_i_solved - omega_i_reference
        ),
        "stage1_mismatch_at_reference": abs(
            mismatch_reference
        ),
        "stage1_mismatch_solved": abs(
            mismatch_solved
        ),
    }

    checks = {
        "abs_cr_error": (
            metrics["abs_cr_error"]
            <= finite(
                acceptance["abs_cr_error_max"]
            )
        ),
        "abs_ci_error": (
            metrics["abs_ci_error"]
            <= finite(
                acceptance["abs_ci_error_max"]
            )
        ),
        "abs_omega_i_error": (
            metrics["abs_omega_i_error"]
            <= finite(
                acceptance[
                    "abs_omega_i_error_max"
                ]
            )
        ),
        "stage1_mismatch": (
            metrics["stage1_mismatch_solved"]
            <= finite(
                acceptance[
                    "stage1_mismatch_max"
                ]
            )
        ),
    }

    passed = all(checks.values())

    document = {
        "case": case,
        "config_path": str(
            args.config.relative_to(
                PACKAGE.parent
            )
        ),
        "config_sha256": sha256(args.config),
        "reference": {
            "cr": cr_reference,
            "ci": ci_reference,
            "omega_i": omega_i_reference,
        },
        "solved": {
            "cr": cr_solved,
            "ci": ci_solved,
            "omega_i": omega_i_solved,
        },
        "metrics": metrics,
        "acceptance": acceptance,
        "checks": checks,
        "passed": passed,
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(document, indent=2) + "\n"
    )

    print(json.dumps(document, indent=2))
    print()
    print("Wrote:", args.output)

    if not passed:
        raise SystemExit(
            "Witness spectral reproduction failed."
        )


if __name__ == "__main__":
    main()
