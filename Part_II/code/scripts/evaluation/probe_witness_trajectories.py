#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent
SRC = PACKAGE / "src"

sys.path.insert(0, str(SRC))

from classic_supersonic_reference.solver.mstab17_supersonic_solver import (  # noqa: E402
    Mstab17SupersonicSolver,
)


CONFIG = (
    PACKAGE
    / "configs"
    / "reproducibility"
    / "witness_M150_a01625.yaml"
)

OUTPUT = (
    PACKAGE
    / "reproducibility"
    / "results"
    / "witness_M150_a01625_trajectory_schema.json"
)


def finite(value: Any) -> float:
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"Non-finite value: {value!r}"
        )

    return result


def array_summary(value: Any) -> dict:
    array = np.asarray(value)

    summary = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }

    if array.size == 0:
        return summary

    if np.iscomplexobj(array):
        summary.update(
            {
                "all_finite": bool(
                    (
                        np.isfinite(array.real)
                        & np.isfinite(array.imag)
                    ).all()
                ),
                "real_min": float(
                    np.nanmin(array.real)
                ),
                "real_max": float(
                    np.nanmax(array.real)
                ),
                "imag_min": float(
                    np.nanmin(array.imag)
                ),
                "imag_max": float(
                    np.nanmax(array.imag)
                ),
            }
        )
    elif np.issubdtype(
        array.dtype,
        np.number,
    ):
        numeric = array.astype(
            float,
            copy=False,
        )

        summary.update(
            {
                "all_finite": bool(
                    np.isfinite(numeric).all()
                ),
                "min": float(
                    np.nanmin(numeric)
                ),
                "max": float(
                    np.nanmax(numeric)
                ),
            }
        )

    return summary


def trajectory_summary(
    value: Any,
    index: int,
) -> dict:
    summary = {
        "index": index,
        "class": type(value).__name__,
        "module": type(value).__module__,
    }

    if hasattr(value, "t"):
        summary["t"] = array_summary(value.t)

    if hasattr(value, "y"):
        summary["y"] = array_summary(value.y)

    for name in [
        "success",
        "status",
        "message",
        "nfev",
        "njev",
        "nlu",
    ]:
        if hasattr(value, name):
            raw = getattr(value, name)

            if isinstance(
                raw,
                (np.integer, np.floating),
            ):
                raw = raw.item()

            summary[name] = raw

    return summary


def main() -> None:
    config = yaml.safe_load(
        CONFIG.read_text()
    )

    case = config["case"]
    solver_config = config["solver"]
    search = config["search"]

    max_step = solver_config.get(
        "max_step"
    )

    if max_step is None:
        max_step = float("inf")
    else:
        max_step = finite(max_step)

    solver = Mstab17SupersonicSolver(
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

    expected = config["expected_spectral"]

    cr_reference = finite(
        expected["cr"]
    )
    ci_reference = finite(
        expected["ci"]
    )

    cr_half_width = finite(
        search["cr_half_width"]
    )
    ci_half_width = finite(
        search["ci_half_width"]
    )

    started = time.perf_counter()

    result = solver.solve(
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

    trajectories = solver.get_trajectories(
        result.cr,
        result.ci,
        result.ln_p_start_right,
    )

    items = []

    for index, value in enumerate(
        trajectories
    ):
        if hasattr(value, "t") or hasattr(
            value,
            "y",
        ):
            items.append(
                trajectory_summary(
                    value,
                    index,
                )
            )
        else:
            scalar = value

            if isinstance(
                scalar,
                (np.integer, np.floating),
            ):
                scalar = scalar.item()

            items.append(
                {
                    "index": index,
                    "class": type(value).__name__,
                    "value": scalar,
                }
            )

    document = {
        "case": case,
        "solve_result": {
            "cr": result.cr,
            "ci": result.ci,
            "omega_i": result.omega_i,
            "stage1_mismatch": (
                result.stage1_mismatch
            ),
            "stage2_mismatch": (
                result.stage2_mismatch
            ),
            "y_limit": result.y_limit,
            "ln_p_start_right": (
                result.ln_p_start_right
            ),
            "spectral_success": (
                result.spectral_success
            ),
            "mode_success": (
                result.mode_success
            ),
            "success": result.success,
        },
        "get_trajectories": {
            "return_type": (
                type(trajectories).__name__
            ),
            "length": len(trajectories),
            "items": items,
        },
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(document, indent=2)
        + "\n"
    )

    print(
        json.dumps(
            document,
            indent=2,
        )
    )

    print()
    print("Wrote:", OUTPUT.relative_to(REPO))


if __name__ == "__main__":
    main()
