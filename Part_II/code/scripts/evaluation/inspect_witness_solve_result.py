#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
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

JSON_OUTPUT = (
    PACKAGE
    / "reproducibility"
    / "results"
    / "witness_M150_a01625_solve_result_schema.json"
)

MD_OUTPUT = (
    PACKAGE
    / "provenance"
    / "witness_solver_result_schema.md"
)

MODAL_KEYWORDS = (
    "y",
    "p",
    "rho",
    "u",
    "v",
    "mode",
    "field",
    "trajectory",
    "solution",
    "left",
    "right",
    "cr",
    "ci",
    "omega",
)


def finite_float(value: Any) -> float:
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"Non-finite value: {value!r}")

    return result


def array_summary(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array)

    summary: dict[str, Any] = {
        "kind": "ndarray",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }

    if array.size == 0:
        return summary

    flat = array.reshape(-1)

    if np.iscomplexobj(array):
        finite_mask = (
            np.isfinite(array.real)
            & np.isfinite(array.imag)
        )

        summary.update(
            {
                "is_complex": True,
                "all_finite": bool(finite_mask.all()),
                "real_min": float(np.nanmin(array.real)),
                "real_max": float(np.nanmax(array.real)),
                "imag_min": float(np.nanmin(array.imag)),
                "imag_max": float(np.nanmax(array.imag)),
                "abs_min": float(np.nanmin(np.abs(array))),
                "abs_max": float(np.nanmax(np.abs(array))),
                "head": [
                    {
                        "real": float(complex(value).real),
                        "imag": float(complex(value).imag),
                    }
                    for value in flat[:3]
                ],
                "tail": [
                    {
                        "real": float(complex(value).real),
                        "imag": float(complex(value).imag),
                    }
                    for value in flat[-3:]
                ],
            }
        )
    elif np.issubdtype(array.dtype, np.number):
        numeric = array.astype(float, copy=False)

        summary.update(
            {
                "is_complex": False,
                "all_finite": bool(np.isfinite(numeric).all()),
                "min": float(np.nanmin(numeric)),
                "max": float(np.nanmax(numeric)),
                "head": [
                    float(value)
                    for value in flat[:3]
                ],
                "tail": [
                    float(value)
                    for value in flat[-3:]
                ],
            }
        )
    else:
        summary.update(
            {
                "head": [
                    repr(value)
                    for value in flat[:3]
                ],
                "tail": [
                    repr(value)
                    for value in flat[-3:]
                ],
            }
        )

    return summary


def solve_ivp_summary(value: Any) -> dict[str, Any] | None:
    if not hasattr(value, "t") or not hasattr(value, "y"):
        return None

    try:
        t = np.asarray(value.t)
        y = np.asarray(value.y)
    except Exception:
        return None

    return {
        "kind": type(value).__name__,
        "module": type(value).__module__,
        "t": array_summary(t),
        "y": array_summary(y),
        "success": getattr(value, "success", None),
        "status": getattr(value, "status", None),
        "message": str(getattr(value, "message", "")),
        "nfev": getattr(value, "nfev", None),
        "njev": getattr(value, "njev", None),
        "nlu": getattr(value, "nlu", None),
    }


def summarize(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return {
            "kind": type(value).__name__,
            "repr": repr(value)[:300],
        }

    if value is None or isinstance(
        value,
        (bool, int, float, str),
    ):
        return value

    if isinstance(value, complex):
        return {
            "kind": "complex",
            "real": float(value.real),
            "imag": float(value.imag),
        }

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return array_summary(value)

    ivp = solve_ivp_summary(value)

    if ivp is not None:
        return ivp

    if dataclasses.is_dataclass(value):
        return {
            "kind": type(value).__name__,
            "module": type(value).__module__,
            "fields": {
                field.name: summarize(
                    getattr(value, field.name),
                    depth + 1,
                )
                for field in dataclasses.fields(value)
            },
        }

    if isinstance(value, dict):
        return {
            str(key): summarize(item, depth + 1)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return {
            "kind": type(value).__name__,
            "length": len(value),
            "items": [
                summarize(item, depth + 1)
                for item in value[:20]
            ],
        }

    try:
        converted = np.asarray(value)

        if converted.ndim > 0 and converted.dtype != object:
            return array_summary(converted)
    except Exception:
        pass

    return {
        "kind": type(value).__name__,
        "module": type(value).__module__,
        "repr": repr(value)[:500],
    }


def public_attributes(value: Any) -> list[str]:
    return sorted(
        name
        for name in dir(value)
        if not name.startswith("_")
    )


def candidate_attribute(name: str) -> bool:
    lowered = name.lower()

    return any(
        keyword == lowered
        or keyword in lowered
        for keyword in MODAL_KEYWORDS
    )


def main() -> None:
    if not CONFIG.exists():
        raise FileNotFoundError(CONFIG)

    config = yaml.safe_load(CONFIG.read_text())

    case = config["case"]
    solver_config = config["solver"]
    search = config["search"]

    max_step = solver_config.get("max_step")

    if max_step is None:
        max_step = float("inf")
    else:
        max_step = finite_float(max_step)

    solver = Mstab17SupersonicSolver(
        alpha=finite_float(case["alpha"]),
        Mach=finite_float(case["Mach"]),
        match_y=finite_float(
            solver_config["match_y"]
        ),
        ln_p_start_left=finite_float(
            solver_config["ln_p_start_left"]
        ),
        rtol=finite_float(
            solver_config["rtol"]
        ),
        atol=finite_float(
            solver_config["atol"]
        ),
        max_step=max_step,
        min_y_limit=finite_float(
            solver_config["min_y_limit"]
        ),
        max_y_limit=finite_float(
            solver_config["max_y_limit"]
        ),
        y_limit_factor=finite_float(
            solver_config["y_limit_factor"]
        ),
        use_mapping=bool(
            solver_config["use_mapping"]
        ),
        mapping_scale=finite_float(
            solver_config["mapping_scale"]
        ),
    )

    expected = config["expected_spectral"]

    cr_reference = finite_float(expected["cr"])
    ci_reference = finite_float(expected["ci"])

    cr_half_width = finite_float(
        search["cr_half_width"]
    )
    ci_half_width = finite_float(
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
        tol=finite_float(search["tol"]),
        grid_size=int(search["grid_size"]),
        constrain_to_initial_box=bool(
            search["constrain_to_initial_box"]
        ),
    )

    elapsed = time.perf_counter() - started

    attribute_names = public_attributes(result)

    attributes = {}

    for name in attribute_names:
        try:
            value = getattr(result, name)
        except Exception as exc:
            attributes[name] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
            continue

        if callable(value):
            attributes[name] = {
                "kind": "callable",
                "repr": repr(value)[:300],
            }
        else:
            attributes[name] = summarize(value)

    if dataclasses.is_dataclass(result):
        dataclass_fields = [
            field.name
            for field in dataclasses.fields(result)
        ]
    else:
        dataclass_fields = []

    candidates = [
        name
        for name in attribute_names
        if candidate_attribute(name)
    ]

    document = {
        "case": case,
        "result_class": {
            "name": type(result).__name__,
            "module": type(result).__module__,
            "is_dataclass": dataclasses.is_dataclass(result),
            "dataclass_fields": dataclass_fields,
        },
        "elapsed_seconds": elapsed,
        "public_attributes": attribute_names,
        "candidate_modal_attributes": candidates,
        "attributes": attributes,
    }

    JSON_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_OUTPUT.write_text(
        json.dumps(document, indent=2) + "\n"
    )

    lines = [
        "# Witness solver result schema",
        "",
        f"- Result class: `{type(result).__module__}.{type(result).__name__}`",
        f"- Dataclass: `{dataclasses.is_dataclass(result)}`",
        f"- Runtime: `{elapsed:.3f} s`",
        "",
        "## Dataclass fields",
        "",
    ]

    if dataclass_fields:
        lines.extend(
            f"- `{name}`"
            for name in dataclass_fields
        )
    else:
        lines.append("- None detected")

    lines.extend([
        "",
        "## Candidate modal attributes",
        "",
    ])

    if candidates:
        for name in candidates:
            summary = attributes.get(name)

            lines.append(f"### `{name}`")
            lines.append("")
            lines.append("```json")
            lines.append(
                json.dumps(
                    summary,
                    indent=2,
                )
            )
            lines.append("```")
            lines.append("")
    else:
        lines.append("- None detected")

    lines.extend([
        "",
        "## All public attributes",
        "",
    ])

    lines.extend(
        f"- `{name}`"
        for name in attribute_names
    )

    MD_OUTPUT.write_text(
        "\n".join(lines) + "\n"
    )

    print("Result class:", type(result).__name__)
    print("Dataclass fields:", dataclass_fields)
    print("Candidate modal attributes:", candidates)
    print("Runtime:", elapsed)
    print("JSON:", JSON_OUTPUT.relative_to(REPO))
    print("Markdown:", MD_OUTPUT.relative_to(REPO))


if __name__ == "__main__":
    main()
