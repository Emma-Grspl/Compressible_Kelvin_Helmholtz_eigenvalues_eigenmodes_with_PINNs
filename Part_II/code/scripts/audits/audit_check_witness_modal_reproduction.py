#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


PACKAGE = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = (
    PACKAGE
    / "configs"
    / "reproducibility"
    / "witness_M150_a01625.yaml"
)

DEFAULT_CALIBRATION = (
    PACKAGE
    / "reproducibility"
    / "results"
    / "witness_M150_a01625_modal_calibration.json"
)

DEFAULT_OUTPUT = (
    PACKAGE
    / "reproducibility"
    / "results"
    / "witness_M150_a01625_modal_reproduction_check.json"
)


def finite(value: Any) -> float:
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
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
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

    document = json.loads(
        args.calibration.read_text()
    )

    acceptance = config["modal_acceptance"]

    variants = {
        variant["variant"]: variant
        for variant in document[
            "comparison"
        ]["variants"]
    }

    checks = {}

    required_variants = acceptance[
        "variants_required"
    ]

    for name in required_variants:
        checks[
            f"variant:{name}:present"
        ] = name in variants

    for name in required_variants:
        if name not in variants:
            continue

        variant = variants[name]

        checks[
            f"variant:{name}:overlap"
        ] = (
            int(variant["n_overlap"])
            >= int(
                acceptance[
                    "min_overlap_points"
                ]
            )
        )

        alignment_deviation = abs(
            finite(
                variant["alignment"]["absolute"]
            )
            - 1.0
        )

        checks[
            f"variant:{name}:alignment"
        ] = (
            alignment_deviation
            <= finite(
                acceptance[
                    "alignment_abs_deviation_max"
                ]
            )
        )

        for field, thresholds in acceptance[
            "fields"
        ].items():
            metrics = variant["fields"][field]

            checks[
                f"{name}:{field}:core_l2"
            ] = (
                finite(
                    metrics[
                        "relative_l2_core_amp_ge_5pct"
                    ]
                )
                <= finite(
                    thresholds[
                        "relative_l2_core_amp_ge_5pct_max"
                    ]
                )
            )

            checks[
                f"{name}:{field}:full_l2"
            ] = (
                finite(
                    metrics[
                        "relative_l2_full_overlap"
                    ]
                )
                <= finite(
                    thresholds[
                        "relative_l2_full_overlap_max"
                    ]
                )
            )

            checks[
                f"{name}:{field}:tail_l2"
            ] = (
                finite(
                    metrics[
                        "relative_l2_tail_amp_0p1pct_to_2pct"
                    ]
                )
                <= finite(
                    thresholds[
                        "relative_l2_tail_amp_0p1pct_to_2pct_max"
                    ]
                )
            )

            checks[
                f"{name}:{field}:core_correlation"
            ] = (
                finite(
                    metrics[
                        "complex_correlation_core"
                    ]
                )
                >= finite(
                    acceptance[
                        "complex_correlation_core_min"
                    ]
                )
            )

    passed = bool(checks) and all(
        checks.values()
    )

    result = {
        "case": document["case"],
        "calibration_path": str(
            args.calibration
        ),
        "acceptance": acceptance,
        "checks": checks,
        "failed_checks": [
            name
            for name, value in checks.items()
            if not value
        ],
        "passed": passed,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print(json.dumps(result, indent=2))
    print()
    print("Wrote:", args.output)

    if not passed:
        raise SystemExit(
            "Witness modal reproduction failed."
        )


if __name__ == "__main__":
    main()
