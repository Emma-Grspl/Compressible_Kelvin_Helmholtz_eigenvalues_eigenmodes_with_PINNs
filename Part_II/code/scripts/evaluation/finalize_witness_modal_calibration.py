#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent

RUN_DIR = (
    PACKAGE
    / "reproducibility"
    / "results"
    / "modal_repeats"
)

CONFIG = (
    PACKAGE
    / "configs"
    / "reproducibility"
    / "witness_M150_a01625.yaml"
)

EXPECTED = (
    PACKAGE
    / "data"
    / "samples"
    / "witness_M150_a01625_expected.json"
)

SAMPLE = (
    PACKAGE
    / "data"
    / "samples"
    / "witness_M150_a01625_raw.parquet"
)

SAMPLE_CHECKSUMS = (
    PACKAGE
    / "data"
    / "samples"
    / "SHA256SUMS.txt"
)

SUMMARY = (
    PACKAGE
    / "provenance"
    / "witness_modal_calibration.md"
)

FIELDS = ("rho", "u", "v", "p")

METRICS = (
    "relative_l2_core_amp_ge_5pct",
    "relative_l2_full_overlap",
    "relative_l2_tail_amp_0p1pct_to_2pct",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def finite(value: Any) -> float:
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"Non-finite value: {value!r}")

    return result


def round_up_one_significant_digit(
    value: float,
) -> float:
    if value <= 0:
        return 0.0

    exponent = math.floor(math.log10(value))
    unit = 10.0**exponent

    return float(
        math.ceil(value / unit - 1.0e-12)
        * unit
    )


def main() -> None:
    run_paths = sorted(RUN_DIR.glob("run_*.json"))

    if len(run_paths) < 3:
        raise ValueError(
            f"Expected at least 3 modal calibration runs, "
            f"found {len(run_paths)}"
        )

    documents = [
        json.loads(path.read_text())
        for path in run_paths
    ]

    variants_required = sorted(
        {
            variant["variant"]
            for document in documents
            for variant in document[
                "comparison"
            ]["variants"]
        }
    )

    observed: dict[str, dict[str, list[float]]] = {
        field: {
            metric: []
            for metric in METRICS
        }
        for field in FIELDS
    }

    correlations = []
    overlap_counts = []
    alignment_deviations = []

    for document in documents:
        variants = document[
            "comparison"
        ]["variants"]

        observed_names = sorted(
            variant["variant"]
            for variant in variants
        )

        if observed_names != variants_required:
            raise ValueError(
                "Inconsistent modal variants between runs"
            )

        for variant in variants:
            overlap_counts.append(
                int(variant["n_overlap"])
            )

            alignment_absolute = finite(
                variant["alignment"]["absolute"]
            )

            alignment_deviations.append(
                abs(alignment_absolute - 1.0)
            )

            for field in FIELDS:
                field_metrics = variant[
                    "fields"
                ][field]

                for metric in METRICS:
                    value = field_metrics[metric]

                    if value is None:
                        raise ValueError(
                            f"Missing {metric} for "
                            f"{variant['variant']}:{field}"
                        )

                    observed[field][metric].append(
                        finite(value)
                    )

                correlations.append(
                    finite(
                        field_metrics[
                            "complex_correlation_core"
                        ]
                    )
                )

    field_acceptance = {}

    for field in FIELDS:
        field_acceptance[field] = {
            (
                "relative_l2_core_"
                "amp_ge_5pct_max"
            ): round_up_one_significant_digit(
                2.0
                * max(
                    observed[field][
                        "relative_l2_core_amp_ge_5pct"
                    ]
                )
            ),
            "relative_l2_full_overlap_max": (
                round_up_one_significant_digit(
                    2.0
                    * max(
                        observed[field][
                            "relative_l2_full_overlap"
                        ]
                    )
                )
            ),
            (
                "relative_l2_tail_"
                "amp_0p1pct_to_2pct_max"
            ): round_up_one_significant_digit(
                2.0
                * max(
                    observed[field][
                        "relative_l2_tail_"
                        "amp_0p1pct_to_2pct"
                    ]
                )
            ),
        }

    minimum_correlation_observed = min(
        correlations
    )

    correlation_candidate = (
        1.0
        - 2.0
        * (
            1.0
            - minimum_correlation_observed
        )
    )

    correlation_minimum = (
        math.floor(
            correlation_candidate * 1.0e5
        )
        / 1.0e5
    )

    minimum_overlap_observed = min(
        overlap_counts
    )

    minimum_overlap_required = (
        minimum_overlap_observed // 100
    ) * 100

    maximum_alignment_deviation = (
        round_up_one_significant_digit(
            2.0
            * max(alignment_deviations)
        )
    )

    acceptance = {
        "calibration_runs": len(run_paths),
        "safety_factor": 2.0,
        "variants_required": variants_required,
        "min_overlap_points": int(
            minimum_overlap_required
        ),
        "complex_correlation_core_min": float(
            correlation_minimum
        ),
        "alignment_abs_deviation_max": float(
            maximum_alignment_deviation
        ),
        "fields": field_acceptance,
    }

    observed_summary = {
        "minimum_overlap_points": int(
            minimum_overlap_observed
        ),
        "minimum_complex_correlation_core": (
            minimum_correlation_observed
        ),
        "maximum_alignment_abs_deviation": (
            max(alignment_deviations)
        ),
        "fields": {
            field: {
                metric: max(values)
                for metric, values in metrics.items()
            }
            for field, metrics in observed.items()
        },
    }

    config = yaml.safe_load(CONFIG.read_text())

    config["modal_acceptance"] = acceptance

    solver_adapter = config.setdefault(
        "solver_adapter",
        {},
    )

    solver_adapter[
        "modal_reproduction_status"
    ] = "calibrated"

    CONFIG.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        )
    )

    expected = json.loads(EXPECTED.read_text())

    numerical = expected.setdefault(
        "numerical_reproduction",
        {},
    )

    numerical["status"] = (
        "spectral_and_modal_reproduction_calibrated"
    )

    numerical["modal_reproduction"] = {
        "status": "calibrated",
        "calibration_runs": len(run_paths),
        "acceptance": acceptance,
        "observed_extrema": observed_summary,
        "comparison_variants": variants_required,
        "note": (
            "Thresholds were derived from three repeated modal "
            "reconstructions. A factor of two was applied to the "
            "largest observed relative errors before rounding upward."
        ),
    }

    EXPECTED.write_text(
        json.dumps(expected, indent=2) + "\n"
    )

    lines = [
        "# Witness modal calibration",
        "",
        "- Case: `witness_M150_a01625`",
        f"- Repeated runs: `{len(run_paths)}`",
        "- Safety factor applied to errors: `2`",
        "",
        "## Global observations",
        "",
        (
            "- Minimum overlap points: "
            f"`{minimum_overlap_observed}`"
        ),
        (
            "- Minimum core complex correlation: "
            f"`{minimum_correlation_observed:.12g}`"
        ),
        (
            "- Maximum alignment amplitude deviation: "
            f"`{max(alignment_deviations):.6e}`"
        ),
        "",
        "## Acceptance thresholds",
        "",
        (
            "- Minimum overlap points: "
            f"`{minimum_overlap_required}`"
        ),
        (
            "- Minimum core complex correlation: "
            f"`{correlation_minimum}`"
        ),
        (
            "- Maximum alignment amplitude deviation: "
            f"`{maximum_alignment_deviation:.6e}`"
        ),
        "",
    ]

    for field in FIELDS:
        thresholds = field_acceptance[field]
        maxima = observed_summary["fields"][field]

        lines.extend(
            [
                f"### `{field}`",
                "",
                (
                    "- Maximum observed core relative L2: "
                    f"`{maxima['relative_l2_core_amp_ge_5pct']:.6e}`"
                ),
                (
                    "- Accepted core relative L2: "
                    f"`{thresholds['relative_l2_core_amp_ge_5pct_max']:.6e}`"
                ),
                (
                    "- Maximum observed full-overlap relative L2: "
                    f"`{maxima['relative_l2_full_overlap']:.6e}`"
                ),
                (
                    "- Accepted full-overlap relative L2: "
                    f"`{thresholds['relative_l2_full_overlap_max']:.6e}`"
                ),
                (
                    "- Maximum observed tail relative L2: "
                    f"`{maxima['relative_l2_tail_amp_0p1pct_to_2pct']:.6e}`"
                ),
                (
                    "- Accepted tail relative L2: "
                    f"`{thresholds['relative_l2_tail_amp_0p1pct_to_2pct_max']:.6e}`"
                ),
                "",
            ]
        )

    SUMMARY.write_text(
        "\n".join(lines) + "\n"
    )

    with SAMPLE_CHECKSUMS.open("w") as stream:
        for path in [SAMPLE, EXPECTED, CONFIG]:
            stream.write(
                f"{sha256(path)}  "
                f"{path.relative_to(REPO)}\n"
            )

    print("Finalized witness modal calibration")
    print(json.dumps(acceptance, indent=2))
    print("Wrote:", CONFIG.relative_to(REPO))
    print("Wrote:", EXPECTED.relative_to(REPO))
    print("Wrote:", SUMMARY.relative_to(REPO))


if __name__ == "__main__":
    main()
