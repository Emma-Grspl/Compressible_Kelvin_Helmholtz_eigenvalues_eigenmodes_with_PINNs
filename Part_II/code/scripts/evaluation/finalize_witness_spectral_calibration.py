#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent

CASE_NAME = "witness_M150_a01625"
SELECTED_CONFIGURATION = "unmapped_y80"

CALIBRATION = (
    PACKAGE
    / "reproducibility"
    / "results"
    / f"{CASE_NAME}_spectral_calibration.csv"
)

CONFIG = (
    PACKAGE
    / "configs"
    / "reproducibility"
    / f"{CASE_NAME}.yaml"
)

EXPECTED = (
    PACKAGE
    / "data"
    / "samples"
    / f"{CASE_NAME}_expected.json"
)

SAMPLE = (
    PACKAGE
    / "data"
    / "samples"
    / f"{CASE_NAME}_raw.parquet"
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
    / "witness_spectral_calibration.md"
)

ACCEPTANCE = {
    "abs_cr_error_max": 1.0e-5,
    "abs_ci_error_max": 2.0e-6,
    "abs_omega_i_error_max": 5.0e-7,
    "stage1_mismatch_max": 1.0e-8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def scalar(row: pd.Series, name: str) -> float:
    value = float(row[name])

    if not pd.notna(value):
        raise ValueError(f"Missing value for {name}")

    return value


def main() -> None:
    for path in [CALIBRATION, CONFIG, EXPECTED, SAMPLE]:
        if not path.exists():
            raise FileNotFoundError(path)

    calibration = pd.read_csv(CALIBRATION)

    selected = calibration[
        calibration["configuration"] == SELECTED_CONFIGURATION
    ]

    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for {SELECTED_CONFIGURATION}; "
            f"found {len(selected)}"
        )

    row = selected.iloc[0]

    if row["status"] != "completed":
        raise ValueError(
            f"Selected calibration did not complete: {row['status']}"
        )

    observed = {
        "cr_solved": scalar(row, "cr_solved"),
        "ci_solved": scalar(row, "ci_solved"),
        "omega_i_solved": scalar(row, "omega_i_solved"),
        "abs_cr_error": scalar(row, "abs_cr_error"),
        "abs_ci_error": scalar(row, "abs_ci_error"),
        "abs_omega_i_error": scalar(row, "abs_omega_i_error"),
        "stage1_mismatch_at_reference": scalar(
            row,
            "stage1_mismatch_at_reference",
        ),
        "stage1_mismatch_solved": scalar(
            row,
            "stage1_mismatch_solved",
        ),
        "elapsed_seconds": scalar(row, "elapsed_seconds"),
    }

    checks = {
        "abs_cr_error": (
            observed["abs_cr_error"]
            <= ACCEPTANCE["abs_cr_error_max"]
        ),
        "abs_ci_error": (
            observed["abs_ci_error"]
            <= ACCEPTANCE["abs_ci_error_max"]
        ),
        "abs_omega_i_error": (
            observed["abs_omega_i_error"]
            <= ACCEPTANCE["abs_omega_i_error_max"]
        ),
        "stage1_mismatch": (
            observed["stage1_mismatch_solved"]
            <= ACCEPTANCE["stage1_mismatch_max"]
        ),
    }

    if not all(checks.values()):
        raise ValueError(
            f"Selected calibration does not satisfy acceptance: {checks}"
        )

    config = yaml.safe_load(CONFIG.read_text())

    config["solver"] = {
        "match_y": 1.0,
        "ln_p_start_left": -5.0,
        "rtol": 1.0e-10,
        "atol": 1.0e-12,
        "max_step": None,
        "min_y_limit": 10.0,
        "max_y_limit": 80.0,
        "y_limit_factor": 4.0,
        "use_mapping": False,
        "mapping_scale": 5.0,
    }

    config["search"] = {
        "cr_half_width": 0.025,
        "ci_half_width": 0.015,
        "grid_size": 5,
        "max_iter": 20,
        "tol": 1.0e-9,
        "constrain_to_initial_box": True,
    }

    config["acceptance"] = ACCEPTANCE

    config["solver_adapter"] = {
        "status": "spectral_reproduction_implemented",
        "selected_configuration": SELECTED_CONFIGURATION,
        "modal_reproduction_status": "pending",
    }

    CONFIG.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        )
    )

    expected = json.loads(EXPECTED.read_text())

    expected["numerical_reproduction"] = {
        "status": "spectral_reproduction_calibrated",
        "selected_configuration": SELECTED_CONFIGURATION,
        "observed": observed,
        "acceptance": ACCEPTANCE,
        "acceptance_checks": checks,
        "modal_tolerances": None,
        "note": (
            "The unmapped y_max=80 configuration was retained because "
            "all three calibration configurations converged to the same "
            "spectral root while this configuration had the lowest runtime."
        ),
    }

    EXPECTED.write_text(
        json.dumps(expected, indent=2) + "\n"
    )

    summary_lines = [
        "# Witness spectral calibration",
        "",
        f"- Case: `{CASE_NAME}`",
        "- Mach: 1.50",
        "- alpha: 0.1625",
        f"- Selected configuration: `{SELECTED_CONFIGURATION}`",
        "",
        "## Calibration result",
        "",
        f"- Solved cr: `{observed['cr_solved']:.16g}`",
        f"- Solved ci: `{observed['ci_solved']:.16g}`",
        f"- Absolute cr error: `{observed['abs_cr_error']:.6e}`",
        f"- Absolute ci error: `{observed['abs_ci_error']:.6e}`",
        (
            "- Absolute omega_i error: "
            f"`{observed['abs_omega_i_error']:.6e}`"
        ),
        (
            "- Stage-1 mismatch at the frozen reference: "
            f"`{observed['stage1_mismatch_at_reference']:.6e}`"
        ),
        (
            "- Stage-1 mismatch at the solved root: "
            f"`{observed['stage1_mismatch_solved']:.6e}`"
        ),
        f"- Runtime: `{observed['elapsed_seconds']:.3f} s`",
        "",
        "## Acceptance thresholds",
        "",
        f"- `|Δcr| <= {ACCEPTANCE['abs_cr_error_max']:.1e}`",
        f"- `|Δci| <= {ACCEPTANCE['abs_ci_error_max']:.1e}`",
        (
            "- `|Δomega_i| <= "
            f"{ACCEPTANCE['abs_omega_i_error_max']:.1e}`"
        ),
        (
            "- `stage1 mismatch <= "
            f"{ACCEPTANCE['stage1_mismatch_max']:.1e}`"
        ),
        "",
        "## Configuration selection",
        "",
        (
            "The unmapped y_max=80, unmapped y_max=500 and mapped "
            "scale-5 y_max=500 configurations converged to the same "
            "spectral root at the reported precision."
        ),
        "",
        (
            "The unmapped y_max=80 configuration is retained for the "
            "release smoke test because it is the least expensive."
        ),
        "",
        (
            "This calibration validates spectral reproducibility only. "
            "Modal-field comparison is handled separately."
        ),
    ]

    SUMMARY.write_text(
        "\n".join(summary_lines) + "\n"
    )

    checksum_paths = [
        SAMPLE,
        EXPECTED,
        CONFIG,
    ]

    with SAMPLE_CHECKSUMS.open("w") as stream:
        for path in checksum_paths:
            stream.write(
                f"{sha256(path)}  {path.relative_to(REPO)}\n"
            )

    print("Finalized witness spectral calibration")
    print("configuration:", CONFIG.relative_to(REPO))
    print("expected:", EXPECTED.relative_to(REPO))
    print("summary:", SUMMARY.relative_to(REPO))
    print("acceptance:", checks)


if __name__ == "__main__":
    main()
