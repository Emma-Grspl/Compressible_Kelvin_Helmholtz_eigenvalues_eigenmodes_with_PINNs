from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PACKAGE_ROOT / "scripts/convergence"
sys.path.insert(0, str(SCRIPT_DIR))

from run_classical_convergence_sweep import (  # noqa: E402
    build_run_matrix,
    load_yaml,
    validate_cases_document,
    validate_sweep_document,
)


CONFIG_DIR = PACKAGE_ROOT / "configs/convergence"
CASES_PATH = CONFIG_DIR / "cases.yaml"
SWEEP_PATHS = [
    CONFIG_DIR / "shooting_box_sweep.yaml",
    CONFIG_DIR / "shooting_accuracy_sweep.yaml",
    CONFIG_DIR / "shooting_modalfix_smoke.yaml",
    CONFIG_DIR / "supersonic_gep_resolution_sweep.yaml",
]


def test_cases_yaml_contains_the_five_audited_cases():
    document = load_yaml(CASES_PATH)
    cases = validate_cases_document(document)
    assert set(cases) == {
        "subsonic_interior",
        "subsonic_neutral",
        "supersonic_M140",
        "supersonic_M180",
        "supersonic_M190",
    }


def test_six_bad_fixed_ci_points_are_explicitly_excluded():
    document = load_yaml(CASES_PATH)
    excluded = {
        (float(item["Mach"]), float(item["alpha"]))
        for item in document["excluded_fixed_ci_points"]
    }
    assert excluded == {
        (1.8, 0.238),
        (1.8, 0.248),
        (1.8, 0.258),
        (1.9, 0.232),
        (1.9, 0.242),
        (1.9, 0.252),
    }


def test_excluded_point_cannot_be_added_as_case():
    document = deepcopy(load_yaml(CASES_PATH))
    document["cases"][0]["Mach"] = 1.8
    document["cases"][0]["alpha"] = 0.238
    with pytest.raises(ValueError, match="Excluded fixed-ci point"):
        validate_cases_document(document)


@pytest.mark.parametrize("sweep_path", SWEEP_PATHS)
def test_sweep_yaml_is_valid_and_has_unique_deterministic_run_ids(sweep_path: Path):
    cases_document = load_yaml(CASES_PATH)
    cases = validate_cases_document(cases_document)
    sweep_document = load_yaml(sweep_path)
    validate_sweep_document(sweep_document, cases)
    first = build_run_matrix(cases_document, sweep_document)
    second = build_run_matrix(cases_document, sweep_document)
    first_ids = [item["run_id"] for item in first]
    second_ids = [item["run_id"] for item in second]
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))
    assert [item["config_hash"] for item in first] == [item["config_hash"] for item in second]
