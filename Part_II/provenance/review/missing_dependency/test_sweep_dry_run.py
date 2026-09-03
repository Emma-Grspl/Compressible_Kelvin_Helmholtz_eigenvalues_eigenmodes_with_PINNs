from __future__ import annotations

from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PACKAGE_ROOT / "scripts/convergence"
sys.path.insert(0, str(SCRIPT_DIR))

import run_classical_convergence_sweep as runner  # noqa: E402


CONFIG_DIR = PACKAGE_ROOT / "configs/convergence"


def test_dry_run_never_calls_a_solver_or_writes_output(monkeypatch, tmp_path: Path):
    def forbidden(*args, **kwargs):
        raise AssertionError("execute_run must not be called by --dry-run")

    monkeypatch.setattr(runner, "execute_run", forbidden)
    output = tmp_path / "must_not_exist"
    result = runner.main(
        [
            "--case-id",
            "subsonic_interior",
            "--sweep-config",
            str(CONFIG_DIR / "shooting_box_sweep.yaml"),
            "--output-dir",
            str(output),
            "--max-runs",
            "1",
            "--dry-run",
        ]
    )
    assert result == 0
    assert not output.exists()


def test_existing_run_is_not_overwritten_without_force(tmp_path: Path):
    cases = runner.load_yaml(CONFIG_DIR / "cases.yaml")
    sweep = runner.load_yaml(CONFIG_DIR / "shooting_box_sweep.yaml")
    configuration = runner.build_run_matrix(cases, sweep, case_id="subsonic_interior")[0]
    (tmp_path / "runs" / configuration["run_id"]).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="--force"):
        runner.execute_run(configuration, tmp_path)


def test_max_step_none_is_only_converted_for_supported_solver():
    assert runner.normalize_max_step(None, supported=False) is None
    assert runner.normalize_max_step(None, supported=True) == float("inf")
    assert runner.normalize_max_step(0.5, supported=True) == 0.5
    with pytest.raises(ValueError, match="not supported"):
        runner.normalize_max_step(0.5, supported=False)
    with pytest.raises(ValueError, match="positive"):
        runner.normalize_max_step(0.0, supported=True)


def test_gep_module_and_cli_parser_are_functional_in_runner():
    module = runner.load_gep_module()
    parser = module.build_parser()
    args = parser.parse_args([])
    assert args.n_points == 301
    assert module.NotebookStyleDenseGEPSolver is not None


def test_max_step_audit_uses_the_frozen_modal_reconstruction():
    audit = PACKAGE_ROOT / "scripts/validation/audit_M18_M19_max_step_convergence.py"
    source = audit.read_text(encoding="utf-8")
    assert (
        "from classic_supersonic_reference.validation.modal_reconstruction "
        "import reconstruct_from_solver"
    ) in source
    assert "from scripts.audits.audit_supersonic_shooting_visual_validation_6969b4f1bf" not in source
    assert "max_step=max_step" in source
