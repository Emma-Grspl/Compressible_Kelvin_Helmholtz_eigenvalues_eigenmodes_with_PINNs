from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PACKAGE_ROOT / "scripts/convergence"
SRC_ROOT = PACKAGE_ROOT / "src"
for path in (SCRIPT_DIR, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.legacy.compute_classical_convergence_errors as errors  # noqa: E402
import plots.scripts.classic_supersonic.plot_classical_convergence as plotting  # noqa: E402
import run_classical_convergence_sweep as runner  # noqa: E402
from classic_supersonic_reference.solver.mstab17_supersonic_solver import (  # noqa: E402
    Mstab17SupersonicResult,
    Mstab17SupersonicSolver,
)
from classic_supersonic_reference.validation.modal_reconstruction import (  # noqa: E402
    _relative_matching_metrics,
)


CONFIG_DIR = PACKAGE_ROOT / "configs/convergence"
FIELDS = ("p", "rho", "u", "v")


def _fields(scale: complex = 1.0 + 0.0j) -> dict[str, np.ndarray]:
    y = np.linspace(-4.0, 4.0, 129)
    envelope = np.exp(-y**2)
    return {
        "y": y,
        **{
            name: scale * (index + 0.2j) * envelope
            for index, name in enumerate(FIELDS, start=1)
        },
    }


def _write_mode(path: Path, fields: dict[str, np.ndarray]) -> None:
    payload: dict[str, np.ndarray] = {"y": fields["y"]}
    for name in FIELDS:
        payload[f"{name}_real"] = fields[name].real
        payload[f"{name}_imag"] = fields[name].imag
    np.savez_compressed(path, **payload)


def _smoke_configuration(case_id: str) -> dict:
    cases = runner.load_yaml(CONFIG_DIR / "cases.yaml")
    sweep = runner.load_yaml(CONFIG_DIR / "shooting_modalfix_smoke.yaml")
    return runner.build_run_matrix(cases, sweep, case_id=case_id)[0]


def _fake_shooting_result(configuration: dict, *, legacy_success: bool = False):
    return (
        {
            "cr": float(configuration["reference_cr"]),
            "ci": float(configuration["reference_ci"]),
            "omega_i": float(configuration["alpha"] * configuration["reference_ci"]),
            "riccati_mismatch": 1.0e-8,
            "amplitude_mismatch": 1000.0,
            "legacy_stage2_mismatch": 1000.0,
            "solver_success": legacy_success,
            "spectral_success": True,
            "modal_reconstruction_success": True,
            "legacy_mode_success": legacy_success,
            "legacy_success": legacy_success,
            "legacy_stage2_bound_hit": True,
            "ln_p_start_right_exact": -52.0,
            "relative_p_window_error": 1.0e-8,
            "relative_py_window_error": 1.0e-8,
            "relative_p_jump": 1.0e-8,
            "relative_py_jump": 1.0e-8,
            "finite_fields": True,
            "nondegenerate_fields": True,
            "pressure_ode_residual_left": 1.0e-6,
            "pressure_ode_residual_right": 1.0e-6,
            "riccati_residual_left": 1.0e-6,
            "riccati_residual_right": 1.0e-6,
        },
        _fields(),
    )


def test_solver_detects_historical_stage2_bound_hit(monkeypatch):
    solver = Mstab17SupersonicSolver(alpha=0.1, Mach=1.9)
    monkeypatch.setattr(solver, "solve_eigenvalue", lambda **kwargs: (0.48, 0.022, 1.0e-8))
    monkeypatch.setattr(solver, "stage2_objective", lambda value, cr, ci: (value + 52.0) ** 2)
    monkeypatch.setattr(solver, "exact_right_log_amplitude", lambda cr, ci: -52.0)
    trajectory = SimpleNamespace(success=True)
    monkeypatch.setattr(
        solver,
        "get_trajectories",
        lambda *args, **kwargs: (trajectory, trajectory, trajectory, 2000.0),
    )

    result = solver.solve()

    assert result.ln_p_start_right_exact == -52.0
    assert result.legacy_stage2_bound_hit
    assert not result.legacy_mode_success
    assert result.mode_success == result.legacy_mode_success
    assert result.success == result.legacy_success


def test_result_constructor_accepts_the_historical_field_list():
    result = Mstab17SupersonicResult(
        alpha=0.1,
        Mach=1.9,
        cr=0.48,
        ci=0.022,
        omega_i=0.0022,
        stage1_mismatch=1.0e-8,
        stage2_mismatch=1.0,
        y_limit=2000.0,
        ln_p_start_right=-15.0,
        spectral_success=True,
        mode_success=False,
        success=False,
        use_mapping=True,
        mapping_scale=3.0,
    )
    assert result.legacy_mode_success is result.mode_success
    assert result.legacy_success is result.success


def test_relative_matching_metrics_are_invariant_to_global_complex_scale():
    y = np.linspace(0.0, 0.25, 64)
    left = np.exp((-0.2 + 0.3j) * y)
    right = left / (2.0 * np.exp(0.4j))
    left_y = (-0.2 + 0.3j) * left
    right_y = (-0.2 + 0.3j) * right
    kwargs = {
        "p_left": left,
        "p_right": right,
        "py_left": left_y,
        "py_right": right_y,
        "p_left_jump": left[0],
        "p_right_jump": right[0],
        "py_left_jump": left_y[0],
        "py_right_jump": right_y[0],
        "alpha": 0.1,
    }
    baseline = _relative_matching_metrics(**kwargs)
    scale = 3.7 * np.exp(-0.8j)
    scaled = _relative_matching_metrics(
        **{
            key: value * scale if key != "alpha" else value
            for key, value in kwargs.items()
        }
    )
    for key in (
        "relative_p_window_error",
        "relative_py_window_error",
        "relative_p_jump",
        "relative_py_jump",
    ):
        assert np.isclose(baseline[key], scaled[key], atol=1.0e-14, rtol=0.0)


def test_m19_exports_mode_after_legacy_rejection_and_tail_is_not_failure(monkeypatch, tmp_path: Path):
    configuration = _smoke_configuration("supersonic_M190")
    monkeypatch.setattr(
        runner,
        "_run_supersonic_shooting",
        lambda config: _fake_shooting_result(config),
    )

    manifest = runner.execute_run(configuration, tmp_path)

    assert manifest["ln_p_start_right_exact"] < -15.0
    assert manifest["legacy_stage2_bound_hit"]
    assert not manifest["legacy_success"]
    assert not manifest["converged"]
    assert manifest["spectral_success"]
    assert manifest["modal_reconstruction_success"]
    assert manifest["tail_sensitivity_flag"]
    assert manifest["overall_validated"]
    assert manifest["mode_file"]
    assert (tmp_path / "runs" / configuration["run_id"] / "mode.npz").exists()


def test_m14_exports_diagnostic_mode_but_stays_unvalidated(monkeypatch, tmp_path: Path):
    configuration = _smoke_configuration("supersonic_M140")
    monkeypatch.setattr(
        runner,
        "_run_supersonic_shooting",
        lambda config: _fake_shooting_result(config),
    )

    manifest = runner.execute_run(configuration, tmp_path)

    assert manifest["modal_reconstruction_success"]
    assert manifest["branch_distance_check_passed"]
    assert not manifest["branch_check_passed"]
    assert not manifest["overall_validated"]
    assert manifest["branch_provenance_status"] == "unresolved_current_solver_mismatch"
    assert manifest["branch_candidate_used"] == "configured_reference"
    assert manifest["mode_file"]


def test_runner_retains_legacy_and_new_status_columns():
    required = {
        "converged",
        "amplitude_mismatch",
        "solver_success",
        "legacy_stage2_mismatch",
        "legacy_mode_success",
        "legacy_success",
        "legacy_stage2_bound_hit",
        "ln_p_start_right_exact",
        "spectral_success",
        "modal_reconstruction_success",
        "branch_check_passed",
        "tail_sensitivity_flag",
        "overall_validated",
    }
    assert required <= set(runner.RUN_COLUMNS)


def test_m14_configuration_records_both_branch_candidates():
    configuration = _smoke_configuration("supersonic_M140")
    assert configuration["branch_provenance_status"] == "unresolved_current_solver_mismatch"
    assert set(configuration["branch_candidates"]) == {
        "configured_reference",
        "current_shooting_root",
    }


def test_modal_errors_are_long_format_for_multiple_core_thresholds(tmp_path: Path):
    reference_path = tmp_path / "reference.npz"
    perturbed_path = tmp_path / "perturbed.npz"
    _write_mode(reference_path, _fields())
    perturbed = _fields()
    perturbed["p"] = perturbed["p"].copy()
    perturbed["p"][np.abs(perturbed["y"]) > 2.0] += 1.0e-3
    _write_mode(perturbed_path, perturbed)
    base = {
        "case_id": "supersonic_M180",
        "regime": "supersonic",
        "solver": "supersonic_shooting",
        "sweep_type": "shooting_box",
        "Mach": 1.8,
        "alpha": 0.1,
        "runtime_seconds": 1.0,
        "converged": False,
        "spectral_success": True,
        "modal_reconstruction_success": True,
        "branch_check_passed": True,
        "overall_validated": True,
        "branch_distance": 0.0,
        "accuracy_label": None,
        "accuracy_order": None,
    }
    runs = pd.DataFrame(
        [
            {**base, "run_id": "small", "Ly": 1000.0, "cr": 0.45, "ci": 0.027, "omega_i": 0.0027, "mode_file": str(perturbed_path)},
            {**base, "run_id": "large", "Ly": 2000.0, "cr": 0.45, "ci": 0.027, "omega_i": 0.0027, "mode_file": str(reference_path)},
            {**base, "run_id": "no_mode", "Ly": 1500.0, "cr": 0.45, "ci": 0.027, "omega_i": 0.0027, "mode_file": None, "modal_reconstruction_success": False, "overall_validated": False},
        ]
    )

    frame = errors.compute_errors_frame(
        runs,
        core_thresholds=[1.0e-3, 5.0e-2],
        common_grid_size=128,
    )

    assert len(frame) == 6
    assert set(frame["core_threshold"]) == {1.0e-3, 5.0e-2}
    assert frame.groupby("run_id").size().eq(2).all()
    assert frame.loc[frame["run_id"].eq("no_mode"), "mode_error_max_core"].isna().all()
    assert frame.loc[frame["run_id"].eq("no_mode"), "modal_error_message"].eq("mode unavailable").all()

    csv_path = tmp_path / "errors.csv"
    frame.to_csv(csv_path, index=False)
    assert plotting.main(["--errors-csv", str(csv_path), "--figure", "shooting-spectral", "--dry-run"]) == 0
