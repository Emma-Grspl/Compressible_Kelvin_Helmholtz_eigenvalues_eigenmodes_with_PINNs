from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
POINTWISE = REPO_ROOT / "classic_supersonic/configs/convergence/pointwise"
CASES = POINTWISE / "pointwise_cases.yaml"
RUNNER = REPO_ROOT / "experiments/atlas_12charts/support/logs/smoke_C00_834382.out"
PLOTTER_PATH = REPO_ROOT / "plots/scripts/pinn_supersonic/plot_pointwise_convergence.py"
ERRORS_PATH = REPO_ROOT / "code/scripts/legacy/compute_classical_convergence_errors.py"
SWEEPS = sorted(POINTWISE.glob("*_sweep.yaml"))


def load_plotter():
    spec = importlib.util.spec_from_file_location("pointwise_plotter", PLOTTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pointwise_cases_have_exact_requested_coverage():
    document = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    frame = pd.DataFrame(document["cases"])
    expected = {
        "subsonic": np.round(np.arange(0.1, 1.0, 0.1), 1).tolist(),
        "supersonic": np.round(np.arange(1.1, 2.0, 0.1), 1).tolist(),
    }
    for regime, machs in expected.items():
        selected = frame[frame["regime"].eq(regime)]
        assert len(selected) == 9
        assert sorted(np.round(selected["Mach"], 1)) == machs
        assert selected.groupby("Mach")["alpha"].nunique().eq(1).all()
    supersonic = set(np.round(frame[frame["regime"].eq("supersonic")]["Mach"], 2))
    assert 1.25 not in supersonic
    assert 1.33 not in supersonic
    assert 1.4 in supersonic


@pytest.mark.parametrize("sweep", SWEEPS, ids=lambda path: path.stem)
def test_each_pointwise_sweep_dry_run_is_nine_by_four(sweep: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--cases-config",
            str(CASES),
            "--sweep-config",
            str(sweep),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = __import__("json").loads(completed.stdout)
    assert payload["n_runs"] == 36
    runs = pd.DataFrame(payload["runs"])
    assert runs["Mach"].nunique() == 9
    assert runs.groupby("case_id")["run_id"].nunique().eq(4).all()
    references = runs[runs["run_id"].eq(runs["reference_run_id"])]
    assert len(references) == 9
    assert (runs["case_id"] == runs["reference_case_id"]).all()


def synthetic_errors(regime: str, sweep_type: str) -> pd.DataFrame:
    machs = np.arange(0.1, 1.0, 0.1) if regime == "subsonic" else np.arange(1.1, 2.0, 0.1)
    records = []
    for mach_index, mach in enumerate(machs):
        case_id = f"{regime}_M{int(round(100 * mach)):03d}"
        reference_run_id = f"{case_id}_run3"
        for level in range(4):
            records.append(
                {
                    "run_id": f"{case_id}_run{level}",
                    "reference_run_id": reference_run_id,
                    "reference_policy": "same_case_most_resolved",
                    "case_id": case_id,
                    "regime": regime,
                    "sweep_type": sweep_type,
                    "Mach": round(float(mach), 1),
                    "alpha": 0.2 + 0.01 * mach_index,
                    "Ly": 40.0 * (level + 1),
                    "accuracy_order": level,
                    "accuracy_label": f"level_{level}",
                    "max_step": 2.0 / (2**level),
                    "core_threshold": 1.0e-3,
                    "abs_error_ci": 0.0 if level == 3 else 1.0e-3 / (level + 1),
                    "abs_error_omega_i": 0.0 if level == 3 else 5.0e-4 / (level + 1),
                    "complex_error_c": 0.0 if level == 3 else 2.0e-3 / (level + 1),
                    "mode_error_max_core": 0.0 if level == 3 else 3.0e-2 / (level + 1),
                    "mode_error_max_full": 0.0 if level == 3 else 5.0e-2 / (level + 1),
                }
            )
    return pd.DataFrame(records)


@pytest.mark.parametrize(
    ("regime", "sweep_type", "level_text"),
    [
        ("subsonic", "shooting_box", "4 box sizes"),
        ("subsonic", "shooting_accuracy", "4 tolerance levels"),
        ("supersonic", "shooting_box", "4 box sizes"),
        ("supersonic", "shooting_accuracy", "4 step sizes"),
    ],
)
def test_plot_title_and_reference_exclusion(regime: str, sweep_type: str, level_text: str):
    plotter = load_plotter()
    frame = synthetic_errors(regime, sweep_type)
    figure = plotter.build_pointwise_figure(frame, regime=regime, sweep_type=sweep_type)
    assert "9 Mach cases" in figure._suptitle.get_text()
    assert level_text in figure._suptitle.get_text()
    assert len(figure.axes[0].lines) == 9
    assert all(len(line.get_xdata()) == 3 for line in figure.axes[0].lines)


def test_output_destinations_are_regime_specific():
    plotter = load_plotter()
    for (regime, _), (directory, _) in plotter.OUTPUTS.items():
        expected = f"assets/classic_{regime}/article"
        assert expected in directory.as_posix()


def test_no_artificial_1e_minus_16_floor():
    assert "1e-16" not in ERRORS_PATH.read_text(encoding="utf-8").lower()


def test_plot_validation_rejects_cross_point_reference():
    plotter = load_plotter()
    frame = synthetic_errors("supersonic", "shooting_box")
    frame.loc[frame["case_id"].eq("supersonic_M110"), "reference_run_id"] = "supersonic_M120_run3"
    with pytest.raises(ValueError, match="reference run|Reference run|Cross-point"):
        plotter.validate_pointwise_frame(frame, regime="supersonic", sweep_type="shooting_box")
