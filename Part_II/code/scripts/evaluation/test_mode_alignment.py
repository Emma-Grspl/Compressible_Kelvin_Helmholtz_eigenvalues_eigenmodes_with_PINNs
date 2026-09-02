from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PACKAGE_ROOT / "scripts/convergence"
sys.path.insert(0, str(SCRIPT_DIR))

from scripts.legacy.compute_classical_convergence_errors import compute_modal_errors, optimal_complex_factor  # noqa: E402


FIELDS = ("p", "rho", "u", "v")


def make_mode(y: np.ndarray, scale: complex = 1.0 + 0.0j) -> dict[str, np.ndarray]:
    envelope = np.exp(-y**2)
    phases = {"p": 0.0, "rho": 0.2, "u": -0.4, "v": 0.7}
    mode: dict[str, np.ndarray] = {"y": y}
    for index, field in enumerate(FIELDS, start=1):
        values = index * envelope * np.exp(1j * phases[field]) / scale
        mode[f"{field}_real"] = values.real
        mode[f"{field}_imag"] = values.imag
    return mode


def test_optimal_complex_factor_recovers_amplitude_and_phase():
    reference = np.array([1 + 2j, 2 - 1j, -0.5 + 0.3j])
    expected = 2.5 * np.exp(0.7j)
    predicted = reference / expected
    observed = optimal_complex_factor(predicted, reference)
    assert np.isclose(observed, expected, atol=1e-12, rtol=0)


def test_complex_alignment_makes_scaled_mode_errors_negligible():
    y = np.linspace(-4.0, 4.0, 501)
    reference = make_mode(y)
    predicted = make_mode(y, scale=2.3 * np.exp(0.61j))
    errors = compute_modal_errors(predicted, reference, core_threshold=1e-3, common_grid_size=600)
    assert errors["phase_alignment_residual"] < 1e-11
    assert errors["mode_error_max_core"] < 1e-11
    assert errors["mode_error_max_full"] < 1e-11


def test_full_error_detects_tail_change_more_than_core_error():
    y = np.linspace(-5.0, 5.0, 801)
    reference = make_mode(y)
    predicted = make_mode(y)
    tail = np.abs(y) > 3.0
    predicted["p_real"] = predicted["p_real"].copy()
    predicted["p_real"][tail] += 0.02
    errors = compute_modal_errors(predicted, reference, core_threshold=1e-2, common_grid_size=800)
    assert errors["mode_error_p_full"] > errors["mode_error_p_core"]
