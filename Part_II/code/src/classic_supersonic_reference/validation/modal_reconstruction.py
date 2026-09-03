from __future__ import annotations

from typing import Any

import numpy as np

from classic_supersonic_reference.solver.mstab17_supersonic_solver import (
    Mstab17SupersonicSolver,
)


FIELD_NAMES = ("rho", "u", "v", "p")
ASSEMBLY_CHOICES = ("zero", "matching_y")


def normalize_fields(
    *,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    rho: np.ndarray,
) -> dict[str, np.ndarray]:
    del y

    index = int(np.argmax(np.abs(rho)))

    if np.abs(rho[index]) > 0.0:
        phase = np.exp(-1j * np.angle(rho[index]))
        u = u * phase
        v = v * phase
        p = p * phase
        rho = rho * phase

    if np.max(np.real(rho)) < abs(np.min(np.real(rho))):
        u = -u
        v = -v
        p = -p
        rho = -rho

    scale = max(
        float(np.max(np.abs(np.real(rho)))),
        float(np.max(np.abs(np.imag(rho)))),
        1.0e-12,
    )

    return {
        "u": u / scale,
        "v": v / scale,
        "p": p / scale,
        "rho": rho / scale,
    }


def exact_right_log_amplitude(
    solver: Mstab17SupersonicSolver,
    *,
    cr: float,
    ci: float,
) -> float:
    return solver.exact_right_log_amplitude(cr, ci)


def _interp_components(solution, target: np.ndarray) -> tuple[np.ndarray, ...]:
    t = np.asarray(solution.t, dtype=float)
    state = np.asarray(solution.y, dtype=float)
    order = np.argsort(t)
    return tuple(np.interp(target, t[order], values[order]) for values in state)


def _optimal_complex_factor(source: np.ndarray, target: np.ndarray) -> complex:
    denominator = np.vdot(source, source)
    if abs(denominator) <= 1.0e-300:
        raise ValueError("Cannot align a degenerate pressure branch.")
    return complex(np.vdot(source, target) / denominator)


def _relative_matching_metrics(
    *,
    p_left: np.ndarray,
    p_right: np.ndarray,
    py_left: np.ndarray,
    py_right: np.ndarray,
    p_left_jump: complex,
    p_right_jump: complex,
    py_left_jump: complex,
    py_right_jump: complex,
    alpha: float,
) -> dict[str, float | complex]:
    """Scale-invariant branch errors after one complex least-squares fit."""

    factor = _optimal_complex_factor(p_right, p_left)
    p_denominator = max(float(np.linalg.norm(p_left)), 1.0e-300)
    py_denominator = max(
        float(np.linalg.norm(py_left)),
        abs(float(alpha)) * p_denominator,
        1.0e-300,
    )
    p_jump_denominator = max(abs(p_left_jump), abs(factor * p_right_jump), 1.0e-300)
    py_jump_denominator = max(
        abs(py_left_jump),
        abs(factor * py_right_jump),
        abs(float(alpha)) * abs(p_left_jump),
        1.0e-300,
    )
    return {
        "matching_factor": factor,
        "relative_p_window_error": float(
            np.linalg.norm(p_left - factor * p_right) / p_denominator
        ),
        "relative_py_window_error": float(
            np.linalg.norm(py_left - factor * py_right) / py_denominator
        ),
        "relative_p_jump": float(
            abs(p_left_jump - factor * p_right_jump) / p_jump_denominator
        ),
        "relative_py_jump": float(
            abs(py_left_jump - factor * py_right_jump) / py_jump_denominator
        ),
    }


def _branch_ode_residuals(
    solver: Mstab17SupersonicSolver,
    *,
    y: np.ndarray,
    p: np.ndarray,
    gamma: np.ndarray,
    cr: float,
    ci: float,
) -> tuple[float, float]:
    """Return pressure and Riccati residuals on one smooth branch only."""

    order = np.argsort(y)
    y = np.asarray(y, dtype=float)[order]
    p = np.asarray(p, dtype=complex)[order]
    gamma = np.asarray(gamma, dtype=complex)[order]
    y, unique = np.unique(y, return_index=True)
    p = p[unique]
    gamma = gamma[unique]
    if len(y) < 8:
        return float("inf"), float("inf")

    p_scale = max(float(np.max(np.abs(p))), 1.0e-300)
    p = p / p_scale
    core = np.abs(p) >= 1.0e-6
    core[:2] = False
    core[-2:] = False
    if int(core.sum()) < 8:
        core = np.ones(len(y), dtype=bool)
        core[:2] = False
        core[-2:] = False

    p_y = np.gradient(p, y, edge_order=2)
    p_yy = np.gradient(p_y, y, edge_order=2)
    gamma_y = np.gradient(gamma, y, edge_order=2)
    c = complex(cr, ci)
    velocity = solver.base_velocity(y)
    velocity_y = solver.base_velocity_derivative(y)
    p_term = -2.0 * velocity_y / (velocity - c)
    r_term = 1.0 - float(solver.Mach) ** 2 * (velocity - c) ** 2
    pressure_terms = (
        p_yy,
        p_term * p_y,
        -(float(solver.alpha) ** 2) * r_term * p,
    )
    pressure_residual = sum(pressure_terms)
    pressure_denominator = max(
        sum(float(np.linalg.norm(term[core])) for term in pressure_terms),
        1.0e-300,
    )
    gamma_rhs = -(gamma**2) - p_term * gamma + (float(solver.alpha) ** 2) * r_term
    riccati_denominator = max(
        float(np.linalg.norm(gamma_y[core])) + float(np.linalg.norm(gamma_rhs[core])),
        1.0e-300,
    )
    return (
        float(np.linalg.norm(pressure_residual[core]) / pressure_denominator),
        float(np.linalg.norm((gamma_y - gamma_rhs)[core]) / riccati_denominator),
    )


def reconstruct_from_solver(
    solver: Mstab17SupersonicSolver,
    *,
    cr: float,
    ci: float,
    ln_p_start_right: float,
    assembly_y: str = "zero",
) -> dict[str, Any]:
    """Reconstruct the four fields without changing the historical default.

    ``assembly_y='zero'`` reproduces the frozen reconstruction.  The explicit
    ``'matching_y'`` option is available for diagnostics but is never selected
    silently.
    """

    if assembly_y not in ASSEMBLY_CHOICES:
        raise ValueError(f"assembly_y must be one of {ASSEMBLY_CHOICES}, got {assembly_y!r}.")
    sol_left, _, sol_right_full, y_limit = (
        solver.get_trajectories(
            cr,
            ci,
            ln_p_start_right=ln_p_start_right,
        )
    )

    if not (
        sol_left.success
        and sol_right_full.success
    ):
        raise RuntimeError(
            "Trajectory failure during modal reconstruction."
        )

    y_left = np.asarray(
        sol_left.t,
        dtype=float,
    )

    y_right = np.asarray(
        sol_right_full.t,
        dtype=float,
    )

    (
        k_left,
        q_left,
        ln_p_left,
        phi_left,
    ) = np.asarray(sol_left.y, dtype=float)

    (
        k_right,
        q_right,
        ln_p_right,
        phi_right,
    ) = np.asarray(
        sol_right_full.y,
        dtype=float,
    )

    abs_p_left = np.exp(ln_p_left)
    abs_p_right = np.exp(ln_p_right)

    assembly_value = 0.0 if assembly_y == "zero" else float(solver.match_y)
    phi_left_zero = solver._interp_component(
        assembly_value,
        sol_left,
        3,
    )

    phi_right_zero = solver._interp_component(
        assembly_value,
        sol_right_full,
        3,
    )

    phase_shift = float(phi_left_zero - phi_right_zero)
    amplitude_shift = 0.0
    if assembly_y == "matching_y":
        amplitude_shift = float(
            solver._interp_component(assembly_value, sol_left, 2)
            - solver._interp_component(assembly_value, sol_right_full, 2)
        )

    p_left = abs_p_left * np.exp(
        1j * phi_left
    )

    p_right = np.exp(ln_p_right + amplitude_shift) * np.exp(
        1j * (phi_right + phase_shift)
    )

    gamma_left = k_left + 1j * q_left
    gamma_right = k_right + 1j * q_right

    left_mask = y_left < assembly_value
    right_mask = y_right >= assembly_value

    y = np.concatenate(
        [
            y_left[left_mask],
            y_right[right_mask][::-1],
        ]
    )

    p = np.concatenate(
        [
            p_left[left_mask],
            p_right[right_mask][::-1],
        ]
    )

    gamma = np.concatenate(
        [
            gamma_left[left_mask],
            gamma_right[right_mask][::-1],
        ]
    )

    order = np.argsort(y)
    y = y[order]
    p = p[order]
    gamma = gamma[order]

    unique_y, unique_indices = np.unique(
        y,
        return_index=True,
    )

    y = unique_y
    p = p[unique_indices]
    gamma = gamma[unique_indices]

    base_velocity = solver.base_velocity(y)
    base_derivative = (
        solver.base_velocity_derivative(y)
    )

    c = complex(cr, ci)
    i_alpha = 1j * float(solver.alpha)

    p_y = gamma * p

    v = -p_y / (
        i_alpha * (base_velocity - c)
    )

    u = -(
        base_derivative * v
        + i_alpha * p
    ) / (
        i_alpha * (base_velocity - c)
    )

    rho = float(solver.Mach) ** 2 * p

    normalized = normalize_fields(
        y=y,
        u=u,
        v=v,
        p=p,
        rho=rho,
    )

    fields = {
        "y": y,
        **normalized,
        "gamma": gamma,
        "y_limit": float(y_limit),
        "ln_p_start_right": float(
            ln_p_start_right
        ),
        "phase_shift": phase_shift,
        "assembly_y": assembly_y,
        "assembly_value": assembly_value,
    }

    for name in FIELD_NAMES:
        values = np.asarray(fields[name])

        if not (
            np.isfinite(values.real).all()
            and np.isfinite(values.imag).all()
        ):
            raise ValueError(
                f"Non-finite reconstructed field: {name}"
            )

    return fields


def compute_modal_matching_diagnostics(
    solver: Mstab17SupersonicSolver,
    *,
    cr: float,
    ci: float,
    assembly_y: str = "zero",
    window_half_width: float = 0.25,
    nondegenerate_min_peak: float = 1.0e-12,
) -> dict[str, Any]:
    """Reconstruct and diagnose a mode with gauge-invariant relative errors.

    The left and right branches are first assigned the exact unbounded relative
    log-amplitude.  A single complex least-squares factor is then computed on
    their overlap window.  All reported matching errors and ODE residuals are
    invariant under a common non-zero complex scaling of the mode.

    No physical acceptance threshold is applied here.  Callers must provide
    configurable thresholds appropriate to their numerical campaign.
    """

    if assembly_y not in ASSEMBLY_CHOICES:
        raise ValueError(f"assembly_y must be one of {ASSEMBLY_CHOICES}, got {assembly_y!r}.")
    if window_half_width <= 0.0:
        raise ValueError("window_half_width must be positive.")
    if nondegenerate_min_peak <= 0.0:
        raise ValueError("nondegenerate_min_peak must be positive.")

    ln_exact = exact_right_log_amplitude(solver, cr=cr, ci=ci)
    sol_left, _, sol_right, _ = solver.get_trajectories(
        cr,
        ci,
        ln_p_start_right=ln_exact,
    )
    if not (sol_left.success and sol_right.success):
        raise RuntimeError("Trajectory failure during modal matching diagnostics.")

    assembly_value = 0.0 if assembly_y == "zero" else float(solver.match_y)
    overlap_min = max(float(np.min(sol_left.t)), float(np.min(sol_right.t)))
    overlap_max = min(float(np.max(sol_left.t)), float(np.max(sol_right.t)))
    if assembly_y == "zero":
        window_min = assembly_value
        window_max = min(overlap_max, assembly_value + window_half_width)
    else:
        window_min = max(overlap_min, assembly_value - window_half_width)
        window_max = assembly_value
    if not window_min < window_max:
        raise ValueError("Left and right trajectories have no diagnostic overlap window.")

    window_y = np.linspace(window_min, window_max, 129)
    left_state = _interp_components(sol_left, window_y)
    right_state = _interp_components(sol_right, window_y)
    p_left = np.exp(left_state[2] + 1j * left_state[3])
    p_right = np.exp(right_state[2] + 1j * right_state[3])
    gamma_left = left_state[0] + 1j * left_state[1]
    gamma_right = right_state[0] + 1j * right_state[1]

    jump_y = np.asarray([assembly_value], dtype=float)
    left_jump_state = _interp_components(sol_left, jump_y)
    right_jump_state = _interp_components(sol_right, jump_y)
    p_left_jump = complex(np.exp(left_jump_state[2][0] + 1j * left_jump_state[3][0]))
    p_right_jump = complex(np.exp(right_jump_state[2][0] + 1j * right_jump_state[3][0]))
    gamma_left_jump = complex(left_jump_state[0][0], left_jump_state[1][0])
    gamma_right_jump = complex(right_jump_state[0][0], right_jump_state[1][0])
    matching = _relative_matching_metrics(
        p_left=p_left,
        p_right=p_right,
        py_left=gamma_left * p_left,
        py_right=gamma_right * p_right,
        p_left_jump=p_left_jump,
        p_right_jump=p_right_jump,
        py_left_jump=gamma_left_jump * p_left_jump,
        py_right_jump=gamma_right_jump * p_right_jump,
        alpha=float(solver.alpha),
    )

    left_values = np.asarray(sol_left.y, dtype=float)
    right_values = np.asarray(sol_right.y, dtype=float)
    p_left_full = np.exp(left_values[2] + 1j * left_values[3])
    p_right_full = np.exp(right_values[2] + 1j * right_values[3])
    gamma_left_full = left_values[0] + 1j * left_values[1]
    gamma_right_full = right_values[0] + 1j * right_values[1]
    pressure_left, riccati_left = _branch_ode_residuals(
        solver,
        y=np.asarray(sol_left.t),
        p=p_left_full,
        gamma=gamma_left_full,
        cr=cr,
        ci=ci,
    )
    pressure_right, riccati_right = _branch_ode_residuals(
        solver,
        y=np.asarray(sol_right.t),
        p=p_right_full,
        gamma=gamma_right_full,
        cr=cr,
        ci=ci,
    )

    fields = reconstruct_from_solver(
        solver,
        cr=cr,
        ci=ci,
        ln_p_start_right=ln_exact,
        assembly_y=assembly_y,
    )
    finite_fields = all(
        np.isfinite(np.asarray(fields[name]).real).all()
        and np.isfinite(np.asarray(fields[name]).imag).all()
        for name in FIELD_NAMES
    )
    peaks = {name: float(np.max(np.abs(np.asarray(fields[name])))) for name in FIELD_NAMES}
    nondegenerate_fields = bool(
        len(np.asarray(fields["y"])) >= 8
        and all(np.isfinite(value) and value >= nondegenerate_min_peak for value in peaks.values())
    )
    factor = complex(matching.pop("matching_factor"))
    return {
        "ln_p_start_right_exact": float(ln_exact),
        "assembly_y": assembly_y,
        "assembly_value": assembly_value,
        "window_y_min": float(window_min),
        "window_y_max": float(window_max),
        "matching_factor_real": float(factor.real),
        "matching_factor_imag": float(factor.imag),
        **matching,
        "finite_fields": bool(finite_fields),
        "nondegenerate_fields": nondegenerate_fields,
        "field_peaks": peaks,
        "pressure_ode_residual_left": pressure_left,
        "pressure_ode_residual_right": pressure_right,
        "riccati_residual_left": riccati_left,
        "riccati_residual_right": riccati_right,
        "fields": fields if finite_fields else None,
    }
