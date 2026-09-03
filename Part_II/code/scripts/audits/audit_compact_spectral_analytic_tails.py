#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import least_squares

import scripts.evaluation.test_kappa_q_modulus_reconstruction as base


@dataclass(frozen=True)
class SpectralCase:
    case_id: str
    family: str
    spectral_extent: float
    matching_y: float
    max_step: float
    rtol: float
    atol: float


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue a supersonic Riccati eigenvalue on a compact spectral box, "
            "audit spectral convergence at the target alpha, and reconstruct the "
            "mode with analytic far-field tails."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--case-id", type=str, required=True)
    parser.add_argument("--Mach", type=float, required=True)

    parser.add_argument("--start-alpha", type=float, required=True)
    parser.add_argument("--start-cr", type=float, required=True)
    parser.add_argument("--start-ci", type=float, required=True)
    parser.add_argument("--target-alpha", type=float, required=True)

    parser.add_argument("--continuation-step", type=float, default=0.01)
    parser.add_argument("--continuation-min-step", type=float, default=2.5e-4)
    parser.add_argument("--continuation-max-step", type=float, default=0.02)
    parser.add_argument("--continuation-growth", type=float, default=1.25)
    parser.add_argument("--continuation-shrink", type=float, default=0.5)
    parser.add_argument("--continuation-max-attempts", type=int, default=200)

    parser.add_argument("--continuation-extent", type=float, default=20.0)
    parser.add_argument("--continuation-matching-y", type=float, default=1.0)
    parser.add_argument("--continuation-max-step-ode", type=float, default=0.25)
    parser.add_argument("--continuation-rtol", type=float, default=1.0e-10)
    parser.add_argument("--continuation-atol", type=float, default=1.0e-12)

    parser.add_argument(
        "--spectral-extents",
        type=parse_float_list,
        default=parse_float_list("12,16,20,30,40"),
    )
    parser.add_argument(
        "--matching-values",
        type=parse_float_list,
        default=parse_float_list("0,0.5,1,1.5,2"),
    )
    parser.add_argument(
        "--max-step-values",
        type=parse_float_list,
        default=parse_float_list("0.1,0.25,0.5,1.0"),
    )
    parser.add_argument(
        "--rtol-values",
        type=parse_float_list,
        default=parse_float_list("1e-11,1e-10,1e-9,1e-8"),
    )
    parser.add_argument("--suite", choices=("smoke", "quick", "full"), default="quick")

    parser.add_argument("--baseline-extent", type=float, default=40.0)
    parser.add_argument("--baseline-matching-y", type=float, default=1.0)
    parser.add_argument("--baseline-max-step", type=float, default=0.1)
    parser.add_argument("--baseline-rtol", type=float, default=1.0e-11)
    parser.add_argument("--baseline-atol", type=float, default=1.0e-13)

    parser.add_argument("--nominal-extent", type=float, default=20.0)
    parser.add_argument("--nominal-matching-y", type=float, default=1.0)
    parser.add_argument("--nominal-max-step", type=float, default=0.25)
    parser.add_argument("--nominal-rtol", type=float, default=1.0e-10)
    parser.add_argument("--nominal-atol", type=float, default=1.0e-12)

    parser.add_argument("--method", choices=("DOP853", "RK45"), default="DOP853")
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--diff-step", type=float, default=1.0e-5)
    parser.add_argument("--root-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--ci-lower", type=float, default=1.0e-12)
    parser.add_argument("--ci-upper", type=float, default=0.15)
    parser.add_argument("--local-cr-half-width", type=float, default=0.06)
    parser.add_argument(
        "--local-ci-factor",
        type=float,
        default=100.0,
        help="Local optimizer bound: ci_seed/factor <= ci <= ci_seed*factor.",
    )

    parser.add_argument("--modal-numerical-extent", type=float, default=20.0)
    parser.add_argument("--modal-dy", type=float, default=0.025)
    parser.add_argument("--tail-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--analytic-tail-points", type=int, default=300)
    parser.add_argument("--core-y", type=float, default=20.0)

    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.Mach <= 0.0 or args.start_alpha <= 0.0 or args.target_alpha <= 0.0:
        raise ValueError("Mach and alpha values must be positive.")
    if args.start_ci <= 0.0:
        raise ValueError("--start-ci must be strictly positive.")
    if args.target_alpha <= args.start_alpha:
        raise ValueError("This script currently expects target_alpha > start_alpha.")
    if not (0.0 < args.continuation_min_step <= args.continuation_step):
        raise ValueError("Invalid continuation step settings.")
    if args.continuation_max_step < args.continuation_step:
        raise ValueError("--continuation-max-step must be >= initial step.")
    if not (0.0 < args.continuation_shrink < 1.0):
        raise ValueError("--continuation-shrink must lie in (0,1).")
    if args.continuation_growth <= 1.0:
        raise ValueError("--continuation-growth must be > 1.")
    if args.root_tolerance <= 0.0:
        raise ValueError("--root-tolerance must be positive.")
    if args.tail_tolerance <= 0.0 or args.tail_tolerance >= 1.0:
        raise ValueError("--tail-tolerance must lie in (0,1).")
    if args.local_ci_factor <= 1.0:
        raise ValueError("--local-ci-factor must be > 1.")
    if args.modal_numerical_extent < args.core_y:
        raise ValueError("--modal-numerical-extent must be >= --core-y.")


def flow(y: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.tanh(y)
    up = 1.0 - u * u
    return u, up


def gamma_rhs(
    y: float,
    state: np.ndarray,
    *,
    Mach: float,
    alpha: float,
    c: complex,
) -> np.ndarray:
    gamma = complex(float(state[0]), float(state[1]))
    u, up = flow(y)
    p_coeff = -2.0 * up / (u - c)
    r_coeff = 1.0 - Mach * Mach * (u - c) ** 2
    derivative = -(gamma * gamma) - p_coeff * gamma + alpha * alpha * r_coeff
    return np.asarray([derivative.real, derivative.imag], dtype=float)


def logmode_rhs(
    y: float,
    state: np.ndarray,
    *,
    Mach: float,
    alpha: float,
    c: complex,
) -> np.ndarray:
    gamma_derivative = gamma_rhs(
        y,
        state[:2],
        Mach=Mach,
        alpha=alpha,
        c=c,
    )
    return np.asarray(
        [gamma_derivative[0], gamma_derivative[1], float(state[0])],
        dtype=float,
    )


def endpoint_gamma(
    *,
    side: str,
    Mach: float,
    alpha: float,
    c: complex,
    spectral_extent: float,
    matching_y: float,
    max_step: float,
    rtol: float,
    atol: float,
    method: str,
) -> dict[str, Any]:
    if side == "left":
        start = -spectral_extent
    elif side == "right":
        start = spectral_extent
    else:
        raise ValueError(f"Unknown side: {side}")

    gamma0 = base.asymptotic_gamma(
        side=side,
        Mach=Mach,
        alpha=alpha,
        c=c,
    )
    solution = solve_ivp(
        lambda y, state: gamma_rhs(
            y,
            state,
            Mach=Mach,
            alpha=alpha,
            c=c,
        ),
        (start, matching_y),
        np.asarray([gamma0.real, gamma0.imag], dtype=float),
        method=method,
        t_eval=np.asarray([matching_y], dtype=float),
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"{side} integration failed: {solution.message}")
    endpoint = np.asarray(solution.y[:, -1], dtype=float)
    if endpoint.shape != (2,) or not np.all(np.isfinite(endpoint)):
        raise RuntimeError(f"{side} integration returned an invalid endpoint.")
    return {
        "kappa": float(endpoint[0]),
        "q": float(endpoint[1]),
        "nfev": int(solution.nfev),
    }


class RootObjective:
    def __init__(
        self,
        *,
        Mach: float,
        alpha: float,
        spectral_extent: float,
        matching_y: float,
        max_step: float,
        rtol: float,
        atol: float,
        method: str,
        history: list[dict[str, Any]],
        label: str,
    ) -> None:
        self.Mach = Mach
        self.alpha = alpha
        self.spectral_extent = spectral_extent
        self.matching_y = matching_y
        self.max_step = max_step
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.history = history
        self.label = label
        self.count = 0

    @staticmethod
    def unpack(vector: np.ndarray) -> tuple[float, float]:
        cr = float(vector[0])
        ci = float(math.exp(float(vector[1])))
        return cr, ci

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        self.count += 1
        cr, ci = self.unpack(vector)
        c = complex(cr, ci)
        record: dict[str, Any] = {
            "evaluation": self.count,
            "cr": cr,
            "ci": ci,
            "log_ci": float(vector[1]),
        }
        try:
            left = endpoint_gamma(
                side="left",
                Mach=self.Mach,
                alpha=self.alpha,
                c=c,
                spectral_extent=self.spectral_extent,
                matching_y=self.matching_y,
                max_step=self.max_step,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
            )
            right = endpoint_gamma(
                side="right",
                Mach=self.Mach,
                alpha=self.alpha,
                c=c,
                spectral_extent=self.spectral_extent,
                matching_y=self.matching_y,
                max_step=self.max_step,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
            )
            delta_kappa = left["kappa"] - right["kappa"]
            delta_q = left["q"] - right["q"]
            norm = math.hypot(delta_kappa, delta_q)
            residual = np.asarray([delta_kappa, delta_q], dtype=float)
            record.update(
                {
                    "success": True,
                    "delta_kappa": delta_kappa,
                    "delta_q": delta_q,
                    "residual_norm": norm,
                    "left_nfev": left["nfev"],
                    "right_nfev": right["nfev"],
                    "error": "",
                }
            )
            print(
                f"[{self.label} {self.count:03d}] "
                f"cr={cr:.15g} ci={ci:.8e} "
                f"dκ={delta_kappa:+.4e} dq={delta_q:+.4e} "
                f"||F||={norm:.4e}",
                flush=True,
            )
        except Exception as exc:
            penalty = 1.0e3
            residual = np.asarray([penalty, penalty], dtype=float)
            record.update(
                {
                    "success": False,
                    "delta_kappa": penalty,
                    "delta_q": penalty,
                    "residual_norm": math.sqrt(2.0) * penalty,
                    "left_nfev": 0,
                    "right_nfev": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[{self.label} {self.count:03d}] FAILED: {exc}", flush=True)

        self.history.append(record)
        return residual


def solve_root(
    *,
    args: argparse.Namespace,
    alpha: float,
    seed_cr: float,
    seed_ci: float,
    spectral_extent: float,
    matching_y: float,
    max_step: float,
    rtol: float,
    atol: float,
    label: str,
    cr_half_width: float | None = None,
    ci_factor: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if seed_ci <= 0.0:
        raise ValueError("Root seed ci must be positive.")

    half_width = (
        args.local_cr_half_width if cr_half_width is None else cr_half_width
    )
    factor = args.local_ci_factor if ci_factor is None else ci_factor

    cr_lower = max(-0.2, seed_cr - half_width)
    cr_upper = min(1.2, seed_cr + half_width)
    ci_lower = max(args.ci_lower, seed_ci / factor)
    ci_upper = min(args.ci_upper, seed_ci * factor)
    if ci_upper <= ci_lower:
        ci_upper = min(args.ci_upper, max(ci_lower * 10.0, ci_lower + 1.0e-12))
    if ci_upper <= ci_lower:
        raise RuntimeError("Could not construct valid local ci bounds.")

    x0 = np.asarray(
        [
            np.clip(seed_cr, cr_lower + 1.0e-12, cr_upper - 1.0e-12),
            math.log(np.clip(seed_ci, ci_lower * (1.0 + 1.0e-12), ci_upper / (1.0 + 1.0e-12))),
        ],
        dtype=float,
    )
    bounds = (
        np.asarray([cr_lower, math.log(ci_lower)], dtype=float),
        np.asarray([cr_upper, math.log(ci_upper)], dtype=float),
    )

    history: list[dict[str, Any]] = []
    objective = RootObjective(
        Mach=args.Mach,
        alpha=alpha,
        spectral_extent=spectral_extent,
        matching_y=matching_y,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        method=args.method,
        history=history,
        label=label,
    )

    result = least_squares(
        objective,
        x0=x0,
        bounds=bounds,
        method="trf",
        jac="3-point",
        diff_step=args.diff_step,
        x_scale=np.asarray([max(0.02, half_width / 2.0), 1.0], dtype=float),
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=args.max_nfev,
        verbose=0,
    )

    final_residual = objective(result.x)
    final_cr, final_ci = objective.unpack(result.x)
    final_norm = float(np.linalg.norm(final_residual))
    active_lower_ci = abs(float(result.x[1]) - bounds[0][1]) < 1.0e-6
    active_upper_ci = abs(float(result.x[1]) - bounds[1][1]) < 1.0e-6
    accepted = bool(
        np.all(np.isfinite(final_residual))
        and final_norm <= args.root_tolerance
        and not active_lower_ci
        and not active_upper_ci
    )

    summary = {
        "alpha": alpha,
        "cr": final_cr,
        "ci": final_ci,
        "omega_i": alpha * final_ci,
        "delta_kappa": float(final_residual[0]),
        "delta_q": float(final_residual[1]),
        "residual_norm": final_norm,
        "accepted": accepted,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "optimizer_njev": int(result.njev) if result.njev is not None else None,
        "cr_lower": cr_lower,
        "cr_upper": cr_upper,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "active_lower_ci": active_lower_ci,
        "active_upper_ci": active_upper_ci,
    }
    return summary, pd.DataFrame(history)


def secant_predictor(
    accepted_rows: list[dict[str, Any]],
    target_alpha: float,
) -> tuple[float, float]:
    last = accepted_rows[-1]
    if len(accepted_rows) < 2:
        return float(last["cr"]), float(last["ci"])

    previous = accepted_rows[-2]
    denominator = float(last["alpha"] - previous["alpha"])
    if abs(denominator) < 1.0e-15:
        return float(last["cr"]), float(last["ci"])

    ratio = (target_alpha - float(last["alpha"])) / denominator
    predicted_cr = float(last["cr"]) + ratio * (
        float(last["cr"]) - float(previous["cr"])
    )
    predicted_log_ci = math.log(float(last["ci"])) + ratio * (
        math.log(float(last["ci"])) - math.log(float(previous["ci"]))
    )
    predicted_ci = math.exp(
        float(np.clip(predicted_log_ci, math.log(1.0e-12), math.log(0.15)))
    )
    return predicted_cr, predicted_ci


def run_continuation(
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    print("=== Compact-box continuation ===")
    accepted_rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []

    start_root, start_history = solve_root(
        args=args,
        alpha=args.start_alpha,
        seed_cr=args.start_cr,
        seed_ci=args.start_ci,
        spectral_extent=args.continuation_extent,
        matching_y=args.continuation_matching_y,
        max_step=args.continuation_max_step_ode,
        rtol=args.continuation_rtol,
        atol=args.continuation_atol,
        label=f"continuation_a{args.start_alpha:.6f}",
        cr_half_width=0.03,
        ci_factor=10.0,
    )
    start_history.to_csv(output_dir / "continuation_start_history.csv", index=False)
    if not start_root["accepted"]:
        raise RuntimeError(
            "The validated starting point did not pass the compact-box root test: "
            f"||F||={start_root['residual_norm']:.6e}."
        )
    start_root.update(
        {
            "step_used": 0.0,
            "attempt": 0,
            "status": "accepted_start",
        }
    )
    accepted_rows.append(start_root)
    attempts.append(dict(start_root))

    current_alpha = args.start_alpha
    step = args.continuation_step
    attempt_number = 0

    while current_alpha < args.target_alpha - 1.0e-14:
        attempt_number += 1
        if attempt_number > args.continuation_max_attempts:
            raise RuntimeError("Maximum continuation attempts exceeded.")

        trial_alpha = min(current_alpha + step, args.target_alpha)
        predicted_cr, predicted_ci = secant_predictor(
            accepted_rows,
            trial_alpha,
        )

        print(
            "\n--- continuation trial ---\n"
            f"current alpha : {current_alpha:.12g}\n"
            f"trial alpha   : {trial_alpha:.12g}\n"
            f"step          : {step:.12g}\n"
            f"predictor c   : {predicted_cr:.15g} + {predicted_ci:.8e} i",
            flush=True,
        )

        root, history = solve_root(
            args=args,
            alpha=trial_alpha,
            seed_cr=predicted_cr,
            seed_ci=predicted_ci,
            spectral_extent=args.continuation_extent,
            matching_y=args.continuation_matching_y,
            max_step=args.continuation_max_step_ode,
            rtol=args.continuation_rtol,
            atol=args.continuation_atol,
            label=f"continuation_a{trial_alpha:.6f}",
        )
        history.to_csv(
            output_dir / f"continuation_attempt_{attempt_number:03d}_a{trial_alpha:.8f}.csv",
            index=False,
        )
        root.update(
            {
                "step_used": step,
                "attempt": attempt_number,
            }
        )

        last = accepted_rows[-1]
        cr_jump = abs(root["cr"] - last["cr"])
        log_ci_jump = abs(math.log(root["ci"]) - math.log(last["ci"]))
        branch_plausible = bool(
            cr_jump <= max(0.08, 8.0 * step)
            and log_ci_jump <= math.log(1.0e4)
        )
        root["cr_jump_from_previous"] = cr_jump
        root["log_ci_jump_from_previous"] = log_ci_jump
        root["branch_plausible"] = branch_plausible

        if root["accepted"] and branch_plausible:
            root["status"] = "accepted"
            accepted_rows.append(root)
            attempts.append(dict(root))
            current_alpha = trial_alpha
            step = min(
                args.continuation_max_step,
                step * args.continuation_growth,
            )
            print(
                f"ACCEPTED alpha={trial_alpha:.8f}: "
                f"c={root['cr']:.15g}+{root['ci']:.8e}i, "
                f"||F||={root['residual_norm']:.3e}",
                flush=True,
            )
        else:
            root["status"] = "rejected"
            attempts.append(dict(root))
            step *= args.continuation_shrink
            print(
                f"REJECTED alpha={trial_alpha:.8f}: "
                f"||F||={root['residual_norm']:.3e}, "
                f"branch_plausible={branch_plausible}; "
                f"new step={step:.6g}",
                flush=True,
            )
            if step < args.continuation_min_step:
                pd.DataFrame(attempts).to_csv(
                    output_dir / "continuation_all_attempts.csv",
                    index=False,
                )
                pd.DataFrame(accepted_rows).to_csv(
                    output_dir / "continuation_accepted.csv",
                    index=False,
                )
                raise RuntimeError(
                    "Continuation failed before the target: the required step "
                    f"fell below {args.continuation_min_step}."
                )

    attempts_frame = pd.DataFrame(attempts)
    accepted_frame = pd.DataFrame(accepted_rows)
    attempts_frame.to_csv(output_dir / "continuation_all_attempts.csv", index=False)
    accepted_frame.to_csv(output_dir / "continuation_accepted.csv", index=False)

    figure = plt.figure(figsize=(8.5, 5.5))
    plt.plot(
        accepted_frame["alpha"],
        accepted_frame["cr"],
        marker="o",
        label=r"$c_r$",
    )
    plt.plot(
        accepted_frame["alpha"],
        accepted_frame["ci"],
        marker="o",
        label=r"$c_i$",
    )
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Eigenvalue component")
    plt.title(f"Compact-box continuation, M={args.Mach:g}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "continuation_cr_ci.png", dpi=220)
    figure.savefig(output_dir / "continuation_cr_ci.pdf")
    plt.close(figure)

    figure = plt.figure(figsize=(8.5, 5.5))
    plt.semilogy(
        accepted_frame["alpha"],
        accepted_frame["ci"],
        marker="o",
    )
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$c_i$")
    plt.title(f"Growth component near neutrality, M={args.Mach:g}")
    plt.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "continuation_ci_log.png", dpi=220)
    figure.savefig(output_dir / "continuation_ci_log.pdf")
    plt.close(figure)

    return accepted_frame


def integrate_logmode_branch(
    *,
    side: str,
    args: argparse.Namespace,
    alpha: float,
    c: complex,
    numerical_extent: float,
    matching_y: float,
    max_step: float,
    rtol: float,
    atol: float,
) -> Any:
    start = -numerical_extent if side == "left" else numerical_extent
    gamma0 = base.asymptotic_gamma(
        side=side,
        Mach=args.Mach,
        alpha=alpha,
        c=c,
    )
    solution = solve_ivp(
        lambda y, state: logmode_rhs(
            y,
            state,
            Mach=args.Mach,
            alpha=alpha,
            c=c,
        ),
        (start, matching_y),
        np.asarray([gamma0.real, gamma0.imag, 0.0], dtype=float),
        method=args.method,
        dense_output=True,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"{side} mode integration failed: {solution.message}")
    return solution


def numerical_mode_and_analytic_tails(
    *,
    args: argparse.Namespace,
    case: SpectralCase,
    alpha: float,
    c: complex,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    extent = max(args.modal_numerical_extent, case.spectral_extent)
    left = integrate_logmode_branch(
        side="left",
        args=args,
        alpha=alpha,
        c=c,
        numerical_extent=extent,
        matching_y=case.matching_y,
        max_step=case.max_step,
        rtol=case.rtol,
        atol=case.atol,
    )
    right = integrate_logmode_branch(
        side="right",
        args=args,
        alpha=alpha,
        c=c,
        numerical_extent=extent,
        matching_y=case.matching_y,
        max_step=case.max_step,
        rtol=case.rtol,
        atol=case.atol,
    )

    left_match = left.sol(case.matching_y)
    right_match = right.sol(case.matching_y)
    right_ell_shift = float(left_match[2] - right_match[2])

    count = max(4, int(math.ceil((2.0 * extent) / args.modal_dy)))
    y = np.linspace(-extent, extent, count + 1)
    if not np.any(np.isclose(y, case.matching_y, atol=1.0e-13)):
        y = np.sort(np.unique(np.append(y, case.matching_y)))

    left_mask = y <= case.matching_y
    right_mask = ~left_mask
    values = np.empty((3, y.size), dtype=float)
    values[:, left_mask] = left.sol(y[left_mask])
    values[:, right_mask] = right.sol(y[right_mask])
    values[2, right_mask] += right_ell_shift

    ell_max = float(
        max(
            np.max(left.y[2]),
            np.max(right.y[2] + right_ell_shift),
            np.max(values[2]),
        )
    )
    kappa = values[0]
    q = values[1]
    ell = values[2]
    modulus = np.exp(np.clip(ell - ell_max, -745.0, 0.0))

    phase_integral = cumulative_trapezoid(q, y, initial=0.0)
    match_index = int(np.argmin(np.abs(y - case.matching_y)))
    phase = phase_integral - phase_integral[match_index]
    pressure = modulus * np.exp(1j * phase)

    numerical = pd.DataFrame(
        {
            "region": "numerical",
            "y": y,
            "kappa": kappa,
            "q": q,
            "log_modulus_normalized": ell - ell_max,
            "modulus": modulus,
            "phase": phase,
            "p_real": pressure.real,
            "p_imag": pressure.imag,
        }
    )

    gamma_left = base.asymptotic_gamma(
        side="left",
        Mach=args.Mach,
        alpha=alpha,
        c=c,
    )
    gamma_right = base.asymptotic_gamma(
        side="right",
        Mach=args.Mach,
        alpha=alpha,
        c=c,
    )
    mu_left = float(gamma_left.real)
    mu_right = float(-gamma_right.real)

    if mu_left <= 0.0 or mu_right <= 0.0:
        raise RuntimeError(
            "The selected asymptotic roots are not decaying on both sides: "
            f"mu_left={mu_left}, mu_right={mu_right}."
        )

    left_interface_ell = float(numerical.iloc[0]["log_modulus_normalized"])
    right_interface_ell = float(numerical.iloc[-1]["log_modulus_normalized"])
    left_interface_phase = float(numerical.iloc[0]["phase"])
    right_interface_phase = float(numerical.iloc[-1]["phase"])
    target_log = math.log(args.tail_tolerance)

    left_distance = max(0.0, (left_interface_ell - target_log) / mu_left)
    right_distance = max(0.0, (right_interface_ell - target_log) / mu_right)
    far_left = -extent - left_distance
    far_right = extent + right_distance

    n_tail = max(2, args.analytic_tail_points)
    y_left_tail = np.linspace(far_left, -extent, n_tail, endpoint=False)
    y_right_tail = np.linspace(extent, far_right, n_tail + 1)[1:]

    left_log = (
        left_interface_ell
        + gamma_left.real * (y_left_tail + extent)
    )
    left_phase = (
        left_interface_phase
        + gamma_left.imag * (y_left_tail + extent)
    )
    right_log = (
        right_interface_ell
        + gamma_right.real * (y_right_tail - extent)
    )
    right_phase = (
        right_interface_phase
        + gamma_right.imag * (y_right_tail - extent)
    )

    left_modulus = np.exp(np.clip(left_log, -745.0, 0.0))
    right_modulus = np.exp(np.clip(right_log, -745.0, 0.0))
    left_pressure = left_modulus * np.exp(1j * left_phase)
    right_pressure = right_modulus * np.exp(1j * right_phase)

    left_tail = pd.DataFrame(
        {
            "region": "analytic_left",
            "y": y_left_tail,
            "kappa": gamma_left.real,
            "q": gamma_left.imag,
            "log_modulus_normalized": left_log,
            "modulus": left_modulus,
            "phase": left_phase,
            "p_real": left_pressure.real,
            "p_imag": left_pressure.imag,
        }
    )
    right_tail = pd.DataFrame(
        {
            "region": "analytic_right",
            "y": y_right_tail,
            "kappa": gamma_right.real,
            "q": gamma_right.imag,
            "log_modulus_normalized": right_log,
            "modulus": right_modulus,
            "phase": right_phase,
            "p_real": right_pressure.real,
            "p_imag": right_pressure.imag,
        }
    )
    complete = pd.concat(
        [left_tail, numerical, right_tail],
        ignore_index=True,
    ).sort_values("y").reset_index(drop=True)

    diagnostics = {
        "numerical_extent": extent,
        "gamma_left_real": gamma_left.real,
        "gamma_left_imag": gamma_left.imag,
        "gamma_right_real": gamma_right.real,
        "gamma_right_imag": gamma_right.imag,
        "mu_left": mu_left,
        "mu_right": mu_right,
        "left_interface_amplitude_ratio": math.exp(max(-745.0, left_interface_ell)),
        "right_interface_amplitude_ratio": math.exp(max(-745.0, right_interface_ell)),
        "analytic_far_left": far_left,
        "analytic_far_right": far_right,
        "analytic_left_distance": left_distance,
        "analytic_right_distance": right_distance,
        "target_tail_tolerance": args.tail_tolerance,
        "left_far_amplitude_ratio": float(left_modulus[0]) if len(left_modulus) else math.exp(left_interface_ell),
        "right_far_amplitude_ratio": float(right_modulus[-1]) if len(right_modulus) else math.exp(right_interface_ell),
        "right_log_amplitude_shift": right_ell_shift,
        "delta_kappa_at_match": float(left_match[0] - right_match[0]),
        "delta_q_at_match": float(left_match[1] - right_match[1]),
        "gamma_mismatch_at_match": float(
            math.hypot(
                left_match[0] - right_match[0],
                left_match[1] - right_match[1],
            )
        ),
    }
    return numerical, complete, diagnostics


def finite_difference_residual(
    frame: pd.DataFrame,
    *,
    args: argparse.Namespace,
    alpha: float,
    c: complex,
    matching_y: float,
) -> dict[str, float]:
    y = frame["y"].to_numpy(float)
    pressure = (
        frame["p_real"].to_numpy(float)
        + 1j * frame["p_imag"].to_numpy(float)
    )
    p_y = np.gradient(pressure, y, edge_order=2)
    p_yy = np.gradient(p_y, y, edge_order=2)
    u, up = flow(y)
    residual = (
        p_yy
        - (2.0 * up / (u - c)) * p_y
        - alpha * alpha
        * (1.0 - args.Mach * args.Mach * (u - c) ** 2)
        * pressure
    )
    scale = (
        np.abs(p_yy)
        + np.abs((2.0 * up / (u - c)) * p_y)
        + np.abs(
            alpha * alpha
            * (1.0 - args.Mach * args.Mach * (u - c) ** 2)
            * pressure
        )
        + 1.0e-14
    )
    relative = np.abs(residual) / scale
    mask = (
        (np.abs(y) <= args.core_y)
        & (np.abs(y - matching_y) > 1.0)
    )
    significant = mask & (
        np.abs(pressure) >= 1.0e-3 * np.max(np.abs(pressure))
    )
    if np.count_nonzero(significant) < 10:
        significant = mask
    return {
        "fd_residual_max_core": float(np.max(relative[mask])),
        "fd_residual_rms_core": float(np.sqrt(np.mean(relative[mask] ** 2))),
        "fd_residual_max_significant": float(np.max(relative[significant])),
        "fd_residual_rms_significant": float(
            np.sqrt(np.mean(relative[significant] ** 2))
        ),
    }


def compare_modes(
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    core_y: float,
) -> dict[str, float]:
    y_base = baseline["y"].to_numpy(float)
    p_base = (
        baseline["p_real"].to_numpy(float)
        + 1j * baseline["p_imag"].to_numpy(float)
    )
    y = frame["y"].to_numpy(float)
    pressure = (
        frame["p_real"].to_numpy(float)
        + 1j * frame["p_imag"].to_numpy(float)
    )

    mask = (
        (np.abs(y_base) <= core_y)
        & (y_base >= np.min(y))
        & (y_base <= np.max(y))
    )
    interpolated = (
        np.interp(y_base, y, pressure.real)
        + 1j * np.interp(y_base, y, pressure.imag)
    )
    denominator = np.vdot(interpolated[mask], interpolated[mask])
    if abs(denominator) < 1.0e-30:
        raise RuntimeError("Mode vanishes on comparison core.")
    scale = np.vdot(interpolated[mask], p_base[mask]) / denominator
    aligned = scale * interpolated

    return {
        "mode_relative_l2_core_vs_baseline": float(
            np.linalg.norm(aligned[mask] - p_base[mask])
            / np.linalg.norm(p_base[mask])
        ),
        "envelope_relative_l2_core_vs_baseline": float(
            np.linalg.norm(np.abs(aligned[mask]) - np.abs(p_base[mask]))
            / np.linalg.norm(np.abs(p_base[mask]))
        ),
        "alignment_scale_real": float(scale.real),
        "alignment_scale_imag": float(scale.imag),
    }


def build_audit_cases(args: argparse.Namespace) -> list[SpectralCase]:
    cases = [
        SpectralCase(
            case_id="baseline_fine",
            family="baseline",
            spectral_extent=args.baseline_extent,
            matching_y=args.baseline_matching_y,
            max_step=args.baseline_max_step,
            rtol=args.baseline_rtol,
            atol=args.baseline_atol,
        ),
        SpectralCase(
            case_id="nominal",
            family="nominal",
            spectral_extent=args.nominal_extent,
            matching_y=args.nominal_matching_y,
            max_step=args.nominal_max_step,
            rtol=args.nominal_rtol,
            atol=args.nominal_atol,
        ),
    ]
    if args.suite == "smoke":
        return cases

    extents = args.spectral_extents
    matching_values = args.matching_values
    max_steps = args.max_step_values
    rtols = args.rtol_values
    if args.suite == "quick":
        extents = [12.0, 20.0, 40.0]
        matching_values = [0.0, 1.0, 2.0]
        max_steps = [0.25, 1.0]
        rtols = [1.0e-10, 1.0e-8]

    for extent in extents:
        if np.isclose(extent, args.nominal_extent):
            continue
        cases.append(
            SpectralCase(
                case_id=f"extent_{str(extent).replace('.', 'p')}",
                family="spectral_extent",
                spectral_extent=extent,
                matching_y=args.nominal_matching_y,
                max_step=args.nominal_max_step,
                rtol=args.nominal_rtol,
                atol=args.nominal_atol,
            )
        )

    for matching_y in matching_values:
        if np.isclose(matching_y, args.nominal_matching_y):
            continue
        cases.append(
            SpectralCase(
                case_id=f"match_{str(matching_y).replace('.', 'p')}",
                family="matching",
                spectral_extent=args.nominal_extent,
                matching_y=matching_y,
                max_step=args.nominal_max_step,
                rtol=args.nominal_rtol,
                atol=args.nominal_atol,
            )
        )

    for max_step in max_steps:
        if np.isclose(max_step, args.nominal_max_step):
            continue
        cases.append(
            SpectralCase(
                case_id=f"step_{str(max_step).replace('.', 'p')}",
                family="max_step",
                spectral_extent=args.nominal_extent,
                matching_y=args.nominal_matching_y,
                max_step=max_step,
                rtol=args.nominal_rtol,
                atol=args.nominal_atol,
            )
        )

    for rtol in rtols:
        if np.isclose(rtol, args.nominal_rtol, rtol=1.0e-12, atol=0.0):
            continue
        cases.append(
            SpectralCase(
                case_id=f"rtol_{rtol:.0e}".replace("-", "m"),
                family="tolerance",
                spectral_extent=args.nominal_extent,
                matching_y=args.nominal_matching_y,
                max_step=args.nominal_max_step,
                rtol=rtol,
                atol=rtol * 1.0e-2,
            )
        )

    unique: list[SpectralCase] = []
    seen: set[tuple[Any, ...]] = set()
    for case in cases:
        key = (
            case.spectral_extent,
            case.matching_y,
            case.max_step,
            case.rtol,
            case.atol,
        )
        if key not in seen:
            seen.add(key)
            unique.append(case)
    return unique


def plot_audit(summary: pd.DataFrame, output_dir: Path) -> None:
    baseline = summary.loc[summary["case_id"] == "baseline_fine"].iloc[0]
    for family, x_column in (
        ("spectral_extent", "spectral_extent"),
        ("matching", "matching_y"),
        ("max_step", "max_step"),
        ("tolerance", "rtol"),
    ):
        subset = summary[summary["family"] == family].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(x_column)

        figure = plt.figure(figsize=(8.5, 5.5))
        plt.plot(
            subset[x_column],
            np.maximum(np.abs(subset["cr"] - baseline["cr"]), 1.0e-18),
            marker="o",
            label=r"$|c_r-c_{r,\mathrm{base}}|$",
        )
        plt.plot(
            subset[x_column],
            np.maximum(np.abs(subset["ci"] - baseline["ci"]), 1.0e-18),
            marker="o",
            label=r"$|c_i-c_{i,\mathrm{base}}|$",
        )
        plt.yscale("log")
        plt.xlabel(x_column)
        plt.ylabel("Absolute eigenvalue difference")
        plt.title(f"Compact spectral audit — {family}")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"eigenvalue_{family}.png", dpi=220)
        figure.savefig(output_dir / f"eigenvalue_{family}.pdf")
        plt.close(figure)

        figure = plt.figure(figsize=(8.5, 5.5))
        plt.plot(
            subset[x_column],
            np.maximum(subset["mode_relative_l2_core_vs_baseline"], 1.0e-18),
            marker="o",
            label="complex pressure",
        )
        plt.plot(
            subset[x_column],
            np.maximum(subset["envelope_relative_l2_core_vs_baseline"], 1.0e-18),
            marker="o",
            label="envelope",
        )
        plt.yscale("log")
        plt.xlabel(x_column)
        plt.ylabel("Relative L2 difference on core")
        plt.title(f"Mode audit — {family}")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"mode_{family}.png", dpi=220)
        figure.savefig(output_dir / f"mode_{family}.pdf")
        plt.close(figure)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    validate_args(args)
    repo = args.repo.expanduser().resolve()

    if args.output_dir is None:
        output_dir = (
            repo
            / "classic_supersonic/reproducibility/results"
            / f"compact_spectral_analytic_tail_audit_{args.case_id}"
        )
    else:
        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = repo / output_dir
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Compact spectral box + analytic tails ===")
    print(f"case_id       : {args.case_id}")
    print(f"Mach          : {args.Mach}")
    print(f"start alpha   : {args.start_alpha}")
    print(f"target alpha  : {args.target_alpha}")
    print(
        f"start c       : {args.start_cr:.15g} + "
        f"{args.start_ci:.15g} i"
    )
    print(f"output        : {output_dir}")

    continuation = run_continuation(
        args=args,
        output_dir=output_dir,
    )
    target_row = continuation.iloc[-1]
    target_cr = float(target_row["cr"])
    target_ci = float(target_row["ci"])
    target_c = complex(target_cr, target_ci)

    print("\n=== Target continuation result ===")
    print(f"alpha         : {args.target_alpha}")
    print(f"cr            : {target_cr:.15g}")
    print(f"ci            : {target_ci:.15g}")
    print(f"omega_i       : {args.target_alpha * target_ci:.15g}")
    print(f"||F||         : {float(target_row['residual_norm']):.6e}")

    cases = build_audit_cases(args)
    rows: list[dict[str, Any]] = []
    numerical_modes: dict[str, pd.DataFrame] = {}

    for index, case in enumerate(cases, start=1):
        print(f"\n=== AUDIT {index}/{len(cases)}: {case.case_id} ===")
        print(asdict(case))

        root, history = solve_root(
            args=args,
            alpha=args.target_alpha,
            seed_cr=target_cr,
            seed_ci=target_ci,
            spectral_extent=case.spectral_extent,
            matching_y=case.matching_y,
            max_step=case.max_step,
            rtol=case.rtol,
            atol=case.atol,
            label=f"audit_{case.case_id}",
            cr_half_width=0.03,
            ci_factor=1000.0,
        )
        history.to_csv(
            output_dir / f"history_{case.case_id}.csv",
            index=False,
        )
        c = complex(root["cr"], root["ci"])
        numerical, complete, mode_diagnostics = numerical_mode_and_analytic_tails(
            args=args,
            case=case,
            alpha=args.target_alpha,
            c=c,
        )
        numerical.to_csv(
            output_dir / f"mode_numerical_{case.case_id}.csv",
            index=False,
        )
        complete.to_csv(
            output_dir / f"mode_with_analytic_tails_{case.case_id}.csv",
            index=False,
        )
        fd = finite_difference_residual(
            numerical,
            args=args,
            alpha=args.target_alpha,
            c=c,
            matching_y=case.matching_y,
        )
        row = {
            **asdict(case),
            "cr": root["cr"],
            "ci": root["ci"],
            "omega_i": root["omega_i"],
            "root_residual_norm": root["residual_norm"],
            "root_accepted": root["accepted"],
            "optimizer_success": root["optimizer_success"],
            "active_lower_ci": root["active_lower_ci"],
            "active_upper_ci": root["active_upper_ci"],
            **mode_diagnostics,
            **fd,
        }
        rows.append(row)
        numerical_modes[case.case_id] = numerical

        print(
            f"c={root['cr']:.15g}+{root['ci']:.8e}i "
            f"||F||={root['residual_norm']:.3e} "
            f"analytic tails=[{mode_diagnostics['analytic_far_left']:.3e}, "
            f"{mode_diagnostics['analytic_far_right']:.3e}]",
            flush=True,
        )

    baseline = numerical_modes["baseline_fine"]
    for row in rows:
        row.update(
            compare_modes(
                numerical_modes[row["case_id"]],
                baseline,
                core_y=args.core_y,
            )
        )

    summary = pd.DataFrame(rows)
    baseline_row = summary.loc[
        summary["case_id"] == "baseline_fine"
    ].iloc[0]
    summary["delta_cr_vs_baseline"] = summary["cr"] - baseline_row["cr"]
    summary["delta_ci_vs_baseline"] = summary["ci"] - baseline_row["ci"]
    summary["delta_omega_i_vs_baseline"] = (
        summary["omega_i"] - baseline_row["omega_i"]
    )
    summary.to_csv(output_dir / "audit_summary.csv", index=False)
    plot_audit(summary, output_dir)

    report = {
        "case": {
            "case_id": args.case_id,
            "Mach": args.Mach,
            "start_alpha": args.start_alpha,
            "target_alpha": args.target_alpha,
        },
        "target_from_continuation": {
            "cr": target_cr,
            "ci": target_ci,
            "omega_i": args.target_alpha * target_ci,
            "residual_norm": float(target_row["residual_norm"]),
        },
        "baseline": baseline_row.to_dict(),
        "all_roots_accepted": bool(summary["root_accepted"].all()),
        "max_abs_delta_cr_vs_baseline": float(
            np.max(np.abs(summary["delta_cr_vs_baseline"]))
        ),
        "max_abs_delta_ci_vs_baseline": float(
            np.max(np.abs(summary["delta_ci_vs_baseline"]))
        ),
        "max_mode_relative_l2_core_vs_baseline": float(
            np.max(summary["mode_relative_l2_core_vs_baseline"])
        ),
        "max_envelope_relative_l2_core_vs_baseline": float(
            np.max(summary["envelope_relative_l2_core_vs_baseline"])
        ),
        "minimum_ci": float(np.min(summary["ci"])),
        "maximum_ci": float(np.max(summary["ci"])),
        "rows": rows,
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )

    display_columns = [
        "case_id",
        "family",
        "spectral_extent",
        "matching_y",
        "max_step",
        "rtol",
        "cr",
        "ci",
        "root_residual_norm",
        "delta_cr_vs_baseline",
        "delta_ci_vs_baseline",
        "mode_relative_l2_core_vs_baseline",
        "envelope_relative_l2_core_vs_baseline",
        "mu_left",
        "mu_right",
        "analytic_far_left",
        "analytic_far_right",
        "fd_residual_rms_significant",
    ]
    print("\n=== AUDIT SUMMARY ===")
    print(summary[display_columns].to_string(index=False))
    print("\n=== VERDICT ===")
    print(f"all_roots_accepted: {report['all_roots_accepted']}")
    print(
        "max_abs_delta_cr_vs_baseline: "
        f"{report['max_abs_delta_cr_vs_baseline']:.6e}"
    )
    print(
        "max_abs_delta_ci_vs_baseline: "
        f"{report['max_abs_delta_ci_vs_baseline']:.6e}"
    )
    print(
        "max_mode_relative_l2_core_vs_baseline: "
        f"{report['max_mode_relative_l2_core_vs_baseline']:.6e}"
    )
    print(
        "max_envelope_relative_l2_core_vs_baseline: "
        f"{report['max_envelope_relative_l2_core_vs_baseline']:.6e}"
    )
    print(f"\nWritten to: {output_dir}")

    return 0 if report["all_roots_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
