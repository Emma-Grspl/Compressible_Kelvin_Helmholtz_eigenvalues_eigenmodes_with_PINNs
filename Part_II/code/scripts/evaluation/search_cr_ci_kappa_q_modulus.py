#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# This companion module must be in the same directory.
import scripts.evaluation.test_kappa_q_modulus_reconstruction as base


DEFAULT_MACH = 1.4
DEFAULT_ALPHA = 0.18125
DEFAULT_REFERENCE_CR = 0.3153193678544234
DEFAULT_REFERENCE_CI = 0.0427209194542578


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search c_r and c_i at fixed Mach/alpha by matching the left and "
            "right Riccati variables, while integrating (kappa, q, |p|). "
            "After optimization, reconstruct the complete mode and compare it "
            "with the historical modal reference."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())

    parser.add_argument("--Mach", type=float, default=DEFAULT_MACH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)

    parser.add_argument("--seed-cr", type=float, default=DEFAULT_REFERENCE_CR)
    parser.add_argument("--seed-ci", type=float, default=DEFAULT_REFERENCE_CI)
    parser.add_argument(
        "--reference-cr", type=float, default=DEFAULT_REFERENCE_CR
    )
    parser.add_argument(
        "--reference-ci", type=float, default=DEFAULT_REFERENCE_CI
    )

    parser.add_argument("--cr-lower", type=float, default=0.25)
    parser.add_argument("--cr-upper", type=float, default=0.40)
    parser.add_argument("--ci-lower", type=float, default=0.01)
    parser.add_argument("--ci-upper", type=float, default=0.08)

    parser.add_argument("--Ly", type=float, default=500.0)
    parser.add_argument("--matching-y", type=float, default=1.0)

    parser.add_argument("--max-step", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument(
        "--method", choices=("RK45", "DOP853"), default="DOP853"
    )

    parser.add_argument(
        "--optimizer-xtol", type=float, default=1.0e-11
    )
    parser.add_argument(
        "--optimizer-ftol", type=float, default=1.0e-11
    )
    parser.add_argument(
        "--optimizer-gtol", type=float, default=1.0e-11
    )
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument(
        "--diff-step",
        type=float,
        default=1.0e-5,
        help="Relative finite-difference step used by scipy least_squares.",
    )
    parser.add_argument(
        "--x-scale-cr", type=float, default=0.05
    )
    parser.add_argument(
        "--x-scale-ci", type=float, default=0.02
    )
    parser.add_argument(
        "--accept-residual",
        type=float,
        default=1.0e-8,
        help="Maximum accepted norm sqrt(delta_kappa^2+delta_q^2).",
    )

    parser.add_argument(
        "--output-dy",
        type=float,
        default=0.025,
        help="Output spacing used only for the final modal reconstruction.",
    )
    parser.add_argument("--core-y", type=float, default=20.0)

    parser.add_argument(
        "--reference-parquet",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--require-accepted-root",
        action="store_true",
        help="Return a nonzero status if the final matching residual is too large.",
    )
    return parser.parse_args()


def endpoint_integration(
    *,
    side: str,
    Mach: float,
    alpha: float,
    c: complex,
    Ly: float,
    matching_y: float,
    max_step: float,
    rtol: float,
    atol: float,
    method: str,
) -> dict[str, Any]:
    """Integrate (kappa, q, |p|) to the matching point.

    The complete adaptive trajectory is integrated internally by solve_ivp,
    but only the endpoint is retained during the eigenvalue search.
    """
    if side == "left":
        start = -Ly
    elif side == "right":
        start = Ly
    else:
        raise ValueError(f"Unknown side: {side}")

    gamma0 = base.asymptotic_gamma(
        side=side,
        Mach=Mach,
        alpha=alpha,
        c=c,
    )
    initial = np.asarray(
        [gamma0.real, gamma0.imag, 1.0],
        dtype=float,
    )

    solution = solve_ivp(
        lambda y, state: base.rhs_kappa_q_modulus(
            y,
            state,
            Mach=Mach,
            alpha=alpha,
            c=c,
        ),
        (start, matching_y),
        initial,
        method=method,
        t_eval=np.asarray([matching_y], dtype=float),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    if not solution.success:
        raise RuntimeError(
            f"{side} integration failed for c={c}: {solution.message}"
        )
    if solution.y.shape != (3, 1):
        raise RuntimeError(
            f"Unexpected {side} endpoint shape: {solution.y.shape}"
        )

    endpoint = np.asarray(solution.y[:, -1], dtype=float)
    if not np.all(np.isfinite(endpoint)):
        raise RuntimeError(
            f"{side} integration produced non-finite endpoint values."
        )
    if endpoint[2] <= 0.0:
        raise RuntimeError(
            f"{side} integrated modulus became non-positive at matching point: "
            f"{endpoint[2]:.6e}"
        )

    return {
        "kappa": float(endpoint[0]),
        "q": float(endpoint[1]),
        "modulus": float(endpoint[2]),
        "gamma0": gamma0,
        "nfev": int(solution.nfev),
        "njev": int(solution.njev),
        "nlu": int(solution.nlu),
    }


@dataclass
class MatchingObjective:
    args: argparse.Namespace
    history: list[dict[str, Any]]
    evaluation_count: int = 0

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        self.evaluation_count += 1
        cr = float(vector[0])
        ci = float(vector[1])
        c = complex(cr, ci)

        record: dict[str, Any] = {
            "evaluation": self.evaluation_count,
            "cr": cr,
            "ci": ci,
        }

        try:
            left = endpoint_integration(
                side="left",
                Mach=self.args.Mach,
                alpha=self.args.alpha,
                c=c,
                Ly=self.args.Ly,
                matching_y=self.args.matching_y,
                max_step=self.args.max_step,
                rtol=self.args.rtol,
                atol=self.args.atol,
                method=self.args.method,
            )
            right = endpoint_integration(
                side="right",
                Mach=self.args.Mach,
                alpha=self.args.alpha,
                c=c,
                Ly=self.args.Ly,
                matching_y=self.args.matching_y,
                max_step=self.args.max_step,
                rtol=self.args.rtol,
                atol=self.args.atol,
                method=self.args.method,
            )

            delta_kappa = left["kappa"] - right["kappa"]
            delta_q = left["q"] - right["q"]
            residual_norm = math.hypot(delta_kappa, delta_q)
            amplitude_ratio = left["modulus"] / right["modulus"]

            record.update(
                {
                    "success": True,
                    "delta_kappa": delta_kappa,
                    "delta_q": delta_q,
                    "residual_norm": residual_norm,
                    "left_kappa_match": left["kappa"],
                    "left_q_match": left["q"],
                    "left_modulus_match": left["modulus"],
                    "right_kappa_match": right["kappa"],
                    "right_q_match": right["q"],
                    "right_modulus_match": right["modulus"],
                    "amplitude_scale_right": amplitude_ratio,
                    "left_nfev": left["nfev"],
                    "right_nfev": right["nfev"],
                    "error": "",
                }
            )
            residual = np.asarray([delta_kappa, delta_q], dtype=float)

            print(
                f"[eval {self.evaluation_count:03d}] "
                f"cr={cr:.15g} ci={ci:.15g} | "
                f"dκ={delta_kappa:+.6e} dq={delta_q:+.6e} | "
                f"||F||={residual_norm:.6e}",
                flush=True,
            )
        except Exception as exc:
            # Keep the optimizer inside the valid part of parameter space.
            penalty = 1.0e3
            record.update(
                {
                    "success": False,
                    "delta_kappa": penalty,
                    "delta_q": penalty,
                    "residual_norm": math.sqrt(2.0) * penalty,
                    "left_kappa_match": math.nan,
                    "left_q_match": math.nan,
                    "left_modulus_match": math.nan,
                    "right_kappa_match": math.nan,
                    "right_q_match": math.nan,
                    "right_modulus_match": math.nan,
                    "amplitude_scale_right": math.nan,
                    "left_nfev": 0,
                    "right_nfev": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            residual = np.asarray([penalty, penalty], dtype=float)
            print(
                f"[eval {self.evaluation_count:03d}] "
                f"cr={cr:.15g} ci={ci:.15g} | FAILED: {exc}",
                flush=True,
            )

        self.history.append(record)
        return residual


def validate_arguments(args: argparse.Namespace) -> None:
    if args.Ly <= 0.0:
        raise ValueError("--Ly must be positive.")
    if not (-args.Ly < args.matching_y < args.Ly):
        raise ValueError("--matching-y must lie inside (-Ly, Ly).")
    if args.max_step <= 0.0:
        raise ValueError("--max-step must be positive.")
    if args.output_dy <= 0.0:
        raise ValueError("--output-dy must be positive.")
    if not args.cr_lower < args.cr_upper:
        raise ValueError("Invalid c_r bounds.")
    if not args.ci_lower < args.ci_upper:
        raise ValueError("Invalid c_i bounds.")
    if args.ci_lower <= 0.0:
        raise ValueError("--ci-lower must be strictly positive.")
    if not args.cr_lower <= args.seed_cr <= args.cr_upper:
        raise ValueError("The seed c_r lies outside the requested bounds.")
    if not args.ci_lower <= args.seed_ci <= args.ci_upper:
        raise ValueError("The seed c_i lies outside the requested bounds.")
    if args.max_nfev < 1:
        raise ValueError("--max-nfev must be positive.")
    if args.accept_residual <= 0.0:
        raise ValueError("--accept-residual must be positive.")


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    repo = args.repo.expanduser().resolve()

    if args.output_dir is None:
        output_dir = (
            repo
            / "classic_supersonic/reproducibility/results"
            / "kappa_q_modulus_eigenvalue_search_M140_a018125"
        )
    else:
        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = repo / output_dir
        output_dir = output_dir.resolve()

    if args.reference_parquet is None:
        reference_path = (
            repo
            / "classic_supersonic/data/modal"
            / "supersonic_reference_v2_modal_raw.parquet"
        )
    else:
        reference_path = args.reference_parquet.expanduser()
        if not reference_path.is_absolute():
            reference_path = repo / reference_path
        reference_path = reference_path.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    return repo, output_dir, reference_path


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    validate_arguments(args)
    repo, output_dir, reference_path = resolve_paths(args)

    reference_c = complex(args.reference_cr, args.reference_ci)
    seed = np.asarray([args.seed_cr, args.seed_ci], dtype=float)

    print("=== Joint eigenvalue/mode test with (kappa, q, |p|) ===")
    print(f"Repository       : {repo}")
    print(f"Output           : {output_dir}")
    print(f"M                : {args.Mach}")
    print(f"alpha            : {args.alpha}")
    print(
        "historical c     : "
        f"{reference_c.real:.15g} + {reference_c.imag:.15g} i"
    )
    print(
        "initial seed     : "
        f"{args.seed_cr:.15g} + {args.seed_ci:.15g} i"
    )
    print(
        f"bounds c_r       : [{args.cr_lower}, {args.cr_upper}]"
    )
    print(
        f"bounds c_i       : [{args.ci_lower}, {args.ci_upper}]"
    )
    print(f"Ly               : {args.Ly}")
    print(f"matching_y       : {args.matching_y}")
    print(f"max_step         : {args.max_step}")
    print(f"rtol / atol      : {args.rtol} / {args.atol}")
    print(f"optimizer max_nfev: {args.max_nfev}")
    print()

    history: list[dict[str, Any]] = []
    objective = MatchingObjective(args=args, history=history)

    result = least_squares(
        objective,
        x0=seed,
        bounds=(
            np.asarray([args.cr_lower, args.ci_lower], dtype=float),
            np.asarray([args.cr_upper, args.ci_upper], dtype=float),
        ),
        method="trf",
        jac="2-point",
        diff_step=args.diff_step,
        x_scale=np.asarray(
            [args.x_scale_cr, args.x_scale_ci], dtype=float
        ),
        xtol=args.optimizer_xtol,
        ftol=args.optimizer_ftol,
        gtol=args.optimizer_gtol,
        max_nfev=args.max_nfev,
        verbose=1,
    )

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "eigenvalue_search_history.csv", index=False)

    optimized_cr = float(result.x[0])
    optimized_ci = float(result.x[1])
    optimized_c = complex(optimized_cr, optimized_ci)

    final_residual = objective(result.x)
    final_delta_kappa = float(final_residual[0])
    final_delta_q = float(final_residual[1])
    final_residual_norm = float(np.linalg.norm(final_residual))
    root_accepted = bool(
        np.all(np.isfinite(final_residual))
        and final_residual_norm <= args.accept_residual
    )

    # Rewrite history after the explicit final evaluation.
    pd.DataFrame(history).to_csv(
        output_dir / "eigenvalue_search_history.csv",
        index=False,
    )

    print("\n=== Optimized eigenvalue ===")
    print(f"optimizer success : {result.success}")
    print(f"optimizer status  : {result.status}")
    print(f"optimizer message : {result.message}")
    print(f"optimized cr      : {optimized_cr:.15g}")
    print(f"optimized ci      : {optimized_ci:.15g}")
    print(f"delta cr vs ref   : {optimized_cr - args.reference_cr:+.12e}")
    print(f"delta ci vs ref   : {optimized_ci - args.reference_ci:+.12e}")
    print(
        "delta omega_i    : "
        f"{args.alpha * (optimized_ci - args.reference_ci):+.12e}"
    )
    print(f"final delta kappa : {final_delta_kappa:+.12e}")
    print(f"final delta q     : {final_delta_q:+.12e}")
    print(f"final ||F||       : {final_residual_norm:.12e}")
    print(f"root accepted     : {root_accepted}")

    # Final full modal integration at the optimized eigenvalue.
    left = base.integrate_branch(
        side="left",
        Mach=args.Mach,
        alpha=args.alpha,
        c=optimized_c,
        Ly=args.Ly,
        matching_y=args.matching_y,
        output_dy=args.output_dy,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
        method=args.method,
    )
    right = base.integrate_branch(
        side="right",
        Mach=args.Mach,
        alpha=args.alpha,
        c=optimized_c,
        Ly=args.Ly,
        matching_y=args.matching_y,
        output_dy=args.output_dy,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
        method=args.method,
    )

    frame, matching_diagnostics = base.reconstruct_mode(left, right)
    residual_diagnostics = base.numerical_residuals(
        frame,
        Mach=args.Mach,
        alpha=args.alpha,
        c=optimized_c,
        matching_y=args.matching_y,
        core_y=args.core_y,
    )

    frame.to_csv(
        output_dir / "kappa_q_modulus_mode_at_optimized_c.csv",
        index=False,
    )

    base.plot_reconstruction(
        frame,
        Mach=args.Mach,
        alpha=args.alpha,
        c=optimized_c,
        matching_y=args.matching_y,
        output_dir=output_dir,
    )

    reference_diagnostics = base.compare_with_reference(
        frame,
        reference_path=reference_path,
        Mach=args.Mach,
        alpha=args.alpha,
        core_y=args.core_y,
        output_dir=output_dir,
    )

    summary = {
        "parameters": {
            "Mach": args.Mach,
            "alpha": args.alpha,
            "Ly": args.Ly,
            "matching_y": args.matching_y,
            "max_step": args.max_step,
            "rtol": args.rtol,
            "atol": args.atol,
            "method": args.method,
            "output_dy": args.output_dy,
            "core_y": args.core_y,
            "bounds": {
                "cr_lower": args.cr_lower,
                "cr_upper": args.cr_upper,
                "ci_lower": args.ci_lower,
                "ci_upper": args.ci_upper,
            },
        },
        "historical_reference": {
            "cr": args.reference_cr,
            "ci": args.reference_ci,
            "omega_i": args.alpha * args.reference_ci,
        },
        "optimized_eigenvalue": {
            "cr": optimized_cr,
            "ci": optimized_ci,
            "omega_i": args.alpha * optimized_ci,
            "delta_cr_vs_reference": optimized_cr - args.reference_cr,
            "delta_ci_vs_reference": optimized_ci - args.reference_ci,
            "delta_omega_i_vs_reference": (
                args.alpha * (optimized_ci - args.reference_ci)
            ),
        },
        "optimizer": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
            "njev": (
                int(result.njev)
                if result.njev is not None
                else None
            ),
            "active_mask": result.active_mask.tolist(),
        },
        "root_test": {
            "delta_kappa": final_delta_kappa,
            "delta_q": final_delta_q,
            "residual_norm": final_residual_norm,
            "accept_residual": args.accept_residual,
            "accepted": root_accepted,
        },
        "final_mode_matching": matching_diagnostics,
        "finite_difference_checks": residual_diagnostics,
        "old_reference_comparison": reference_diagnostics,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )

    print("\n=== Final mode diagnostics ===")
    for key, value in matching_diagnostics.items():
        print(f"{key}: {value}")

    print("\n=== Independent finite-difference checks ===")
    for key, value in residual_diagnostics.items():
        print(f"{key}: {value}")

    print("\n=== Comparison with historical mode ===")
    for key, value in reference_diagnostics.items():
        if key != "reference_columns":
            print(f"{key}: {value}")

    print("\nWritten files:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path}")

    if args.require_accepted_root and not root_accepted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
