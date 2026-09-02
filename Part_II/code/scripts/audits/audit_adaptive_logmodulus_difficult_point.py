#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
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


SPECTRAL_CANDIDATES = (
    "assets/classic_supersonic/csv/pinn_direct/spectral/table_supersonic_primary_modal_spectral_133pts.csv",
    "assets/classic_supersonic/csv/pinn_direct/spectral/table_supersonic_reference_v2_spectral.csv",
    "assets/classic_supersonic/csv/pinn_direct/tables/table_supersonic_reference_v2_spectral_validated_50pts.csv",
    "assets/classic_supersonic/csv/pinn_direct/spectral/table_supersonic_reference_v2_spectral.csv",
    "assets/classic_supersonic/csv/pinn_direct/tables/table_supersonic_reference_v2_spectral_validated_50pts.csv",
)


@dataclass(frozen=True)
class AuditCase:
    case_id: str
    family: str
    y_left: float
    y_right: float
    matching_y: float
    max_step: float
    rtol: float
    atol: float


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive convergence audit for the supersonic Riccati eigenvalue "
            "solver and the reconstructed mode using (kappa, q, log|p|)."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--case-id", type=str, required=True)
    parser.add_argument("--Mach", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)

    parser.add_argument("--seed-cr", type=float, default=None)
    parser.add_argument("--seed-ci", type=float, default=None)
    parser.add_argument("--reference-cr", type=float, default=None)
    parser.add_argument("--reference-ci", type=float, default=None)
    parser.add_argument("--spectral-csv", type=Path, default=None)
    parser.add_argument(
        "--allow-nearest-seed",
        action="store_true",
        help=(
            "When no exact spectral row exists, use the nearest alpha at the "
            "same Mach only as an optimizer seed. The provenance is recorded."
        ),
    )
    parser.add_argument(
        "--nearest-alpha-max-distance",
        type=float,
        default=0.05,
    )

    parser.add_argument("--suite", choices=("smoke", "quick", "full"), default="full")
    parser.add_argument("--method", choices=("DOP853", "RK45"), default="DOP853")
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument("--diff-step", type=float, default=1.0e-5)
    parser.add_argument("--root-tolerance", type=float, default=1.0e-8)

    parser.add_argument("--matching-y", type=float, default=1.0)
    parser.add_argument("--core-y", type=float, default=20.0)
    parser.add_argument("--core-dy", type=float, default=0.025)

    parser.add_argument(
        "--tail-tolerance",
        type=float,
        default=1.0e-3,
        help="Target normalized modal amplitude at each boundary.",
    )
    parser.add_argument("--box-safety-factor", type=float, default=1.20)
    parser.add_argument("--minimum-extent", type=float, default=20.0)
    parser.add_argument("--maximum-extent", type=float, default=5000.0)
    parser.add_argument("--max-box-updates", type=int, default=3)
    parser.add_argument(
        "--box-update-relative-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--left-extent",
        type=float,
        default=None,
        help="Optional manual positive extent |y_left|; disables automatic value on the left.",
    )
    parser.add_argument(
        "--right-extent",
        type=float,
        default=None,
        help="Optional manual positive extent y_right; disables automatic value on the right.",
    )

    parser.add_argument(
        "--box-scales",
        type=parse_float_list,
        default=parse_float_list("0.6,0.8,1.0,1.25,1.5"),
    )
    parser.add_argument(
        "--matching-values",
        type=parse_float_list,
        default=parse_float_list("0.0,0.5,1.0,1.5,2.0"),
    )
    parser.add_argument(
        "--max-step-values",
        type=parse_float_list,
        default=parse_float_list("0.25,0.5,1.0"),
    )
    parser.add_argument(
        "--rtol-values",
        type=parse_float_list,
        default=parse_float_list("1e-10,1e-9,1e-8"),
    )

    parser.add_argument("--nominal-max-step", type=float, default=0.5)
    parser.add_argument("--nominal-rtol", type=float, default=1.0e-9)
    parser.add_argument("--nominal-atol", type=float, default=1.0e-11)
    parser.add_argument("--fine-max-step", type=float, default=0.25)
    parser.add_argument("--fine-rtol", type=float, default=1.0e-11)
    parser.add_argument("--fine-atol", type=float, default=1.0e-13)

    parser.add_argument("--cr-half-width", type=float, default=0.15)
    parser.add_argument("--ci-half-width", type=float, default=0.08)
    parser.add_argument("--ci-lower", type=float, default=1.0e-10)
    parser.add_argument("--ci-upper", type=float, default=0.15)

    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_existing_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def candidate_quality(frame: pd.DataFrame) -> pd.Series:
    quality = pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)

    for column in ("trusted_spectral", "best_spectral_success"):
        if column in frame.columns:
            values = frame[column].astype(str).str.lower()
            quality += values.isin(("true", "1", "yes")).astype(float) * 20.0

    for column in ("validation_status", "best_status", "campaign_target_status"):
        if column in frame.columns:
            values = frame[column].astype(str).str.lower()
            quality += values.str.contains("valid", na=False).astype(float) * 10.0
            quality -= values.str.contains("reject|fail|unresolved", na=False).astype(float) * 20.0

    mismatch_column = first_existing_column(
        frame,
        (
            "best_stage1_mismatch",
            "stage1_mismatch",
            "max_stage1",
            "canonical_stage1_mismatch",
        ),
    )
    if mismatch_column is not None:
        mismatch = numeric(frame[mismatch_column])
        finite = np.isfinite(mismatch)
        quality.loc[finite] += np.maximum(
            0.0,
            -np.log10(np.maximum(mismatch.loc[finite], 1.0e-30)),
        )
    return quality


def load_seed_from_tables(
    *,
    repo: Path,
    Mach: float,
    alpha: float,
    spectral_csv: Path | None,
    allow_nearest: bool,
    nearest_max_distance: float,
) -> dict[str, Any]:
    if spectral_csv is not None:
        paths = [spectral_csv if spectral_csv.is_absolute() else repo / spectral_csv]
    else:
        paths = [repo / relative for relative in SPECTRAL_CANDIDATES]

    inspected: list[str] = []
    exact_records: list[dict[str, Any]] = []
    nearest_records: list[dict[str, Any]] = []

    for path in paths:
        if not path.exists():
            continue
        inspected.append(str(path))
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue

        mach_column = first_existing_column(frame, ("Mach", "M", "mach"))
        alpha_column = first_existing_column(frame, ("alpha", "Alpha"))
        cr_column = first_existing_column(frame, ("cr", "reference_cr", "best_cr"))
        ci_column = first_existing_column(frame, ("ci", "reference_ci", "best_ci"))
        if None in (mach_column, alpha_column, cr_column, ci_column):
            continue

        work = frame.copy()
        work["__Mach"] = numeric(work[mach_column])
        work["__alpha"] = numeric(work[alpha_column])
        work["__cr"] = numeric(work[cr_column])
        work["__ci"] = numeric(work[ci_column])
        work["__quality"] = candidate_quality(work)
        work = work[
            np.isfinite(work["__Mach"])
            & np.isfinite(work["__alpha"])
            & np.isfinite(work["__cr"])
            & np.isfinite(work["__ci"])
        ].copy()
        if work.empty:
            continue

        same_mach = work[np.isclose(work["__Mach"], Mach, rtol=0.0, atol=5.0e-10)].copy()
        if same_mach.empty:
            continue

        exact = same_mach[
            np.isclose(same_mach["__alpha"], alpha, rtol=0.0, atol=5.0e-10)
        ].copy()
        if not exact.empty:
            exact = exact.sort_values("__quality", ascending=False)
            row = exact.iloc[0]
            exact_records.append(
                {
                    "cr": float(row["__cr"]),
                    "ci": float(row["__ci"]),
                    "source_path": str(path),
                    "source_kind": "exact",
                    "source_alpha": float(row["__alpha"]),
                    "quality": float(row["__quality"]),
                    "row": {
                        str(key): value
                        for key, value in row.items()
                        if not str(key).startswith("__")
                    },
                }
            )

        nearest = same_mach.iloc[
            np.argmin(np.abs(same_mach["__alpha"].to_numpy(float) - alpha))
        ]
        distance = abs(float(nearest["__alpha"]) - alpha)
        nearest_records.append(
            {
                "cr": float(nearest["__cr"]),
                "ci": float(nearest["__ci"]),
                "source_path": str(path),
                "source_kind": "nearest",
                "source_alpha": float(nearest["__alpha"]),
                "alpha_distance": distance,
                "quality": float(nearest["__quality"]),
                "row": {
                    str(key): value
                    for key, value in nearest.items()
                    if not str(key).startswith("__")
                },
            }
        )

    if exact_records:
        exact_records.sort(key=lambda record: record["quality"], reverse=True)
        best = exact_records[0]
        best["inspected_paths"] = inspected
        return best

    if allow_nearest and nearest_records:
        nearest_records = [
            record
            for record in nearest_records
            if record["alpha_distance"] <= nearest_max_distance
        ]
        if nearest_records:
            nearest_records.sort(
                key=lambda record: (
                    record["alpha_distance"],
                    -record["quality"],
                )
            )
            best = nearest_records[0]
            best["inspected_paths"] = inspected
            return best

    raise RuntimeError(
        "No exact spectral seed found for "
        f"M={Mach}, alpha={alpha}. Inspected: {inspected}. "
        "Provide --seed-cr/--seed-ci, or use --allow-nearest-seed after "
        "checking that the nearest point belongs to the intended branch."
    )


def flow(y: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U = np.tanh(y)
    Up = 1.0 - U * U
    return U, Up


def rhs_logmodulus(
    y: float,
    state: np.ndarray,
    *,
    Mach: float,
    alpha: float,
    c: complex,
) -> np.ndarray:
    kappa = float(state[0])
    q = float(state[1])
    gamma = complex(kappa, q)
    U, Up = flow(y)
    P = -2.0 * Up / (U - c)
    R = 1.0 - Mach * Mach * (U - c) ** 2
    gamma_prime = -(gamma * gamma) - P * gamma + alpha * alpha * R
    return np.asarray(
        [gamma_prime.real, gamma_prime.imag, kappa],
        dtype=float,
    )


def decay_rates(
    *,
    Mach: float,
    alpha: float,
    c: complex,
) -> dict[str, Any]:
    gamma_left = base.asymptotic_gamma(
        side="left", Mach=Mach, alpha=alpha, c=c
    )
    gamma_right = base.asymptotic_gamma(
        side="right", Mach=Mach, alpha=alpha, c=c
    )
    mu_left = float(gamma_left.real)
    mu_right = float(-gamma_right.real)
    return {
        "gamma_left": gamma_left,
        "gamma_right": gamma_right,
        "mu_left": mu_left,
        "mu_right": mu_right,
    }


def required_extent(
    *,
    mu: float,
    tail_tolerance: float,
    safety_factor: float,
    minimum_extent: float,
    maximum_extent: float,
) -> tuple[float, str]:
    if not np.isfinite(mu) or mu <= 0.0:
        return float(maximum_extent), "nondecaying_or_unresolved"
    estimate = safety_factor * abs(math.log(tail_tolerance)) / mu
    extent = max(minimum_extent, estimate)
    if extent >= maximum_extent:
        return float(maximum_extent), "capped_weak_decay"
    return float(extent), "resolved"


def estimate_box(
    *,
    Mach: float,
    alpha: float,
    c: complex,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rates = decay_rates(Mach=Mach, alpha=alpha, c=c)

    if args.left_extent is None:
        left_extent, left_status = required_extent(
            mu=rates["mu_left"],
            tail_tolerance=args.tail_tolerance,
            safety_factor=args.box_safety_factor,
            minimum_extent=args.minimum_extent,
            maximum_extent=args.maximum_extent,
        )
    else:
        left_extent = float(args.left_extent)
        left_status = "manual"

    if args.right_extent is None:
        right_extent, right_status = required_extent(
            mu=rates["mu_right"],
            tail_tolerance=args.tail_tolerance,
            safety_factor=args.box_safety_factor,
            minimum_extent=args.minimum_extent,
            maximum_extent=args.maximum_extent,
        )
    else:
        right_extent = float(args.right_extent)
        right_status = "manual"

    return {
        **rates,
        "left_extent": left_extent,
        "right_extent": right_extent,
        "y_left": -left_extent,
        "y_right": right_extent,
        "left_status": left_status,
        "right_status": right_status,
    }


def integrate_endpoint(
    *,
    side: str,
    Mach: float,
    alpha: float,
    c: complex,
    y_left: float,
    y_right: float,
    matching_y: float,
    max_step: float,
    rtol: float,
    atol: float,
    method: str,
) -> dict[str, Any]:
    if side == "left":
        start = y_left
    elif side == "right":
        start = y_right
    else:
        raise ValueError(side)

    gamma0 = base.asymptotic_gamma(
        side=side,
        Mach=Mach,
        alpha=alpha,
        c=c,
    )
    initial = np.asarray([gamma0.real, gamma0.imag, 0.0], dtype=float)

    solution = solve_ivp(
        lambda y, state: rhs_logmodulus(
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
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"{side} integration failed: {solution.message}")

    endpoint = solution.y[:, -1]
    if not np.all(np.isfinite(endpoint)):
        raise RuntimeError(f"{side} endpoint is non-finite.")

    return {
        "kappa": float(endpoint[0]),
        "q": float(endpoint[1]),
        "ell": float(endpoint[2]),
        "nfev": int(solution.nfev),
    }


class Objective:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        case: AuditCase,
        history: list[dict[str, Any]],
    ) -> None:
        self.args = args
        self.case = case
        self.history = history
        self.count = 0

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        self.count += 1
        cr, ci = map(float, vector)
        c = complex(cr, ci)
        record: dict[str, Any] = {
            "evaluation": self.count,
            "cr": cr,
            "ci": ci,
        }

        try:
            left = integrate_endpoint(
                side="left",
                Mach=self.args.Mach,
                alpha=self.args.alpha,
                c=c,
                y_left=self.case.y_left,
                y_right=self.case.y_right,
                matching_y=self.case.matching_y,
                max_step=self.case.max_step,
                rtol=self.case.rtol,
                atol=self.case.atol,
                method=self.args.method,
            )
            right = integrate_endpoint(
                side="right",
                Mach=self.args.Mach,
                alpha=self.args.alpha,
                c=c,
                y_left=self.case.y_left,
                y_right=self.case.y_right,
                matching_y=self.case.matching_y,
                max_step=self.case.max_step,
                rtol=self.case.rtol,
                atol=self.case.atol,
                method=self.args.method,
            )
            dk = left["kappa"] - right["kappa"]
            dq = left["q"] - right["q"]
            norm = math.hypot(dk, dq)
            record.update(
                {
                    "success": True,
                    "delta_kappa": dk,
                    "delta_q": dq,
                    "residual_norm": norm,
                    "left_ell_match": left["ell"],
                    "right_ell_match": right["ell"],
                    "log_amplitude_shift_right": left["ell"] - right["ell"],
                    "left_nfev": left["nfev"],
                    "right_nfev": right["nfev"],
                    "error": "",
                }
            )
            residual = np.asarray([dk, dq], dtype=float)
            print(
                f"[{self.case.case_id} eval {self.count:03d}] "
                f"cr={cr:.15g} ci={ci:.15g} "
                f"dκ={dk:+.4e} dq={dq:+.4e} ||F||={norm:.4e}",
                flush=True,
            )
        except Exception as exc:
            penalty = 1.0e3
            record.update(
                {
                    "success": False,
                    "delta_kappa": penalty,
                    "delta_q": penalty,
                    "residual_norm": math.sqrt(2.0) * penalty,
                    "left_ell_match": np.nan,
                    "right_ell_match": np.nan,
                    "log_amplitude_shift_right": np.nan,
                    "left_nfev": 0,
                    "right_nfev": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            residual = np.asarray([penalty, penalty], dtype=float)
            print(
                f"[{self.case.case_id} eval {self.count:03d}] FAILED: {exc}",
                flush=True,
            )

        self.history.append(record)
        return residual


def solve_root(
    *,
    args: argparse.Namespace,
    case: AuditCase,
    seed: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    history: list[dict[str, Any]] = []
    objective = Objective(args=args, case=case, history=history)

    cr_lower = max(-0.2, float(seed[0]) - args.cr_half_width)
    cr_upper = min(1.2, float(seed[0]) + args.cr_half_width)
    ci_lower = args.ci_lower
    ci_upper = max(
        args.ci_upper,
        min(0.5, float(seed[1]) + args.ci_half_width),
    )
    x0 = np.asarray(
        [
            np.clip(seed[0], cr_lower + 1.0e-12, cr_upper - 1.0e-12),
            np.clip(seed[1], ci_lower + 1.0e-12, ci_upper - 1.0e-12),
        ],
        dtype=float,
    )

    result = least_squares(
        objective,
        x0=x0,
        bounds=(
            np.asarray([cr_lower, ci_lower], dtype=float),
            np.asarray([cr_upper, ci_upper], dtype=float),
        ),
        method="trf",
        jac="2-point",
        diff_step=args.diff_step,
        x_scale=np.asarray(
            [
                max(0.02, args.cr_half_width / 2.0),
                max(0.005, args.ci_half_width / 2.0),
            ],
            dtype=float,
        ),
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=args.max_nfev,
        verbose=0,
    )

    residual = objective(result.x)
    norm = float(np.linalg.norm(residual))
    row = {
        "cr": float(result.x[0]),
        "ci": float(result.x[1]),
        "omega_i": float(args.alpha * result.x[1]),
        "root_delta_kappa": float(residual[0]),
        "root_delta_q": float(residual[1]),
        "root_residual_norm": norm,
        "root_accepted": bool(norm <= args.root_tolerance),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "optimizer_njev": int(result.njev) if result.njev is not None else np.nan,
    }
    return row, pd.DataFrame(history)


def integrate_dense_branch(
    *,
    side: str,
    args: argparse.Namespace,
    case: AuditCase,
    c: complex,
) -> Any:
    start = case.y_left if side == "left" else case.y_right
    gamma0 = base.asymptotic_gamma(
        side=side,
        Mach=args.Mach,
        alpha=args.alpha,
        c=c,
    )
    initial = np.asarray([gamma0.real, gamma0.imag, 0.0], dtype=float)
    solution = solve_ivp(
        lambda y, state: rhs_logmodulus(
            y,
            state,
            Mach=args.Mach,
            alpha=args.alpha,
            c=c,
        ),
        (start, case.matching_y),
        initial,
        method=args.method,
        dense_output=True,
        max_step=case.max_step,
        rtol=case.rtol,
        atol=case.atol,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"Dense {side} integration failed: {solution.message}")
    return solution


def reconstruct_core_mode(
    *,
    args: argparse.Namespace,
    case: AuditCase,
    c: complex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = integrate_dense_branch(side="left", args=args, case=case, c=c)
    right = integrate_dense_branch(side="right", args=args, case=case, c=c)

    left_at_match = left.sol(case.matching_y)
    right_at_match = right.sol(case.matching_y)
    ell_shift_right = float(left_at_match[2] - right_at_match[2])

    core_left = max(case.y_left, -args.core_y)
    core_right = min(case.y_right, args.core_y)
    if not (core_left < case.matching_y < core_right):
        raise RuntimeError(
            "The matching point must lie inside the exported common core."
        )

    n_intervals = max(2, int(math.ceil((core_right - core_left) / args.core_dy)))
    y = np.linspace(core_left, core_right, n_intervals + 1)
    if not np.any(np.isclose(y, case.matching_y, atol=1.0e-13)):
        y = np.sort(np.unique(np.append(y, case.matching_y)))

    left_mask = y <= case.matching_y
    right_mask = ~left_mask

    values = np.empty((3, y.size), dtype=float)
    values[:, left_mask] = left.sol(y[left_mask])
    values[:, right_mask] = right.sol(y[right_mask])
    values[2, right_mask] += ell_shift_right

    # Estimate the global maximum log-amplitude from all adaptive solver nodes.
    left_ell_nodes = left.y[2]
    right_ell_nodes = right.y[2] + ell_shift_right
    ell_global_max = float(
        max(
            np.max(left_ell_nodes),
            np.max(right_ell_nodes),
            np.max(values[2]),
        )
    )

    kappa = values[0]
    q = values[1]
    ell = values[2]
    amplitude = np.exp(np.clip(ell - ell_global_max, -745.0, 0.0))

    cumulative = cumulative_trapezoid(q, y, initial=0.0)
    match_index = int(np.argmin(np.abs(y - case.matching_y)))
    phase = cumulative - cumulative[match_index]
    pressure = amplitude * np.exp(1j * phase)

    left_boundary_log_ratio = float(left.y[2, 0] - ell_global_max)
    right_boundary_log_ratio = float(right.y[2, 0] + ell_shift_right - ell_global_max)
    left_boundary_ratio = float(math.exp(max(-745.0, left_boundary_log_ratio)))
    right_boundary_ratio = float(math.exp(max(-745.0, right_boundary_log_ratio)))

    frame = pd.DataFrame(
        {
            "y": y,
            "kappa": kappa,
            "q": q,
            "log_modulus_aligned": ell,
            "modulus": amplitude,
            "phase": phase,
            "p_real": pressure.real,
            "p_imag": pressure.imag,
        }
    )
    diagnostics = {
        "delta_kappa_at_match": float(left_at_match[0] - right_at_match[0]),
        "delta_q_at_match": float(left_at_match[1] - right_at_match[1]),
        "complex_gamma_mismatch_at_match": float(
            math.hypot(
                left_at_match[0] - right_at_match[0],
                left_at_match[1] - right_at_match[1],
            )
        ),
        "log_amplitude_shift_right": ell_shift_right,
        "ell_global_max": ell_global_max,
        "left_boundary_amplitude_ratio": left_boundary_ratio,
        "right_boundary_amplitude_ratio": right_boundary_ratio,
        "left_boundary_log_ratio": left_boundary_log_ratio,
        "right_boundary_log_ratio": right_boundary_log_ratio,
        "left_internal_steps": int(left.t.size),
        "right_internal_steps": int(right.t.size),
        "left_nfev": int(left.nfev),
        "right_nfev": int(right.nfev),
    }
    return frame, diagnostics


def finite_difference_residual(
    frame: pd.DataFrame,
    *,
    args: argparse.Namespace,
    c: complex,
    matching_y: float,
) -> dict[str, float]:
    y = frame["y"].to_numpy(float)
    p = frame["p_real"].to_numpy(float) + 1j * frame["p_imag"].to_numpy(float)
    p_y = np.gradient(p, y, edge_order=2)
    p_yy = np.gradient(p_y, y, edge_order=2)

    U, Up = flow(y)
    residual = (
        p_yy
        - (2.0 * Up / (U - c)) * p_y
        - args.alpha * args.alpha
        * (1.0 - args.Mach * args.Mach * (U - c) ** 2)
        * p
    )
    scale = (
        np.abs(p_yy)
        + np.abs((2.0 * Up / (U - c)) * p_y)
        + np.abs(
            args.alpha * args.alpha
            * (1.0 - args.Mach * args.Mach * (U - c) ** 2)
            * p
        )
        + 1.0e-12
    )
    relative = np.abs(residual) / scale

    mask = (
        (np.abs(y) <= args.core_y)
        & (np.abs(y - matching_y) > 1.0)
    )
    significant = mask & (
        np.abs(p) >= 1.0e-3 * np.max(np.abs(p))
    )
    if np.count_nonzero(significant) < 5:
        significant = mask

    return {
        "fd_relative_ode_residual_max_core": float(np.max(relative[mask])),
        "fd_relative_ode_residual_rms_core": float(
            np.sqrt(np.mean(relative[mask] ** 2))
        ),
        "fd_relative_ode_residual_max_significant": float(
            np.max(relative[significant])
        ),
        "fd_relative_ode_residual_rms_significant": float(
            np.sqrt(np.mean(relative[significant] ** 2))
        ),
    }


def align_and_compare(
    mode: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, float]:
    y_base = baseline["y"].to_numpy(float)
    p_base = (
        baseline["p_real"].to_numpy(float)
        + 1j * baseline["p_imag"].to_numpy(float)
    )
    y = mode["y"].to_numpy(float)
    p = mode["p_real"].to_numpy(float) + 1j * mode["p_imag"].to_numpy(float)

    lower = max(np.min(y), np.min(y_base))
    upper = min(np.max(y), np.max(y_base))
    mask = (y_base >= lower) & (y_base <= upper)
    if np.count_nonzero(mask) < 10:
        raise RuntimeError("Insufficient mode overlap.")

    interp = (
        np.interp(y_base, y, p.real)
        + 1j * np.interp(y_base, y, p.imag)
    )
    denominator = np.vdot(interp[mask], interp[mask])
    scale = np.vdot(interp[mask], p_base[mask]) / denominator
    aligned = scale * interp

    return {
        "mode_relative_l2_core_vs_baseline": float(
            np.linalg.norm(aligned[mask] - p_base[mask])
            / np.linalg.norm(p_base[mask])
        ),
        "envelope_relative_l2_core_vs_baseline": float(
            np.linalg.norm(np.abs(aligned[mask]) - np.abs(p_base[mask]))
            / np.linalg.norm(np.abs(p_base[mask]))
        ),
        "alignment_scale_real_vs_baseline": float(scale.real),
        "alignment_scale_imag_vs_baseline": float(scale.imag),
    }


def adaptive_nominal_box(
    *,
    args: argparse.Namespace,
    seed: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    current_c = complex(float(seed[0]), float(seed[1]))
    box = estimate_box(
        Mach=args.Mach,
        alpha=args.alpha,
        c=current_c,
        args=args,
    )
    updates: list[dict[str, Any]] = []

    for iteration in range(1, args.max_box_updates + 1):
        case = AuditCase(
            case_id=f"box_update_{iteration}",
            family="box_update",
            y_left=float(box["y_left"]),
            y_right=float(box["y_right"]),
            matching_y=args.matching_y,
            max_step=args.nominal_max_step,
            rtol=args.nominal_rtol,
            atol=args.nominal_atol,
        )
        root, history = solve_root(
            args=args,
            case=case,
            seed=np.asarray([current_c.real, current_c.imag], dtype=float),
        )
        history.to_csv(
            output_dir / f"box_update_{iteration}_history.csv",
            index=False,
        )
        solved_c = complex(root["cr"], root["ci"])
        proposed = estimate_box(
            Mach=args.Mach,
            alpha=args.alpha,
            c=solved_c,
            args=args,
        )

        left_change = abs(
            proposed["left_extent"] - box["left_extent"]
        ) / max(box["left_extent"], 1.0)
        right_change = abs(
            proposed["right_extent"] - box["right_extent"]
        ) / max(box["right_extent"], 1.0)

        updates.append(
            {
                "iteration": iteration,
                "input_box": {
                    key: value
                    for key, value in box.items()
                    if key not in ("gamma_left", "gamma_right")
                },
                "input_gamma_left": {
                    "real": box["gamma_left"].real,
                    "imag": box["gamma_left"].imag,
                },
                "input_gamma_right": {
                    "real": box["gamma_right"].real,
                    "imag": box["gamma_right"].imag,
                },
                "root": root,
                "proposed_box": {
                    key: value
                    for key, value in proposed.items()
                    if key not in ("gamma_left", "gamma_right")
                },
                "relative_left_change": left_change,
                "relative_right_change": right_change,
            }
        )

        current_c = solved_c
        # Never shrink during the outer adaptation; shrinking is tested later.
        box["left_extent"] = max(box["left_extent"], proposed["left_extent"])
        box["right_extent"] = max(box["right_extent"], proposed["right_extent"])
        box["y_left"] = -box["left_extent"]
        box["y_right"] = box["right_extent"]
        box.update(
            {
                "gamma_left": proposed["gamma_left"],
                "gamma_right": proposed["gamma_right"],
                "mu_left": proposed["mu_left"],
                "mu_right": proposed["mu_right"],
                "left_status": proposed["left_status"],
                "right_status": proposed["right_status"],
            }
        )

        if (
            left_change <= args.box_update_relative_tolerance
            and right_change <= args.box_update_relative_tolerance
        ):
            break

    (output_dir / "adaptive_box_history.json").write_text(
        json.dumps(json_safe(updates), indent=2),
        encoding="utf-8",
    )
    return box, np.asarray([current_c.real, current_c.imag], dtype=float)


def build_cases(
    *,
    args: argparse.Namespace,
    box: dict[str, Any],
) -> list[AuditCase]:
    left_extent = float(box["left_extent"])
    right_extent = float(box["right_extent"])
    cases: list[AuditCase] = [
        AuditCase(
            case_id="baseline_fine",
            family="baseline",
            y_left=-1.25 * left_extent,
            y_right=1.25 * right_extent,
            matching_y=args.matching_y,
            max_step=args.fine_max_step,
            rtol=args.fine_rtol,
            atol=args.fine_atol,
        ),
        AuditCase(
            case_id="nominal",
            family="nominal",
            y_left=-left_extent,
            y_right=right_extent,
            matching_y=args.matching_y,
            max_step=args.nominal_max_step,
            rtol=args.nominal_rtol,
            atol=args.nominal_atol,
        ),
    ]
    if args.suite == "smoke":
        return cases

    box_scales = args.box_scales
    if args.suite == "quick":
        box_scales = [0.75, 1.0, 1.25]
    for scale in box_scales:
        cases.append(
            AuditCase(
                case_id=f"box_x{str(scale).replace('.', 'p')}",
                family="box",
                y_left=-scale * left_extent,
                y_right=scale * right_extent,
                matching_y=args.matching_y,
                max_step=args.nominal_max_step,
                rtol=args.nominal_rtol,
                atol=args.nominal_atol,
            )
        )

    matching_values = args.matching_values
    if args.suite == "quick":
        matching_values = [0.0, 1.0, 2.0]
    for matching_y in matching_values:
        if np.isclose(matching_y, args.matching_y):
            continue
        cases.append(
            AuditCase(
                case_id=f"match_y{str(matching_y).replace('.', 'p')}",
                family="matching",
                y_left=-left_extent,
                y_right=right_extent,
                matching_y=matching_y,
                max_step=args.nominal_max_step,
                rtol=args.nominal_rtol,
                atol=args.nominal_atol,
            )
        )

    step_values = args.max_step_values
    if args.suite == "quick":
        step_values = [1.0]
    for max_step in step_values:
        if np.isclose(max_step, args.nominal_max_step):
            continue
        cases.append(
            AuditCase(
                case_id=f"step_{str(max_step).replace('.', 'p')}",
                family="max_step",
                y_left=-left_extent,
                y_right=right_extent,
                matching_y=args.matching_y,
                max_step=max_step,
                rtol=args.nominal_rtol,
                atol=args.nominal_atol,
            )
        )

    rtol_values = args.rtol_values
    if args.suite == "quick":
        rtol_values = [1.0e-8]
    for rtol in rtol_values:
        if np.isclose(rtol, args.nominal_rtol, rtol=1.0e-12, atol=0.0):
            continue
        cases.append(
            AuditCase(
                case_id=f"rtol_{rtol:.0e}".replace("-", "m"),
                family="tolerance",
                y_left=-left_extent,
                y_right=right_extent,
                matching_y=args.matching_y,
                max_step=args.nominal_max_step,
                rtol=rtol,
                atol=rtol * 1.0e-2,
            )
        )

    unique: list[AuditCase] = []
    seen: set[tuple[Any, ...]] = set()
    for case in cases:
        key = (
            round(case.y_left, 10),
            round(case.y_right, 10),
            round(case.matching_y, 10),
            case.max_step,
            case.rtol,
            case.atol,
        )
        if key not in seen:
            seen.add(key)
            unique.append(case)
    return unique


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    baseline = summary.loc[summary["case_id"] == "baseline_fine"].iloc[0]

    for family, x_column in (
        ("box", "box_scale"),
        ("matching", "matching_y"),
        ("max_step", "max_step"),
        ("tolerance", "rtol"),
    ):
        data = summary[summary["family"] == family].copy()
        if data.empty:
            continue
        data = data.sort_values(x_column)

        figure = plt.figure(figsize=(8.5, 5.5))
        plt.plot(
            data[x_column],
            np.maximum(np.abs(data["cr"] - baseline["cr"]), 1.0e-18),
            marker="o",
            label=r"$|c_r-c_{r,\mathrm{base}}|$",
        )
        plt.plot(
            data[x_column],
            np.maximum(np.abs(data["ci"] - baseline["ci"]), 1.0e-18),
            marker="o",
            label=r"$|c_i-c_{i,\mathrm{base}}|$",
        )
        plt.yscale("log")
        plt.xlabel(x_column)
        plt.ylabel("Absolute difference")
        plt.title(f"Eigenvalue convergence — {family}")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"eigenvalue_{family}.png", dpi=220)
        figure.savefig(output_dir / f"eigenvalue_{family}.pdf")
        plt.close(figure)

        figure = plt.figure(figsize=(8.5, 5.5))
        plt.plot(
            data[x_column],
            np.maximum(data["mode_relative_l2_core_vs_baseline"], 1.0e-18),
            marker="o",
            label="complex pressure",
        )
        plt.plot(
            data[x_column],
            np.maximum(data["envelope_relative_l2_core_vs_baseline"], 1.0e-18),
            marker="o",
            label="envelope",
        )
        plt.yscale("log")
        plt.xlabel(x_column)
        plt.ylabel("Relative L2 difference")
        plt.title(f"Mode convergence — {family}")
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
    repo = args.repo.expanduser().resolve()

    if args.output_dir is None:
        output_dir = (
            repo
            / "classic_supersonic/reproducibility/results"
            / f"adaptive_logmodulus_audit_{args.case_id}"
        )
    else:
        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = repo / output_dir
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if (args.seed_cr is None) != (args.seed_ci is None):
        raise ValueError("--seed-cr and --seed-ci must be provided together.")

    seed_provenance: dict[str, Any]
    if args.seed_cr is None:
        seed_provenance = load_seed_from_tables(
            repo=repo,
            Mach=args.Mach,
            alpha=args.alpha,
            spectral_csv=args.spectral_csv,
            allow_nearest=args.allow_nearest_seed,
            nearest_max_distance=args.nearest_alpha_max_distance,
        )
        seed_cr = float(seed_provenance["cr"])
        seed_ci = float(seed_provenance["ci"])
    else:
        seed_cr = float(args.seed_cr)
        seed_ci = float(args.seed_ci)
        seed_provenance = {
            "source_kind": "manual",
            "cr": seed_cr,
            "ci": seed_ci,
        }

    if (args.reference_cr is None) != (args.reference_ci is None):
        raise ValueError(
            "--reference-cr and --reference-ci must be provided together."
        )
    if args.reference_cr is not None:
        reference_cr = float(args.reference_cr)
        reference_ci = float(args.reference_ci)
        reference_kind = "manual"
    elif seed_provenance.get("source_kind") == "exact":
        reference_cr = seed_cr
        reference_ci = seed_ci
        reference_kind = "exact_spectral_row"
    else:
        reference_cr = math.nan
        reference_ci = math.nan
        reference_kind = "unavailable_nearest_seed_only"

    print("=== Adaptive difficult-point convergence audit ===")
    print(f"case_id          : {args.case_id}")
    print(f"M                : {args.Mach}")
    print(f"alpha            : {args.alpha}")
    print(f"seed             : {seed_cr:.15g} + {seed_ci:.15g} i")
    print(f"seed source      : {seed_provenance.get('source_kind')}")
    print(f"seed path        : {seed_provenance.get('source_path', 'manual')}")
    print(f"suite            : {args.suite}")
    print(f"tail tolerance   : {args.tail_tolerance}")
    print(f"output           : {output_dir}")
    print()

    initial_seed = np.asarray([seed_cr, seed_ci], dtype=float)
    adaptive_box, adapted_seed = adaptive_nominal_box(
        args=args,
        seed=initial_seed,
        output_dir=output_dir,
    )

    print("\n=== Adapted nominal box ===")
    print(f"mu_left          : {adaptive_box['mu_left']:.12e}")
    print(f"mu_right         : {adaptive_box['mu_right']:.12e}")
    print(f"left extent      : {adaptive_box['left_extent']:.6f}")
    print(f"right extent     : {adaptive_box['right_extent']:.6f}")
    print(f"left status      : {adaptive_box['left_status']}")
    print(f"right status     : {adaptive_box['right_status']}")
    print(
        f"adapted seed     : {adapted_seed[0]:.15g} + "
        f"{adapted_seed[1]:.15g} i"
    )

    cases = build_cases(args=args, box=adaptive_box)
    print(f"number audit cases: {len(cases)}")

    rows: list[dict[str, Any]] = []
    modes: dict[str, pd.DataFrame] = {}

    for index, case in enumerate(cases, start=1):
        print(f"\n=== CASE {index}/{len(cases)}: {case.case_id} ===")
        print(asdict(case))
        root, history = solve_root(
            args=args,
            case=case,
            seed=adapted_seed,
        )
        history.to_csv(
            output_dir / f"history_{case.case_id}.csv",
            index=False,
        )
        c = complex(root["cr"], root["ci"])
        mode, mode_diagnostics = reconstruct_core_mode(
            args=args,
            case=case,
            c=c,
        )
        mode.to_csv(output_dir / f"mode_{case.case_id}.csv", index=False)
        fd = finite_difference_residual(
            mode,
            args=args,
            c=c,
            matching_y=case.matching_y,
        )

        row = {
            **asdict(case),
            **root,
            **mode_diagnostics,
            **fd,
            "delta_cr_vs_reference": root["cr"] - reference_cr,
            "delta_ci_vs_reference": root["ci"] - reference_ci,
            "delta_omega_i_vs_reference": (
                args.alpha * (root["ci"] - reference_ci)
            ),
            "left_extent": abs(case.y_left),
            "right_extent": case.y_right,
            "box_scale": (
                abs(case.y_left) / adaptive_box["left_extent"]
                if case.family == "box"
                else np.nan
            ),
            "tails_accepted": bool(
                mode_diagnostics["left_boundary_amplitude_ratio"]
                <= args.tail_tolerance
                and mode_diagnostics["right_boundary_amplitude_ratio"]
                <= args.tail_tolerance
            ),
        }
        rows.append(row)
        modes[case.case_id] = mode

        print(
            f"c={root['cr']:.15g}+{root['ci']:.15g}i "
            f"||F||={root['root_residual_norm']:.3e} "
            f"tails=({mode_diagnostics['left_boundary_amplitude_ratio']:.3e}, "
            f"{mode_diagnostics['right_boundary_amplitude_ratio']:.3e})"
        )

    baseline = modes["baseline_fine"]
    for row in rows:
        row.update(align_and_compare(modes[row["case_id"]], baseline))

    summary = pd.DataFrame(rows)
    baseline_row = summary[
        summary["case_id"] == "baseline_fine"
    ].iloc[0]
    summary["delta_cr_vs_baseline"] = summary["cr"] - baseline_row["cr"]
    summary["delta_ci_vs_baseline"] = summary["ci"] - baseline_row["ci"]
    summary["delta_omega_i_vs_baseline"] = (
        summary["omega_i"] - baseline_row["omega_i"]
    )
    summary.to_csv(output_dir / "convergence_summary.csv", index=False)
    plot_summary(summary, output_dir)

    report = {
        "case": {
            "case_id": args.case_id,
            "Mach": args.Mach,
            "alpha": args.alpha,
        },
        "seed_provenance": seed_provenance,
        "reference": {
            "kind": reference_kind,
            "cr": reference_cr,
            "ci": reference_ci,
            "omega_i": args.alpha * reference_ci,
        },
        "adaptive_box": {
            key: value
            for key, value in adaptive_box.items()
            if key not in ("gamma_left", "gamma_right")
        },
        "adaptive_gamma_left": adaptive_box["gamma_left"],
        "adaptive_gamma_right": adaptive_box["gamma_right"],
        "baseline": baseline_row.to_dict(),
        "all_roots_accepted": bool(summary["root_accepted"].all()),
        "all_nominal_or_larger_tails_accepted": bool(
            summary[
                (summary["family"] != "box")
                | (summary["box_scale"] >= 1.0)
            ]["tails_accepted"].all()
        ),
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
        "rows": rows,
    }
    (output_dir / "convergence_summary.json").write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )

    columns = [
        "case_id",
        "family",
        "left_extent",
        "right_extent",
        "matching_y",
        "max_step",
        "rtol",
        "cr",
        "ci",
        "root_residual_norm",
        "left_boundary_amplitude_ratio",
        "right_boundary_amplitude_ratio",
        "tails_accepted",
        "delta_cr_vs_baseline",
        "delta_ci_vs_baseline",
        "mode_relative_l2_core_vs_baseline",
        "envelope_relative_l2_core_vs_baseline",
        "fd_relative_ode_residual_rms_significant",
    ]
    print("\n=== SUMMARY ===")
    print(summary[columns].to_string(index=False))
    print(f"\nWritten to: {output_dir}")

    return 0 if summary["root_accepted"].all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
