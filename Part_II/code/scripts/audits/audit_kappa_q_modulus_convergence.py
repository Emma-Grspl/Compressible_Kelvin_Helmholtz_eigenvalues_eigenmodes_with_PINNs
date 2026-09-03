#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import scripts.evaluation.search_cr_ci_kappa_q_modulus as eig
import scripts.evaluation.test_kappa_q_modulus_reconstruction as base


DEFAULT_CR = 0.3334594428308841
DEFAULT_CI = 0.0426275027406403


@dataclass(frozen=True)
class AuditCase:
    case_id: str
    family: str
    Ly: float
    matching_y: float
    max_step: float
    rtol: float
    atol: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convergence audit for the joint (c_r,c_i,kappa,q,|p|) "
            "supersonic solver at fixed Mach and alpha."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--Mach", type=float, default=1.4)
    parser.add_argument("--alpha", type=float, default=0.18125)
    parser.add_argument("--seed-cr", type=float, default=DEFAULT_CR)
    parser.add_argument("--seed-ci", type=float, default=DEFAULT_CI)
    parser.add_argument(
        "--suite",
        choices=("quick", "full"),
        default="quick",
    )
    parser.add_argument("--output-dy", type=float, default=0.025)
    parser.add_argument("--core-y", type=float, default=20.0)
    parser.add_argument("--method", choices=("DOP853", "RK45"), default="DOP853")
    parser.add_argument("--max-nfev", type=int, default=30)
    parser.add_argument("--diff-step", type=float, default=1.0e-5)
    parser.add_argument("--accept-residual", type=float, default=1.0e-8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def build_cases(suite: str) -> list[AuditCase]:
    cases: list[AuditCase] = [
        AuditCase(
            case_id="baseline_fine",
            family="baseline",
            Ly=500.0,
            matching_y=1.0,
            max_step=0.125,
            rtol=1.0e-11,
            atol=1.0e-13,
        ),
        AuditCase(
            case_id="nominal",
            family="nominal",
            Ly=500.0,
            matching_y=1.0,
            max_step=0.25,
            rtol=1.0e-10,
            atol=1.0e-12,
        ),
        AuditCase(
            case_id="box_L250",
            family="box",
            Ly=250.0,
            matching_y=1.0,
            max_step=0.25,
            rtol=1.0e-10,
            atol=1.0e-12,
        ),
        AuditCase(
            case_id="box_L750",
            family="box",
            Ly=750.0,
            matching_y=1.0,
            max_step=0.25,
            rtol=1.0e-10,
            atol=1.0e-12,
        ),
        AuditCase(
            case_id="match_y0",
            family="matching",
            Ly=500.0,
            matching_y=0.0,
            max_step=0.25,
            rtol=1.0e-10,
            atol=1.0e-12,
        ),
        AuditCase(
            case_id="match_y2",
            family="matching",
            Ly=500.0,
            matching_y=2.0,
            max_step=0.25,
            rtol=1.0e-10,
            atol=1.0e-12,
        ),
        AuditCase(
            case_id="step_0p5",
            family="max_step",
            Ly=500.0,
            matching_y=1.0,
            max_step=0.5,
            rtol=1.0e-10,
            atol=1.0e-12,
        ),
        AuditCase(
            case_id="tol_r1e9",
            family="tolerance",
            Ly=500.0,
            matching_y=1.0,
            max_step=0.25,
            rtol=1.0e-9,
            atol=1.0e-11,
        ),
    ]

    if suite == "full":
        cases.extend(
            [
                AuditCase(
                    case_id="box_L375",
                    family="box",
                    Ly=375.0,
                    matching_y=1.0,
                    max_step=0.25,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                ),
                AuditCase(
                    case_id="box_L625",
                    family="box",
                    Ly=625.0,
                    matching_y=1.0,
                    max_step=0.25,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                ),
                AuditCase(
                    case_id="match_y0p5",
                    family="matching",
                    Ly=500.0,
                    matching_y=0.5,
                    max_step=0.25,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                ),
                AuditCase(
                    case_id="match_y1p5",
                    family="matching",
                    Ly=500.0,
                    matching_y=1.5,
                    max_step=0.25,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                ),
                AuditCase(
                    case_id="step_1p0",
                    family="max_step",
                    Ly=500.0,
                    matching_y=1.0,
                    max_step=1.0,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                ),
                AuditCase(
                    case_id="tol_r1e8",
                    family="tolerance",
                    Ly=500.0,
                    matching_y=1.0,
                    max_step=0.25,
                    rtol=1.0e-8,
                    atol=1.0e-10,
                ),
            ]
        )

    seen: set[tuple[float, float, float, float, float]] = set()
    unique: list[AuditCase] = []
    for case in cases:
        key = (
            case.Ly,
            case.matching_y,
            case.max_step,
            case.rtol,
            case.atol,
        )
        if key not in seen:
            unique.append(case)
            seen.add(key)
    return unique


def objective_args(
    args: argparse.Namespace,
    case: AuditCase,
) -> SimpleNamespace:
    return SimpleNamespace(
        Mach=args.Mach,
        alpha=args.alpha,
        Ly=case.Ly,
        matching_y=case.matching_y,
        max_step=case.max_step,
        rtol=case.rtol,
        atol=case.atol,
        method=args.method,
    )


def solve_case(
    args: argparse.Namespace,
    case: AuditCase,
    seed: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    local_args = objective_args(args, case)
    history: list[dict[str, Any]] = []
    objective = eig.MatchingObjective(
        args=local_args,
        history=history,
    )

    result = least_squares(
        objective,
        x0=np.asarray(seed, dtype=float),
        bounds=(
            np.asarray([0.20, 1.0e-5], dtype=float),
            np.asarray([0.50, 0.15], dtype=float),
        ),
        method="trf",
        jac="2-point",
        diff_step=args.diff_step,
        x_scale=np.asarray([0.05, 0.02], dtype=float),
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=args.max_nfev,
        verbose=0,
    )

    c = complex(float(result.x[0]), float(result.x[1]))
    residual = objective(result.x)
    residual_norm = float(np.linalg.norm(residual))

    left = base.integrate_branch(
        side="left",
        Mach=args.Mach,
        alpha=args.alpha,
        c=c,
        Ly=case.Ly,
        matching_y=case.matching_y,
        output_dy=args.output_dy,
        max_step=case.max_step,
        rtol=case.rtol,
        atol=case.atol,
        method=args.method,
    )
    right = base.integrate_branch(
        side="right",
        Mach=args.Mach,
        alpha=args.alpha,
        c=c,
        Ly=case.Ly,
        matching_y=case.matching_y,
        output_dy=args.output_dy,
        max_step=case.max_step,
        rtol=case.rtol,
        atol=case.atol,
        method=args.method,
    )

    mode, matching = base.reconstruct_mode(left, right)
    fd = base.numerical_residuals(
        mode,
        Mach=args.Mach,
        alpha=args.alpha,
        c=c,
        matching_y=case.matching_y,
        core_y=args.core_y,
    )

    row: dict[str, Any] = {
        **asdict(case),
        "cr": c.real,
        "ci": c.imag,
        "omega_i": args.alpha * c.imag,
        "delta_cr_vs_seed": c.real - args.seed_cr,
        "delta_ci_vs_seed": c.imag - args.seed_ci,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_nfev": int(result.nfev),
        "optimizer_njev": (
            int(result.njev) if result.njev is not None else np.nan
        ),
        "root_delta_kappa": float(residual[0]),
        "root_delta_q": float(residual[1]),
        "root_residual_norm": residual_norm,
        "root_accepted": residual_norm <= args.accept_residual,
        "amplitude_scale_right": matching["amplitude_scale_right"],
        "global_modulus_normalization": (
            matching["global_modulus_normalization"]
        ),
        **fd,
    }
    return row, mode


def interpolate_complex(
    source_y: np.ndarray,
    source_p: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    return (
        np.interp(target_y, source_y, source_p.real)
        + 1j * np.interp(target_y, source_y, source_p.imag)
    )


def compare_mode_to_baseline(
    mode: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    core_y: float,
) -> dict[str, float]:
    y_base = baseline["y"].to_numpy(float)
    p_base = (
        baseline["p_real"].to_numpy(float)
        + 1j * baseline["p_imag"].to_numpy(float)
    )

    y = mode["y"].to_numpy(float)
    p = (
        mode["p_real"].to_numpy(float)
        + 1j * mode["p_imag"].to_numpy(float)
    )

    mask = (
        (np.abs(y_base) <= core_y)
        & (y_base >= np.min(y))
        & (y_base <= np.max(y))
    )
    if np.count_nonzero(mask) < 8:
        raise RuntimeError("Insufficient common core for mode comparison.")

    p_interp = interpolate_complex(y, p, y_base)
    denominator = np.vdot(p_interp[mask], p_interp[mask])
    if abs(denominator) < 1.0e-30:
        raise RuntimeError("Mode is zero on comparison core.")

    scale = np.vdot(p_interp[mask], p_base[mask]) / denominator
    aligned = scale * p_interp

    p_error = (
        np.linalg.norm(aligned[mask] - p_base[mask])
        / np.linalg.norm(p_base[mask])
    )
    a_error = (
        np.linalg.norm(np.abs(aligned[mask]) - np.abs(p_base[mask]))
        / np.linalg.norm(np.abs(p_base[mask]))
    )

    return {
        "mode_relative_l2_core_vs_baseline": float(p_error),
        "envelope_relative_l2_core_vs_baseline": float(a_error),
        "alignment_scale_real_vs_baseline": float(scale.real),
        "alignment_scale_imag_vs_baseline": float(scale.imag),
    }


def plot_family(
    frame: pd.DataFrame,
    family: str,
    x_column: str,
    output_dir: Path,
) -> None:
    data = frame[frame["family"] == family].copy()
    if data.empty:
        return
    data = data.sort_values(x_column)

    figure = plt.figure(figsize=(8.5, 5.5))
    plt.plot(
        data[x_column],
        np.abs(data["cr"] - frame.loc[
            frame["case_id"] == "baseline_fine", "cr"
        ].iloc[0]),
        marker="o",
        label=r"$|c_r-c_{r,\mathrm{base}}|$",
    )
    plt.plot(
        data[x_column],
        np.abs(data["ci"] - frame.loc[
            frame["case_id"] == "baseline_fine", "ci"
        ].iloc[0]),
        marker="o",
        label=r"$|c_i-c_{i,\mathrm{base}}|$",
    )
    plt.yscale("log")
    plt.xlabel(x_column)
    plt.ylabel("Absolute eigenvalue difference")
    plt.title(f"Eigenvalue convergence — {family}")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / f"eigenvalue_convergence_{family}.png",
        dpi=220,
    )
    figure.savefig(
        output_dir / f"eigenvalue_convergence_{family}.pdf"
    )
    plt.close(figure)

    figure = plt.figure(figsize=(8.5, 5.5))
    plt.plot(
        data[x_column],
        data["mode_relative_l2_core_vs_baseline"],
        marker="o",
        label="complex mode",
    )
    plt.plot(
        data[x_column],
        data["envelope_relative_l2_core_vs_baseline"],
        marker="o",
        label="envelope",
    )
    plt.yscale("log")
    plt.xlabel(x_column)
    plt.ylabel("Relative L2 error on common core")
    plt.title(f"Mode convergence — {family}")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / f"mode_convergence_{family}.png",
        dpi=220,
    )
    figure.savefig(
        output_dir / f"mode_convergence_{family}.pdf"
    )
    plt.close(figure)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
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
            / f"kappa_q_modulus_convergence_{args.suite}_M140_a018125"
        )
    else:
        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = repo / output_dir
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args.suite)
    print("=== Kappa-q-modulus convergence audit ===")
    print(f"suite       : {args.suite}")
    print(f"number cases: {len(cases)}")
    print(f"M           : {args.Mach}")
    print(f"alpha       : {args.alpha}")
    print(f"seed        : {args.seed_cr} + {args.seed_ci} i")
    print(f"output      : {output_dir}")
    print()

    rows: list[dict[str, Any]] = []
    modes: dict[str, pd.DataFrame] = {}

    seed = np.asarray([args.seed_cr, args.seed_ci], dtype=float)
    for index, case in enumerate(cases, start=1):
        print(
            f"\n=== CASE {index}/{len(cases)}: {case.case_id} ===",
            flush=True,
        )
        print(asdict(case), flush=True)
        row, mode = solve_case(args, case, seed)
        rows.append(row)
        modes[case.case_id] = mode
        mode.to_csv(
            output_dir / f"mode_{case.case_id}.csv",
            index=False,
        )
        print(
            f"result: c={row['cr']:.15g}+{row['ci']:.15g}i, "
            f"||F||={row['root_residual_norm']:.3e}",
            flush=True,
        )

    baseline = modes["baseline_fine"]
    for row in rows:
        comparison = compare_mode_to_baseline(
            modes[row["case_id"]],
            baseline,
            core_y=args.core_y,
        )
        row.update(comparison)

    summary = pd.DataFrame(rows)
    baseline_row = summary[
        summary["case_id"] == "baseline_fine"
    ].iloc[0]

    summary["delta_cr_vs_baseline"] = (
        summary["cr"] - baseline_row["cr"]
    )
    summary["delta_ci_vs_baseline"] = (
        summary["ci"] - baseline_row["ci"]
    )
    summary["delta_omega_i_vs_baseline"] = (
        summary["omega_i"] - baseline_row["omega_i"]
    )

    summary.to_csv(
        output_dir / "convergence_summary.csv",
        index=False,
    )

    for family, x_column in (
        ("box", "Ly"),
        ("matching", "matching_y"),
        ("max_step", "max_step"),
        ("tolerance", "rtol"),
    ):
        plot_family(summary, family, x_column, output_dir)

    report = {
        "parameters": {
            "Mach": args.Mach,
            "alpha": args.alpha,
            "seed_cr": args.seed_cr,
            "seed_ci": args.seed_ci,
            "suite": args.suite,
            "output_dy": args.output_dy,
            "core_y": args.core_y,
            "method": args.method,
            "accept_residual": args.accept_residual,
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
        "cases": rows,
    }
    (output_dir / "convergence_summary.json").write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")
    display_columns = [
        "case_id",
        "family",
        "Ly",
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
        "fd_relative_ode_residual_rms_significant",
    ]
    print(summary[display_columns].to_string(index=False))
    print(f"\nWritten to: {output_dir}")

    return 0 if summary["root_accepted"].all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
