#!/usr/bin/env python3
"""Build the publication figure for subsonic classical shooting convergence.

The numerical data are produced with the repository's validated
``Mstab17SubsonicSolver``.  This script does not reimplement the Riccati
equation, the asymptotic conditions, the branch search, or the matching
criteria.  It only fixes the solver's adaptive box to prescribed half-lengths,
reconstructs the already-defined physical fields, aligns each candidate mode
to the largest-domain mode with one pressure-derived complex factor, computes
convergence metrics, and renders the article figure.

Default representative points:

* interior: ``M=0.5, alpha=0.5``;
* long wave: ``M=0.49, alpha=0.09`` from the canonical hybrid growth map;
* near neutral: ``M=0.49, alpha=0.86`` from the canonical near-neutral audit.

Examples
--------
Compute tables and render the figure, reusing an existing table when present::

    python code/plots/scripts/pinn_subsonic/plot_subsonic_classical_convergence_article.py

Explicitly recompute the classical solutions::

    python code/plots/scripts/pinn_subsonic/plot_subsonic_classical_convergence_article.py --force-recompute

Render only from the existing detailed table::

    python code/plots/scripts/pinn_subsonic/plot_subsonic_classical_convergence_article.py --plot-only
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "code"))

from src.scripts.classical.solve_mstab17_subsonic_solver import (
    Mstab17SubsonicResult,
    Mstab17SubsonicSolver,
)


DEFAULT_FIGURE_DIR = (
    ROOT_DIR / "assets/classic_subsonic/article/convergence/figures"
)
DEFAULT_TABLE_DIR = ROOT_DIR / "assets/classic_subsonic/article/convergence/tables"
DEFAULT_DATA_OUTPUT = DEFAULT_TABLE_DIR / "Table_classical_subsonic_convergence.csv"
DEFAULT_SUMMARY_OUTPUT = (
    DEFAULT_TABLE_DIR / "Table_classical_subsonic_convergence_summary.csv"
)
FIGURE_STEM = "Fig_classical_subsonic_convergence"

FIELDS = ("p", "rho", "u", "v")
DEFAULT_LY_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
MATCHING_Y = 1.0
INTEGRATION_RTOL = 1.0e-10
INTEGRATION_ATOL = 1.0e-12
ROOT_TOLERANCE = 1.0e-4
N_SCAN = 121
CI_MIN = 1.0e-3
CI_MAX = 1.0
COMMON_GRID_SIZE = 4001

# The residual threshold is already used by the classical hybrid pipeline.
# The three error thresholds are explicit diagnostic criteria proposed for
# this article figure because no official joint spectral-modal criterion was
# found in the current classical solver.
RESIDUAL_THRESHOLD = 5.0e-4
ABS_CI_THRESHOLD = 1.0e-6
P_REL_L2_THRESHOLD = 1.0e-3
MODAL_MEAN_THRESHOLD = 1.0e-2

REQUIRED_COLUMNS = [
    "point_id",
    "point_category",
    "Mach",
    "alpha",
    "eta",
    "Ly",
    "Ly_production",
    "matching_point",
    "integration_rtol",
    "integration_atol",
    "root_tolerance",
    "ci",
    "omega_i",
    "ci_reference",
    "omega_i_reference",
    "abs_ci_error",
    "abs_omega_error",
    "shooting_residual_real",
    "shooting_residual_imag",
    "shooting_residual_abs",
    "p_rel_l2",
    "rho_rel_l2",
    "u_rel_l2",
    "v_rel_l2",
    "modal_rel_l2_mean",
    "p_overlap",
    "p_overlap_defect",
    "runtime_seconds",
    "converged",
    "branch_check",
    "notes",
]


@dataclass(frozen=True)
class ConvergencePoint:
    point_id: str
    category: str
    Mach: float
    alpha: float
    expected_ci: float
    source: str
    note: str

    @property
    def eta(self) -> float:
        return self.alpha / np.sqrt(1.0 - self.Mach**2)

    @property
    def production_Ly(self) -> float:
        solver = Mstab17SubsonicSolver(alpha=self.alpha, Mach=self.Mach)
        return solver.estimate_y_limit(self.expected_ci)


POINTS = (
    ConvergencePoint(
        point_id="interior",
        category="interior atlas point",
        Mach=0.5,
        alpha=0.5,
        expected_ci=0.2673341164147236,
        source=(
            "assets/classic_subsonic/csv/convergence_box_resolution/"
            "article_test_a050_M050_exact_solver/"
            "Table_subsonic_classical_robust_reference_points.csv"
        ),
        note="Interior point required by the article protocol.",
    ),
    ConvergencePoint(
        point_id="long_wave",
        category="long wavelength",
        Mach=0.49,
        alpha=0.09,
        expected_ci=0.688987011514646,
        source="assets/classic_subsonic/csv/data/Table_subsonic_hybrid_growth_map.csv",
        note="Canonical primary-branch point with a complete Mstab17 reconstruction.",
    ),
    ConvergencePoint(
        point_id="near_neutral",
        category="near neutral boundary",
        Mach=0.49,
        alpha=0.86,
        expected_ci=0.0077519783829051,
        source="assets/classic_subsonic/csv/data/Table_subsonic_hybrid_growth_map.csv",
        note=(
            "Canonical secondary near-neutral point; absolute ci errors are used "
            "because ci is close to zero."
        ),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument(
        "--Ly-factors",
        default=",".join(f"{value:g}" for value in DEFAULT_LY_FACTORS),
        help="Comma-separated factors applied to each point's adaptive production Ly.",
    )
    parser.add_argument("--n-scan", type=int, default=N_SCAN)
    parser.add_argument("--rtol", type=float, default=INTEGRATION_RTOL)
    parser.add_argument("--atol", type=float, default=INTEGRATION_ATOL)
    parser.add_argument("--common-grid-size", type=int, default=COMMON_GRID_SIZE)
    parser.add_argument("--force-recompute", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compute-only", action="store_true")
    mode.add_argument("--plot-only", action="store_true")
    return parser


def parse_ly_factors(text: str) -> list[float]:
    values = sorted({float(value.strip()) for value in text.split(",") if value.strip()})
    if len(values) < 3 or any(value <= 0.0 for value in values):
        raise ValueError("At least three positive Ly factors are required.")
    if not any(np.isclose(value, 1.0) for value in values):
        raise ValueError("--Ly-factors must include 1.0 for the production box.")
    return values


def _interp_component(
    solution: Any,
    component: int,
    coordinate: float,
) -> float:
    time = np.asarray(solution.t, dtype=float)
    values = np.asarray(solution.y[component], dtype=float)
    if time[0] > time[-1]:
        time = time[::-1]
        values = values[::-1]
    return float(np.interp(coordinate, time, values))


def reconstruct_fields(
    solver: Mstab17SubsonicSolver,
    result: Mstab17SubsonicResult,
) -> dict[str, np.ndarray]:
    """Reconstruct p, rho, u and v with the validated Mstab17 trajectories."""
    left, right, _ = solver.get_trajectories(
        result.ci,
        ln_p_start_right=result.ln_p_start_right,
    )
    if not left.success or not right.success:
        raise RuntimeError("The left or right modal trajectory did not converge.")

    y_left = np.asarray(left.t, dtype=float)
    y_right = np.asarray(right.t, dtype=float)
    k_left, q_left, logp_left, phase_left = np.asarray(left.y)
    k_right, q_right, logp_right, phase_right = np.asarray(right.y)

    phase_shift = _interp_component(left, 3, 0.0) - _interp_component(right, 3, 0.0)
    p_left = np.exp(logp_left) * np.exp(1j * phase_left)
    p_right = np.exp(logp_right) * np.exp(1j * (phase_right + phase_shift))
    gamma_left = k_left + 1j * q_left
    gamma_right = k_right + 1j * q_right

    left_mask = y_left < 0.0
    y = np.concatenate((y_left[left_mask], y_right[::-1]))
    p = np.concatenate((p_left[left_mask], p_right[::-1]))
    gamma = np.concatenate((gamma_left[left_mask], gamma_right[::-1]))
    if len(y) < 2 or not np.all(np.diff(y) > 0.0):
        raise RuntimeError("The reconstructed physical grid is not strictly increasing.")

    p_y = gamma * p
    c = 1j * result.ci
    base_velocity = np.tanh(y)
    base_velocity_y = 1.0 / np.cosh(y) ** 2
    denominator = base_velocity - c
    i_alpha = 1j * solver.alpha
    v = -p_y / (i_alpha * denominator)
    u = -(base_velocity_y * v + i_alpha * p) / (i_alpha * denominator)
    rho = solver.Mach**2 * p

    fields = {"y": y, "p": p, "rho": rho, "u": u, "v": v}
    if not all(np.all(np.isfinite(fields[name])) for name in FIELDS):
        raise FloatingPointError("At least one reconstructed modal field is non-finite.")
    return fields


def shooting_residual(
    solver: Mstab17SubsonicSolver,
    result: Mstab17SubsonicResult,
) -> complex:
    left, right, _ = solver.get_trajectories(
        result.ci,
        ln_p_start_right=result.ln_p_start_right,
    )
    gamma_left = complex(
        _interp_component(left, 0, solver.match_y),
        _interp_component(left, 1, solver.match_y),
    )
    gamma_right = complex(
        _interp_component(right, 0, solver.match_y),
        _interp_component(right, 1, solver.match_y),
    )
    return gamma_left - gamma_right


def interpolate_complex(
    y_source: np.ndarray,
    values: np.ndarray,
    y_target: np.ndarray,
) -> np.ndarray:
    return np.interp(y_target, y_source, values.real) + 1j * np.interp(
        y_target, y_source, values.imag
    )


def inner_product(left: np.ndarray, right: np.ndarray, y: np.ndarray) -> complex:
    return complex(np.trapezoid(np.conjugate(left) * right, y))


def relative_l2(candidate: np.ndarray, reference: np.ndarray, y: np.ndarray) -> float:
    numerator = float(np.real(inner_product(candidate - reference, candidate - reference, y)))
    denominator = float(np.real(inner_product(reference, reference, y)))
    if denominator <= 1.0e-300:
        return float("nan")
    return float(np.sqrt(max(numerator, 0.0) / denominator))


def compare_modes(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    common_grid_size: int,
) -> dict[str, float]:
    lower = max(float(candidate["y"].min()), float(reference["y"].min()))
    upper = min(float(candidate["y"].max()), float(reference["y"].max()))
    if not lower < upper:
        raise ValueError("Candidate and reference modal domains do not overlap.")
    y = np.linspace(lower, upper, common_grid_size)

    candidate_i = {
        field: interpolate_complex(candidate["y"], candidate[field], y)
        for field in FIELDS
    }
    reference_i = {
        field: interpolate_complex(reference["y"], reference[field], y)
        for field in FIELDS
    }

    denominator = inner_product(candidate_i["p"], candidate_i["p"], y)
    if abs(denominator) <= 1.0e-300:
        raise ValueError("The candidate pressure mode is degenerate.")
    factor = inner_product(candidate_i["p"], reference_i["p"], y) / denominator
    aligned = {field: factor * candidate_i[field] for field in FIELDS}

    errors = {
        f"{field}_rel_l2": relative_l2(aligned[field], reference_i[field], y)
        for field in FIELDS
    }
    p_candidate_norm = np.sqrt(
        max(float(np.real(inner_product(candidate_i["p"], candidate_i["p"], y))), 0.0)
    )
    p_reference_norm = np.sqrt(
        max(float(np.real(inner_product(reference_i["p"], reference_i["p"], y))), 0.0)
    )
    overlap = abs(inner_product(candidate_i["p"], reference_i["p"], y)) / max(
        p_candidate_norm * p_reference_norm,
        1.0e-300,
    )
    overlap = float(np.clip(overlap, 0.0, 1.0))
    errors.update(
        {
            "modal_rel_l2_mean": float(
                np.nanmean([errors[f"{field}_rel_l2"] for field in FIELDS])
            ),
            "p_overlap": overlap,
            "p_overlap_defect": 1.0 - overlap,
            "alignment_factor_real": float(factor.real),
            "alignment_factor_imag": float(factor.imag),
            "comparison_y_min": lower,
            "comparison_y_max": upper,
            "comparison_grid_size": int(common_grid_size),
        }
    )
    return errors


def branch_check(point: ConvergencePoint, ci: float) -> tuple[bool, str]:
    absolute_distance = abs(ci - point.expected_ci)
    relative_distance = absolute_distance / max(abs(point.expected_ci), 1.0e-12)
    passed = absolute_distance <= 0.04 and relative_distance <= 0.35
    status = (
        "passed_expected_branch"
        if passed
        else (
            f"failed_expected_branch(abs={absolute_distance:.3e},"
            f"rel={relative_distance:.3e})"
        )
    )
    return passed, status


def solve_case(
    point: ConvergencePoint,
    Ly: float,
    *,
    n_scan: int,
    rtol: float,
    atol: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    solver = Mstab17SubsonicSolver(
        alpha=point.alpha,
        Mach=point.Mach,
        match_y=MATCHING_Y,
        rtol=rtol,
        atol=atol,
        min_y_limit=Ly,
        max_y_limit=Ly,
    )
    started = perf_counter()
    result: Mstab17SubsonicResult | None = None
    fields: dict[str, np.ndarray] | None = None
    error_message = ""
    residual = np.nan + 1j * np.nan
    try:
        result = solver.solve(ci_min=CI_MIN, ci_max=CI_MAX, n_scan=n_scan)
        residual = shooting_residual(solver, result)
        fields = reconstruct_fields(solver, result)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
    runtime = perf_counter() - started

    ci = float(result.ci) if result is not None else np.nan
    omega_i = point.alpha * ci if np.isfinite(ci) else np.nan
    branch_passed, branch_status = (
        branch_check(point, ci)
        if np.isfinite(ci)
        else (False, "not_checked_nonfinite_ci")
    )
    solver_success = bool(
        result is not None
        and result.success
        and fields is not None
        and np.isfinite(abs(residual))
    )
    row: dict[str, Any] = {
        "point_id": point.point_id,
        "point_category": point.category,
        "Mach": point.Mach,
        "alpha": point.alpha,
        "eta": point.eta,
        "Ly": Ly,
        "Ly_production": point.production_Ly,
        "matching_point": MATCHING_Y,
        "integration_rtol": rtol,
        "integration_atol": atol,
        "root_tolerance": ROOT_TOLERANCE,
        "n_scan": n_scan,
        "ci": ci,
        "omega_i": omega_i,
        "shooting_residual_real": float(residual.real),
        "shooting_residual_imag": float(residual.imag),
        "shooting_residual_abs": float(abs(residual)),
        "stage2_amplitude_mismatch": (
            float(result.stage2_mismatch) if result is not None else np.nan
        ),
        "runtime_seconds": runtime,
        "solver_success": solver_success,
        "branch_check_passed": branch_passed,
        "branch_check": branch_status,
        "point_source": point.source,
        "notes": point.note if not error_message else f"{point.note} {error_message}",
    }
    return row, fields


def compute_table(
    points: tuple[ConvergencePoint, ...],
    Ly_factors: list[float],
    *,
    n_scan: int,
    rtol: float,
    atol: float,
    common_grid_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for point in points:
        production_Ly = point.production_Ly
        Ly_values = [production_Ly * factor for factor in Ly_factors]
        largest_Ly = max(Ly_values)
        print(
            f"\n[{point.point_id}] M={point.Mach:g}, alpha={point.alpha:g}, "
            f"eta={point.eta:.6f}, expected ci={point.expected_ci:.12g}, "
            f"production Ly={production_Ly:.12g}"
        )
        solved: list[tuple[dict[str, Any], dict[str, np.ndarray] | None]] = []
        for Ly in Ly_values:
            is_reference_candidate = np.isclose(Ly, largest_Ly)
            run_rtol = min(rtol, 1.0e-12) if is_reference_candidate else rtol
            run_atol = min(atol, 1.0e-14) if is_reference_candidate else atol
            print(f"  solving Ly={Ly:g} ...", flush=True)
            row, fields = solve_case(
                point,
                Ly,
                n_scan=n_scan,
                rtol=run_rtol,
                atol=run_atol,
            )
            row["Ly_factor"] = Ly / production_Ly
            row["high_fidelity_tolerances"] = is_reference_candidate
            solved.append((row, fields))
            print(
                f"    ci={row['ci']:.12g} |F|={row['shooting_residual_abs']:.3e} "
                f"success={row['solver_success']} branch={row['branch_check']}"
            )

        valid_references = [
            (row, fields)
            for row, fields in solved
            if row["solver_success"] and row["branch_check_passed"] and fields is not None
        ]
        if not valid_references:
            raise RuntimeError(f"No valid high-fidelity reference for {point.point_id}.")
        reference_row, reference_fields = max(
            valid_references,
            key=lambda item: float(item[0]["Ly"]),
        )
        assert reference_fields is not None
        ci_reference = float(reference_row["ci"])
        omega_reference = float(reference_row["omega_i"])

        for row, fields in solved:
            row["reference_Ly"] = float(reference_row["Ly"])
            row["reference_type"] = "largest-domain classical reference"
            row["ci_reference"] = ci_reference
            row["omega_i_reference"] = omega_reference
            row["abs_ci_error"] = (
                abs(float(row["ci"]) - ci_reference)
                if np.isfinite(row["ci"])
                else np.nan
            )
            row["abs_omega_error"] = (
                abs(float(row["omega_i"]) - omega_reference)
                if np.isfinite(row["omega_i"])
                else np.nan
            )
            for column in (
                "p_rel_l2",
                "rho_rel_l2",
                "u_rel_l2",
                "v_rel_l2",
                "modal_rel_l2_mean",
                "p_overlap",
                "p_overlap_defect",
                "alignment_factor_real",
                "alignment_factor_imag",
                "comparison_y_min",
                "comparison_y_max",
                "comparison_grid_size",
            ):
                row[column] = np.nan

            if fields is not None:
                try:
                    row.update(
                        compare_modes(fields, reference_fields, common_grid_size)
                    )
                except Exception as exc:
                    row["notes"] = (
                        f"{row['notes']} Modal comparison failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

            if np.isclose(float(row["Ly"]), float(reference_row["Ly"])):
                row["abs_ci_error"] = 0.0
                row["abs_omega_error"] = 0.0
                for field in FIELDS:
                    row[f"{field}_rel_l2"] = 0.0
                row["modal_rel_l2_mean"] = 0.0
                row["p_overlap"] = 1.0
                row["p_overlap_defect"] = 0.0
                row["alignment_factor_real"] = 1.0
                row["alignment_factor_imag"] = 0.0

            joint_converged = bool(
                row["solver_success"]
                and row["branch_check_passed"]
                and row["abs_ci_error"] <= ABS_CI_THRESHOLD
                and row["shooting_residual_abs"] <= RESIDUAL_THRESHOLD
                and row["p_rel_l2"] <= P_REL_L2_THRESHOLD
                and row["modal_rel_l2_mean"] <= MODAL_MEAN_THRESHOLD
            )
            row["converged"] = joint_converged
            row["convergence_criteria"] = (
                f"abs_ci<={ABS_CI_THRESHOLD:g};"
                f"|F|<={RESIDUAL_THRESHOLD:g};"
                f"p_rel_l2<={P_REL_L2_THRESHOLD:g};"
                f"modal_mean<={MODAL_MEAN_THRESHOLD:g}"
            )
            rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["point_id", "Ly"]).reset_index(drop=True)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise RuntimeError(f"Internal error: missing required columns {missing}")
    return frame


def build_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for point_id, group in table.groupby("point_id", sort=False):
        group = group.sort_values("Ly")
        production = group[np.isclose(group["Ly"], group["Ly_production"])]
        if len(production) != 1:
            raise ValueError(f"Expected one production row for {point_id}.")
        prod = production.iloc[0]
        largest_two = group.tail(2)
        converged = group[group["converged"].astype(bool)]
        rows.append(
            {
                "point_id": point_id,
                "point_category": prod["point_category"],
                "Mach": float(prod["Mach"]),
                "alpha": float(prod["alpha"]),
                "eta": float(prod["eta"]),
                "Ly_production": float(prod["Ly_production"]),
                "reference_Ly": float(prod["reference_Ly"]),
                "ci_reference": float(prod["ci_reference"]),
                "abs_ci_error_at_production": float(prod["abs_ci_error"]),
                "abs_omega_error_at_production": float(prod["abs_omega_error"]),
                "shooting_residual_at_production": float(
                    prod["shooting_residual_abs"]
                ),
                "p_rel_l2_at_production": float(prod["p_rel_l2"]),
                "modal_rel_l2_mean_at_production": float(
                    prod["modal_rel_l2_mean"]
                ),
                "p_overlap_defect_at_production": float(prod["p_overlap_defect"]),
                "minimum_converged_Ly": (
                    float(converged["Ly"].min()) if not converged.empty else np.nan
                ),
                "ci_variation_two_largest_domains": abs(
                    float(largest_two.iloc[-1]["ci"])
                    - float(largest_two.iloc[-2]["ci"])
                ),
                "modal_mean_variation_two_largest_domains": float(
                    largest_two.iloc[-2]["modal_rel_l2_mean"]
                ),
                "production_jointly_converged": bool(prod["converged"]),
                "final_status": (
                    "production box satisfies joint criterion"
                    if bool(prod["converged"])
                    else "production box does not satisfy joint criterion"
                ),
                "notes": prod["notes"],
            }
        )
    return pd.DataFrame(rows)


def positive_for_log(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float).copy()
    array[~np.isfinite(array) | (array <= 0.0)] = np.nan
    return array


def plot_figure(
    table: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    styles = {
        "interior": {"color": "#0072B2", "marker": "o"},
        "long_wave": {"color": "#D55E00", "marker": "s"},
        "near_neutral": {"color": "#009E73", "marker": "^"},
    }
    labels = {
        row.point_id: (
            rf"$M={row.Mach:g}$, $\alpha={row.alpha:g}$, "
            rf"$\eta={row.eta:.3f}$"
        )
        for row in table.drop_duplicates("point_id").itertuples()
    }
    panels = (
        ("abs_ci_error", r"$|c_i(L_y)-c_i^{\mathrm{ref}}|$"),
        ("p_rel_l2", r"$\varepsilon_p$"),
        ("modal_rel_l2_mean", r"$\overline{\varepsilon}_{\mathrm{mode}}$"),
        ("p_overlap_defect", r"$1-\mathcal{O}_p$"),
    )
    diagnostic_thresholds = {
        "abs_ci_error": (
            ABS_CI_THRESHOLD,
            rf"diagnostic threshold: ${ABS_CI_THRESHOLD:.0e}$",
        ),
        "p_rel_l2": (
            P_REL_L2_THRESHOLD,
            rf"diagnostic threshold: ${P_REL_L2_THRESHOLD:.0e}$",
        ),
        "modal_rel_l2_mean": (
            MODAL_MEAN_THRESHOLD,
            rf"diagnostic threshold: ${MODAL_MEAN_THRESHOLD:.0e}$",
        ),
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True)
    for panel_index, (axis, (column, ylabel)) in enumerate(
        zip(axes.flat, panels)
    ):
        for point_id, group in table.groupby("point_id", sort=False):
            group = group.sort_values("Ly")
            style = styles[point_id]
            normalized_Ly = (
                pd.to_numeric(group["Ly"], errors="coerce")
                / pd.to_numeric(group["Ly_production"], errors="coerce")
            )
            axis.plot(
                normalized_Ly,
                positive_for_log(group[column]),
                color=style["color"],
                marker=style["marker"],
                markersize=5,
                linewidth=1.6,
                label=labels[point_id],
            )
        axis.axvline(
            1.0,
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
        )
        if column in diagnostic_thresholds:
            threshold, threshold_label = diagnostic_thresholds[column]
            axis.axhline(
                threshold,
                color="0.45",
                linestyle=":",
                linewidth=1.0,
                alpha=0.85,
            )
            axis.annotate(
                threshold_label,
                xy=(0.985, threshold),
                xycoords=("axes fraction", "data"),
                xytext=(0, -3),
                textcoords="offset points",
                ha="right",
                va="top",
                color="0.35",
                fontsize=7.5,
            )
        axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", color="0.87", linewidth=0.6)
        axis.text(
            0.02,
            0.96,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    axes[1, 0].set_xlabel(r"$L_y/L_{y,\mathrm{prod}}$")
    axes[1, 1].set_xlabel(r"$L_y/L_{y,\mathrm{prod}}$")
    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=True,
        title=(
            "Reference: largest-domain solution for the same point.\n"
            "The zero-error reference point is omitted on logarithmic axes."
        ),
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.085,
        top=0.80,
        wspace=0.25,
        hspace=0.14,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{FIGURE_STEM}.png"
    pdf = output_dir / f"{FIGURE_STEM}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def validate_table(table: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in table]
    if missing:
        raise ValueError(f"The detailed table is missing columns: {missing}")
    if not (pd.to_numeric(table["Mach"], errors="coerce") < 1.0).all():
        raise ValueError("A non-subsonic point is present in the convergence table.")
    if table["point_id"].eq("near_neutral").sum() == 0:
        raise ValueError("The near-neutral point is absent.")
    if "relative_ci_error" in table:
        raise ValueError("A raw relative ci error must not be used in this table.")


def main() -> None:
    args = build_parser().parse_args()
    Ly_factors = parse_ly_factors(args.Ly_factors)
    data_output = args.data_output
    summary_output = args.summary_output

    if args.plot_only:
        if not data_output.exists():
            raise FileNotFoundError(f"Cannot plot: data table not found: {data_output}")
        table = pd.read_csv(data_output)
    elif data_output.exists() and not args.force_recompute:
        print(f"[reuse] existing convergence table: {data_output}")
        table = pd.read_csv(data_output)
    else:
        if data_output.exists() and args.force_recompute:
            print(f"[overwrite requested] {data_output}")
        table = compute_table(
            POINTS,
            Ly_factors,
            n_scan=args.n_scan,
            rtol=args.rtol,
            atol=args.atol,
            common_grid_size=args.common_grid_size,
        )
        validate_table(table)
        data_output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(data_output, index=False)
        summary = build_summary(table)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_output, index=False)
        print(f"\n[wrote] {data_output}")
        print(f"[wrote] {summary_output}")

    validate_table(table)
    if not summary_output.exists() or args.force_recompute:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        build_summary(table).to_csv(summary_output, index=False)

    if not args.compute_only:
        png, pdf = plot_figure(
            table,
            args.output_dir,
        )
        print(f"[wrote] {png}")
        print(f"[wrote] {pdf}")

    print("\nSelected points:")
    for point in POINTS:
        print(
            f"  {point.point_id:12s} M={point.Mach:.3f} "
            f"alpha={point.alpha:.3f} eta={point.eta:.6f}"
        )
    print("\nProduction-domain metrics:")
    summary = build_summary(table)
    print(
        summary[
            [
                "point_id",
                "abs_ci_error_at_production",
                "shooting_residual_at_production",
                "p_rel_l2_at_production",
                "modal_rel_l2_mean_at_production",
                "minimum_converged_Ly",
                "final_status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
