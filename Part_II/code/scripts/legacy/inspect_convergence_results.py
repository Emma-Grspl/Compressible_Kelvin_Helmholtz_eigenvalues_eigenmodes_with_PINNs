#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = PACKAGE_ROOT / "reproducibility/results/classical_convergence"
DEFAULT_RUNS = DEFAULT_RESULTS / "runs/convergence_runs.csv"
DEFAULT_ERRORS = DEFAULT_RESULTS / "errors/convergence_errors.csv"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "inspection"


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _status(runs: pd.DataFrame, column: str, fallback: pd.Series) -> pd.Series:
    return _as_bool(runs[column]) if column in runs else fallback


def _axis_column(sweep_type: str) -> str:
    if sweep_type == "shooting_box":
        return "Ly"
    if sweep_type == "shooting_accuracy":
        return "accuracy_order"
    if sweep_type == "supersonic_gep_resolution":
        return "n_points"
    raise ValueError(f"Unsupported sweep type: {sweep_type}")


def _plateau_index(
    group: pd.DataFrame,
    *,
    spectral_tolerance: float,
    modal_tolerance: float,
) -> int | None:
    spectral_columns = [column for column in ("abs_error_cr", "abs_error_ci", "abs_error_omega_i") if column in group]
    if not spectral_columns:
        return None
    spectral = group[spectral_columns].max(axis=1, skipna=True).to_numpy(dtype=float)
    modal = (
        group["mode_error_max_full"].to_numpy(dtype=float)
        if "mode_error_max_full" in group
        else np.full(len(group), np.nan)
    )
    for index in range(len(group)):
        spectral_tail = spectral[index:]
        if not np.isfinite(spectral_tail).all() or np.any(spectral_tail > spectral_tolerance):
            continue
        modal_tail = modal[index:]
        finite_modal = modal_tail[np.isfinite(modal_tail)]
        if len(finite_modal) and np.any(finite_modal > modal_tolerance):
            continue
        return index
    return None


def build_summary(
    runs: pd.DataFrame,
    errors: pd.DataFrame | None,
    *,
    spectral_tolerance: float,
    modal_tolerance: float,
) -> pd.DataFrame:
    required = {"run_id", "case_id", "solver", "sweep_type", "converged", "branch_check_passed"}
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"Runs CSV is missing required columns: {missing}")
    merged = runs.copy()
    if errors is not None:
        errors = errors.sort_values(["run_id", "core_threshold"] if "core_threshold" in errors else ["run_id"])
        errors = errors.drop_duplicates("run_id", keep="last")
        error_columns = [
            column
            for column in (
                "run_id",
                "reference_run_id",
                "abs_error_cr",
                "abs_error_ci",
                "abs_error_omega_i",
                "mode_error_max_core",
                "mode_error_max_full",
            )
            if column in errors.columns
        ]
        merged = merged.merge(errors[error_columns], on="run_id", how="left", suffixes=("", "_error"))

    rows: list[dict] = []
    for key, group in merged.groupby(["case_id", "solver", "sweep_type"], sort=True):
        sweep_type = str(key[2])
        axis = _axis_column(sweep_type)
        group = group.sort_values(axis).reset_index(drop=True)
        legacy_converged = _as_bool(group["converged"])
        spectral = _status(group, "spectral_success", legacy_converged)
        modal = _status(group, "modal_reconstruction_success", legacy_converged)
        branch_ok = _as_bool(group["branch_check_passed"])
        overall = _status(group, "overall_validated", spectral & modal & branch_ok)
        tail_sensitive = _status(group, "tail_sensitivity_flag", pd.Series(False, index=group.index))
        bound_hit = _status(group, "legacy_stage2_bound_hit", pd.Series(False, index=group.index))
        unresolved = (
            group.get("branch_provenance_status", pd.Series("resolved", index=group.index))
            .astype(str)
            .eq("unresolved_current_solver_mismatch")
        )
        eligible = group[overall].copy().sort_values(axis).reset_index(drop=True)
        plateau = None
        nominal = None
        if not eligible.empty:
            plateau = _plateau_index(
                eligible,
                spectral_tolerance=spectral_tolerance,
                modal_tolerance=modal_tolerance,
            )
            nominal = eligible.iloc[plateau] if plateau is not None else eligible.iloc[-1]
        rows.append(
            {
                "case_id": key[0],
                "solver": key[1],
                "sweep_type": sweep_type,
                "n_runs": int(len(group)),
                "n_converged": int(legacy_converged.sum()),
                "n_failed": int((~legacy_converged).sum()),
                "n_spectral_success": int(spectral.sum()),
                "n_modal_reconstruction_success": int(modal.sum()),
                "n_branch_failures": int((~branch_ok).sum()),
                "n_tail_sensitive": int(tail_sensitive.sum()),
                "n_overall_validated": int(overall.sum()),
                "n_legacy_stage2_bound_hits": int(bound_hit.sum()),
                "n_unresolved_branch_provenance": int(unresolved.sum()),
                "plateau_reached": plateau is not None,
                "plateau_axis": axis,
                "plateau_value": None if plateau is None else float(eligible.iloc[plateau][axis]),
                "proposed_run_id": None if nominal is None else nominal["run_id"],
                "proposed_Ly": None if nominal is None else nominal.get("Ly"),
                "proposed_n_points": None if nominal is None else nominal.get("n_points"),
                "proposed_rtol": None if nominal is None else nominal.get("rtol"),
                "proposed_atol": None if nominal is None else nominal.get("atol"),
                "proposed_max_step": None if nominal is None else nominal.get("max_step"),
                "spectral_plateau_tolerance": spectral_tolerance,
                "modal_plateau_tolerance": modal_tolerance,
            }
        )
    return pd.DataFrame(rows)


def render_text(runs: pd.DataFrame, summary: pd.DataFrame) -> str:
    converged = _as_bool(runs["converged"])
    spectral = _status(runs, "spectral_success", converged)
    modal = _status(runs, "modal_reconstruction_success", converged)
    branch_ok = _as_bool(runs["branch_check_passed"])
    overall = _status(runs, "overall_validated", spectral & modal & branch_ok)
    tail = _status(runs, "tail_sensitivity_flag", pd.Series(False, index=runs.index))
    bound = _status(runs, "legacy_stage2_bound_hit", pd.Series(False, index=runs.index))
    unresolved = (
        runs.get("branch_provenance_status", pd.Series("resolved", index=runs.index))
        .astype(str)
        .eq("unresolved_current_solver_mismatch")
    )
    lines = [
        "Classical convergence inspection",
        "================================",
        f"Runs: {len(runs)}",
        f"Legacy converged: {int(converged.sum())}",
        f"Spectral successes: {int(spectral.sum())}",
        f"Modal reconstruction successes: {int(modal.sum())}",
        f"Branch check failures: {int((~branch_ok).sum())}",
        f"Tail-sensitive runs: {int(tail.sum())}",
        f"Overall validated: {int(overall.sum())}",
        f"Legacy stage-2 bound hits: {int(bound.sum())}",
        f"Unresolved M1.4 provenance runs: {int(unresolved.sum())}",
        "",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"{row['case_id']} / {row['solver']} / {row['sweep_type']}: "
            f"spectral={row['n_spectral_success']}/{row['n_runs']}, "
            f"modal={row['n_modal_reconstruction_success']}/{row['n_runs']}, "
            f"overall={row['n_overall_validated']}/{row['n_runs']}, "
            f"branch_failures={row['n_branch_failures']}, "
            f"tail_sensitive={row['n_tail_sensitive']}, "
            f"legacy_bound_hits={row['n_legacy_stage2_bound_hits']}, "
            f"unresolved_provenance={row['n_unresolved_branch_provenance']}, "
            f"plateau={row['plateau_reached']} at {row['plateau_axis']}={row['plateau_value']}, "
            f"proposed={row['proposed_run_id']}"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect classical convergence runs, branch failures, plateaus and proposed nominal settings."
    )
    parser.add_argument("--runs-csv", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--errors-csv", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spectral-plateau-tol", type=float, default=1e-5)
    parser.add_argument("--modal-plateau-tol", type=float, default=1e-2)
    parser.add_argument("--dry-run", action="store_true", help="Print the summary without writing files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.runs_csv.exists():
        raise FileNotFoundError(f"Runs CSV not found: {args.runs_csv}")
    runs = pd.read_csv(args.runs_csv)
    errors = pd.read_csv(args.errors_csv) if args.errors_csv.exists() else None
    summary = build_summary(
        runs,
        errors,
        spectral_tolerance=args.spectral_plateau_tol,
        modal_tolerance=args.modal_plateau_tol,
    )
    text = render_text(runs, summary)
    if args.dry_run:
        print(text, end="")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "convergence_summary.csv"
    text_path = args.output_dir / "convergence_summary.txt"
    summary.to_csv(csv_path, index=False)
    text_path.write_text(text, encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
