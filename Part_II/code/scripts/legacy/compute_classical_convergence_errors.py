#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = PACKAGE_ROOT / "reproducibility/results/classical_convergence"
DEFAULT_RUNS = DEFAULT_RESULTS / "runs/convergence_runs.csv"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "errors/convergence_errors.csv"
FIELDS = ("p", "rho", "u", "v")
MODE_COLUMNS = {"y"} | {
    f"{field}_{component}"
    for field in FIELDS
    for component in ("real", "imag")
}


def _ensure_script_path() -> None:
    value = str(SCRIPT_DIR)
    if value not in sys.path:
        sys.path.insert(0, value)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def validate_runs_frame(frame: pd.DataFrame) -> None:
    required = {
        "run_id",
        "case_id",
        "regime",
        "solver",
        "sweep_type",
        "Mach",
        "alpha",
        "cr",
        "ci",
        "omega_i",
        "runtime_seconds",
        "converged",
        "branch_check_passed",
        "mode_file",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"convergence_runs.csv is missing required columns: {missing}")
    if frame["run_id"].duplicated().any():
        duplicates = frame.loc[frame["run_id"].duplicated(), "run_id"].tolist()
        raise ValueError(f"Duplicate run_id values: {duplicates}")


def _reference_map(path: Path | None) -> dict[tuple[str, str, str], str]:
    if path is None:
        return {}
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = document.get("references") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        raise ValueError("Reference configuration must contain a references list.")
    result: dict[tuple[str, str, str], str] = {}
    for entry in entries:
        key = (str(entry["case_id"]), str(entry["solver"]), str(entry["sweep_type"]))
        result[key] = str(entry["run_id"])
    return result


def _finite_number(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _selection_mask(group: pd.DataFrame, selection: str) -> pd.Series:
    if selection == "overall":
        if "overall_validated" in group:
            return _as_bool(group["overall_validated"])
        return _as_bool(group["converged"]) & _as_bool(group["branch_check_passed"])
    if selection == "spectral":
        if "spectral_success" in group:
            return _as_bool(group["spectral_success"])
        return _as_bool(group["converged"])
    raise ValueError(f"Unsupported reference selection: {selection!r}")


def select_reference_run(
    group: pd.DataFrame,
    explicit_run_id: str | None = None,
    *,
    selection: str = "overall",
) -> pd.Series:
    valid = group[_selection_mask(group, selection)].copy()
    if valid.empty:
        raise ValueError(
            f"No {selection}-eligible run for "
            f"{group[['case_id', 'solver', 'sweep_type']].iloc[0].to_dict()}"
        )
    if explicit_run_id is not None:
        selected = valid[valid["run_id"].eq(explicit_run_id)]
        if len(selected) != 1:
            raise ValueError(f"Explicit reference run is absent or invalid: {explicit_run_id}")
        return selected.iloc[0]

    sweep_type = str(valid["sweep_type"].iloc[0])
    if sweep_type == "shooting_box":
        return valid.sort_values(["Ly", "runtime_seconds"], ascending=[False, True]).iloc[0]
    if sweep_type == "supersonic_gep_resolution":
        return valid.sort_values(["n_points", "runtime_seconds"], ascending=[False, True]).iloc[0]
    if sweep_type == "shooting_accuracy":
        valid["_rtol"] = valid["rtol"].map(lambda value: _finite_number(value, float("inf")))
        valid["_atol"] = valid["atol"].map(lambda value: _finite_number(value, float("inf")))
        valid["_max_step"] = valid["max_step"].map(lambda value: _finite_number(value, float("inf")))
        return valid.sort_values(
            ["_rtol", "_atol", "_max_step", "runtime_seconds"],
            ascending=[True, True, True, True],
        ).iloc[0]
    raise ValueError(f"Unsupported sweep type: {sweep_type}")


def load_mode(path_value: str | Path) -> dict[str, np.ndarray]:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Mode file not found: {path}")
    with np.load(path) as payload:
        missing = sorted(MODE_COLUMNS - set(payload.files))
        if missing:
            raise ValueError(f"Mode file {path} is missing arrays: {missing}")
        result = {name: np.asarray(payload[name]) for name in payload.files}
    y = np.asarray(result["y"], dtype=float)
    if len(y) < 2 or not np.all(np.diff(y) > 0.0):
        raise ValueError(f"Mode coordinates must be strictly increasing: {path}")
    return result


def _complex_field(mode: dict[str, np.ndarray], field: str) -> np.ndarray:
    return np.asarray(mode[f"{field}_real"], dtype=float) + 1j * np.asarray(
        mode[f"{field}_imag"], dtype=float
    )


def _interp_complex(y: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.interp(target, y, values.real) + 1j * np.interp(target, y, values.imag)


def optimal_complex_factor(predicted: np.ndarray, reference: np.ndarray) -> complex:
    denominator = np.vdot(predicted, predicted)
    if abs(denominator) <= 1e-30:
        raise ValueError("Cannot align a degenerate predicted mode.")
    return complex(np.vdot(predicted, reference) / denominator)


def _relative_l2(predicted: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    denominator = np.linalg.norm(reference[mask])
    if denominator <= 1e-30:
        return float("nan")
    return float(np.linalg.norm(predicted[mask] - reference[mask]) / denominator)


def compute_modal_errors(
    mode: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    core_threshold: float = 1e-3,
    common_grid_size: int = 2000,
) -> dict[str, Any]:
    if not (0.0 < core_threshold < 1.0):
        raise ValueError("core_threshold must be between zero and one.")
    if common_grid_size < 32:
        raise ValueError("common_grid_size must be at least 32.")

    y = np.asarray(mode["y"], dtype=float)
    y_ref = np.asarray(reference["y"], dtype=float)
    lower = max(float(y.min()), float(y_ref.min()))
    upper = min(float(y.max()), float(y_ref.max()))
    if not lower < upper:
        raise ValueError("The modal domains do not overlap.")
    common_y = np.linspace(lower, upper, common_grid_size)

    predicted: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}
    normalized_envelopes: list[np.ndarray] = []
    for field in FIELDS:
        predicted[field] = _interp_complex(y, _complex_field(mode, field), common_y)
        target[field] = _interp_complex(y_ref, _complex_field(reference, field), common_y)
        peak = max(float(np.max(np.abs(target[field]))), 1e-30)
        normalized_envelopes.append(np.abs(target[field]) / peak)

    envelope = np.maximum.reduce(normalized_envelopes)
    core = envelope >= core_threshold
    if int(core.sum()) < 8:
        raise ValueError("The configured modal core contains fewer than eight points.")
    full = np.ones(common_grid_size, dtype=bool)

    predicted_stack = np.concatenate([predicted[field][core] for field in FIELDS])
    reference_stack = np.concatenate([target[field][core] for field in FIELDS])
    factor = optimal_complex_factor(predicted_stack, reference_stack)
    aligned = {field: factor * predicted[field] for field in FIELDS}

    output: dict[str, Any] = {
        "core_threshold": float(core_threshold),
        "common_grid_size": int(common_grid_size),
        "common_y_min": lower,
        "common_y_max": upper,
        "alignment_real": float(factor.real),
        "alignment_imag": float(factor.imag),
        "phase_alignment_residual": _relative_l2(
            factor * predicted_stack,
            reference_stack,
            np.ones(len(reference_stack), dtype=bool),
        ),
        "n_core_points": int(core.sum()),
    }
    for field in FIELDS:
        output[f"mode_error_{field}_core"] = _relative_l2(aligned[field], target[field], core)
        output[f"mode_error_{field}_full"] = _relative_l2(aligned[field], target[field], full)
    output["mode_error_max_core"] = float(
        np.nanmax([output[f"mode_error_{field}_core"] for field in FIELDS])
    )
    output["mode_error_max_full"] = float(
        np.nanmax([output[f"mode_error_{field}_full"] for field in FIELDS])
    )
    return output


def compute_errors_frame(
    runs: pd.DataFrame,
    *,
    explicit_references: dict[tuple[str, str, str], str] | None = None,
    core_threshold: float | None = None,
    core_thresholds: list[float] | tuple[float, ...] | None = None,
    common_grid_size: int = 2000,
    ci_relative_floor: float = 1e-8,
    reference_selection: str = "overall",
) -> pd.DataFrame:
    validate_runs_frame(runs)
    explicit_references = explicit_references or {}
    if core_thresholds is None:
        core_thresholds = [1.0e-3 if core_threshold is None else float(core_threshold)]
    thresholds = sorted({float(value) for value in core_thresholds})
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("core_thresholds must contain values strictly between zero and one.")
    rows: list[dict[str, Any]] = []
    grouping = ["case_id", "solver", "sweep_type"]
    if "sweep_name" in runs.columns:
        grouping.append("sweep_name")
    for key, group in runs.groupby(grouping, dropna=False, sort=True):
        if group["Mach"].nunique(dropna=False) != 1 or group["alpha"].nunique(dropna=False) != 1:
            raise ValueError(
                "A convergence group mixes distinct (Mach, alpha) points: "
                f"{group[['case_id', 'Mach', 'alpha']].drop_duplicates().to_dict('records')}"
            )
        legacy_key = tuple(str(value) for value in key[:3])
        configured_reference = None
        if "reference_run_id" in group.columns:
            configured_ids = {
                str(value)
                for value in group["reference_run_id"].dropna()
                if str(value).strip()
            }
            if len(configured_ids) > 1:
                raise ValueError(f"A convergence group declares multiple references: {configured_ids}")
            if configured_ids:
                configured_reference = next(iter(configured_ids))
        try:
            reference = select_reference_run(
                group,
                explicit_references.get(legacy_key, configured_reference),
                selection=reference_selection,
            )
        except ValueError as exc:
            for _, run in group.iterrows():
                for threshold in thresholds:
                    row = run.to_dict()
                    row.update(
                        {
                            "reference_run_id": None,
                            "is_reference_run": False,
                            "reference_selection": reference_selection,
                            "reference_error_message": str(exc),
                            "abs_error_cr": np.nan,
                            "abs_error_ci": np.nan,
                            "abs_error_omega_i": np.nan,
                            "complex_error_c": np.nan,
                            "relative_error_ci": np.nan,
                            "core_threshold": threshold,
                            "common_grid_size": int(common_grid_size),
                            "phase_alignment_residual": np.nan,
                            "modal_error_message": "reference unavailable",
                            "mode_error_max_core": np.nan,
                            "mode_error_max_full": np.nan,
                        }
                    )
                    for field in FIELDS:
                        row[f"mode_error_{field}_core"] = np.nan
                        row[f"mode_error_{field}_full"] = np.nan
                    rows.append(row)
            continue
        c_ref = complex(float(reference["cr"]), float(reference["ci"]))
        omega_ref = float(reference["omega_i"])
        reference_mode = None
        if isinstance(reference.get("mode_file"), str) and reference["mode_file"].strip():
            reference_mode = load_mode(reference["mode_file"])

        for _, run in group.iterrows():
            mode_value = run.get("mode_file")
            mode = None
            mode_error = ""
            if reference_mode is not None and isinstance(mode_value, str) and mode_value.strip():
                try:
                    mode = load_mode(mode_value)
                except Exception as exc:
                    mode_error = f"{type(exc).__name__}: {exc}"
            elif not (isinstance(mode_value, str) and mode_value.strip()):
                mode_error = "mode unavailable"

            for threshold in thresholds:
                row = run.to_dict()
                row["reference_run_id"] = str(reference["run_id"])
                row["is_reference_run"] = str(run["run_id"]) == str(reference["run_id"])
                row["reference_selection"] = reference_selection
                row["reference_error_message"] = ""
                row["abs_error_cr"] = abs(float(run["cr"]) - c_ref.real) if pd.notna(run["cr"]) else np.nan
                row["abs_error_ci"] = abs(float(run["ci"]) - c_ref.imag) if pd.notna(run["ci"]) else np.nan
                row["abs_error_omega_i"] = (
                    abs(float(run["omega_i"]) - omega_ref) if pd.notna(run["omega_i"]) else np.nan
                )
                row["complex_error_c"] = (
                    abs(complex(float(run["cr"]), float(run["ci"])) - c_ref)
                    if pd.notna(run["cr"]) and pd.notna(run["ci"])
                    else np.nan
                )
                row["relative_error_ci"] = (
                    row["abs_error_ci"] / abs(c_ref.imag)
                    if abs(c_ref.imag) >= ci_relative_floor and pd.notna(row["abs_error_ci"])
                    else np.nan
                )
                row["core_threshold"] = threshold
                row["common_grid_size"] = int(common_grid_size)
                row["phase_alignment_residual"] = np.nan
                row["modal_error_message"] = mode_error
                for field in FIELDS:
                    row[f"mode_error_{field}_core"] = np.nan
                    row[f"mode_error_{field}_full"] = np.nan
                row["mode_error_max_core"] = np.nan
                row["mode_error_max_full"] = np.nan
                if reference_mode is not None and mode is not None:
                    try:
                        row.update(
                            compute_modal_errors(
                                mode,
                                reference_mode,
                                core_threshold=threshold,
                                common_grid_size=common_grid_size,
                            )
                        )
                    except Exception as exc:
                        row["modal_error_message"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    return pd.DataFrame(rows).sort_values(grouping + ["run_id", "core_threshold"]).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute spectral and complex-aligned modal errors for classical KH convergence runs."
    )
    parser.add_argument("--runs-csv", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-config", type=Path, default=None)
    parser.add_argument(
        "--core-thresholds",
        type=float,
        nargs="+",
        default=[1.0e-3, 1.0e-2, 5.0e-2],
        help="Long-format modal core thresholds; provisional defaults, not article choices.",
    )
    parser.add_argument("--reference-selection", choices=["overall", "spectral"], default="overall")
    parser.add_argument("--common-grid-size", type=int, default=2000)
    parser.add_argument("--ci-relative-floor", type=float, default=1e-8)
    parser.add_argument("--dry-run", action="store_true", help="Validate reference selection without writing output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.runs_csv.exists():
        raise FileNotFoundError(f"Runs CSV not found: {args.runs_csv}")
    runs = pd.read_csv(args.runs_csv)
    explicit = _reference_map(args.reference_config)
    errors = compute_errors_frame(
        runs,
        explicit_references=explicit,
        core_thresholds=args.core_thresholds,
        common_grid_size=args.common_grid_size,
        ci_relative_floor=args.ci_relative_floor,
        reference_selection=args.reference_selection,
    )
    if args.dry_run:
        references = errors[["case_id", "solver", "sweep_type", "reference_run_id"]].drop_duplicates()
        print(json.dumps({"n_rows": len(errors), "references": references.to_dict("records")}, indent=2))
        return 0
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
