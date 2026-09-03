#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

import scripts.evaluation.test_kappa_q_modulus_reconstruction as base


SPECTRAL_CANDIDATES = (
    "assets/classic_supersonic/csv/pinn_direct/spectral/table_supersonic_primary_modal_spectral_133pts.csv",
    "assets/classic_supersonic/csv/pinn_direct/spectral/table_supersonic_reference_v2_spectral.csv",
    "assets/classic_supersonic/csv/pinn_direct/tables/table_supersonic_reference_v2_spectral_validated_50pts.csv",
    "assets/classic_supersonic/csv/pinn_direct/spectral/table_supersonic_reference_v2_spectral.csv",
    "assets/classic_supersonic/csv/pinn_direct/tables/table_supersonic_reference_v2_spectral_validated_50pts.csv",
)

TERMINAL_STATUSES = {
    "anchor_converged",
    "converged",
    "rejected",
    "stable_beyond_neutral",
    "not_reached_after_failure",
}


@dataclass(frozen=True)
class SolverSettings:
    name: str
    extent: float
    matching_y: float
    max_step: float
    rtol: float
    atol: float


DEFAULT_FALLBACKS = (
    SolverSettings("nominal", 20.0, 1.0, 0.50, 1.0e-9, 1.0e-11),
    SolverSettings("strict_L30", 30.0, 1.0, 0.25, 1.0e-10, 1.0e-12),
    SolverSettings("strict_L40", 40.0, 1.0, 0.20, 1.0e-11, 1.0e-13),
    SolverSettings("strict_L30_y0p5", 30.0, 0.5, 0.20, 1.0e-11, 1.0e-13),
    SolverSettings("strict_L30_y1p5", 30.0, 1.5, 0.20, 1.0e-11, 1.0e-13),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(json_safe(value), indent=2, sort_keys=True))


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ("mach_values", "alpha_min", "alpha_max", "alpha_step", "output_root")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config fields: {missing}")
    if not config["mach_values"]:
        raise ValueError("mach_values must not be empty.")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Production continuation campaign for the supersonic Riccati spectrum. "
            "Each invocation processes one Mach, checkpoints after every accepted "
            "point, resumes automatically, continues down and up from an anchor, "
            "switches from log(ci) to linear ci near neutrality, and stops at the "
            "estimated neutral boundary."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--Mach", type=float, default=None)
    parser.add_argument("--mach-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_mach(args: argparse.Namespace, config: dict[str, Any]) -> float:
    if args.Mach is not None:
        return float(args.Mach)
    index = args.mach_index
    if index is None:
        slurm_value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_value is None:
            raise ValueError("Provide --Mach, --mach-index, or SLURM_ARRAY_TASK_ID.")
        index = int(slurm_value)
    values = [float(item) for item in config["mach_values"]]
    if index < 0 or index >= len(values):
        raise IndexError(f"Mach index {index} outside [0, {len(values)-1}].")
    return values[index]


def decimal_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0 or stop < start:
        raise ValueError("Invalid grid bounds.")
    count = int(math.floor((stop - start) / step + 1.0e-10))
    grid = start + step * np.arange(count + 1, dtype=float)
    if grid[-1] < stop - 1.0e-10:
        grid = np.append(grid, stop)
    grid[-1] = min(grid[-1], stop)
    return np.unique(np.round(grid, 12))


def flow(y: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.tanh(y)
    return u, 1.0 - u * u


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


def endpoint_gamma(
    *,
    side: str,
    Mach: float,
    alpha: float,
    c: complex,
    settings: SolverSettings,
    method: str,
) -> dict[str, Any]:
    start = -settings.extent if side == "left" else settings.extent
    gamma0 = base.asymptotic_gamma(side=side, Mach=Mach, alpha=alpha, c=c)
    solution = solve_ivp(
        lambda y, state: gamma_rhs(y, state, Mach=Mach, alpha=alpha, c=c),
        (start, settings.matching_y),
        np.asarray([gamma0.real, gamma0.imag], dtype=float),
        method=method,
        t_eval=np.asarray([settings.matching_y], dtype=float),
        max_step=settings.max_step,
        rtol=settings.rtol,
        atol=settings.atol,
    )
    if not solution.success:
        raise RuntimeError(f"{side} integration failed: {solution.message}")
    endpoint = solution.y[:, -1]
    if not np.all(np.isfinite(endpoint)):
        raise RuntimeError(f"{side} endpoint is non-finite.")
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
        settings: SolverSettings,
        method: str,
        parameterization: str,
        direct_ci_scale: float,
    ) -> None:
        self.Mach = Mach
        self.alpha = alpha
        self.settings = settings
        self.method = method
        self.parameterization = parameterization
        self.direct_ci_scale = direct_ci_scale
        self.count = 0
        self.integration_failures = 0

    def unpack(self, vector: np.ndarray) -> tuple[float, float]:
        cr = float(vector[0])
        if self.parameterization == "log_ci":
            ci = float(math.exp(float(vector[1])))
        elif self.parameterization == "linear_ci":
            ci = float(self.direct_ci_scale * float(vector[1]))
        else:
            raise ValueError(self.parameterization)
        return cr, ci

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        self.count += 1
        cr, ci = self.unpack(vector)
        if not np.isfinite(cr) or not np.isfinite(ci) or ci <= 0.0:
            return np.asarray([1.0e3, 1.0e3], dtype=float)
        try:
            c = complex(cr, ci)
            left = endpoint_gamma(
                side="left",
                Mach=self.Mach,
                alpha=self.alpha,
                c=c,
                settings=self.settings,
                method=self.method,
            )
            right = endpoint_gamma(
                side="right",
                Mach=self.Mach,
                alpha=self.alpha,
                c=c,
                settings=self.settings,
                method=self.method,
            )
            return np.asarray(
                [left["kappa"] - right["kappa"], left["q"] - right["q"]],
                dtype=float,
            )
        except Exception:
            self.integration_failures += 1
            return np.asarray([1.0e3, 1.0e3], dtype=float)


def solver_parameterization(seed_ci: float, config: dict[str, Any]) -> str:
    threshold = float(config.get("direct_ci_switch", 1.0e-3))
    return "linear_ci" if seed_ci <= threshold else "log_ci"


def solve_once(
    *,
    Mach: float,
    alpha: float,
    seed_cr: float,
    seed_ci: float,
    settings: SolverSettings,
    config: dict[str, Any],
    parameterization: str | None = None,
) -> dict[str, Any]:
    if seed_ci <= 0.0:
        raise ValueError("seed_ci must be positive.")

    method = str(config.get("method", "DOP853"))
    root_tolerance = float(config.get("root_tolerance", 1.0e-8))
    ci_floor = float(config.get("ci_floor", 1.0e-12))
    ci_upper_global = float(config.get("ci_upper", 0.20))
    cr_half_width = float(config.get("cr_half_width", 0.08))
    ci_factor = float(config.get("ci_factor", 100.0))
    max_nfev = int(config.get("max_nfev", 100))
    parameterization = parameterization or solver_parameterization(seed_ci, config)

    cr_lower = max(-0.2, seed_cr - cr_half_width)
    cr_upper = min(1.2, seed_cr + cr_half_width)

    if parameterization == "log_ci":
        ci_lower = max(ci_floor, seed_ci / ci_factor)
        ci_upper = min(ci_upper_global, max(seed_ci * ci_factor, seed_ci + 1.0e-3))
        if ci_upper <= ci_lower:
            ci_upper = min(ci_upper_global, ci_lower * 10.0)
        x0 = np.asarray([seed_cr, math.log(np.clip(seed_ci, ci_lower, ci_upper))])
        bounds = (
            np.asarray([cr_lower, math.log(ci_lower)], dtype=float),
            np.asarray([cr_upper, math.log(ci_upper)], dtype=float),
        )
        x_scale = np.asarray([max(0.02, cr_half_width / 2.0), 1.0], dtype=float)
        diff_step = float(config.get("log_ci_diff_step", 1.0e-4))
        direct_scale = 1.0
    else:
        direct_scale = max(
            seed_ci,
            float(config.get("direct_ci_scale_floor", 1.0e-4)),
        )
        ci_lower = ci_floor
        ci_upper = min(
            ci_upper_global,
            max(seed_ci * ci_factor, float(config.get("direct_ci_switch", 1.0e-3)) * 5.0),
        )
        z0 = seed_ci / direct_scale
        bounds = (
            np.asarray([cr_lower, ci_lower / direct_scale], dtype=float),
            np.asarray([cr_upper, ci_upper / direct_scale], dtype=float),
        )
        x0 = np.asarray([seed_cr, np.clip(z0, bounds[0][1], bounds[1][1])])
        x_scale = np.asarray([max(0.02, cr_half_width / 2.0), 1.0], dtype=float)
        diff_step = float(config.get("linear_ci_diff_step", 1.0e-3))

    objective = RootObjective(
        Mach=Mach,
        alpha=alpha,
        settings=settings,
        method=method,
        parameterization=parameterization,
        direct_ci_scale=direct_scale,
    )

    result = least_squares(
        objective,
        x0=x0,
        bounds=bounds,
        method="trf",
        jac="3-point",
        diff_step=diff_step,
        x_scale=x_scale,
        xtol=float(config.get("optimizer_xtol", 1.0e-11)),
        ftol=float(config.get("optimizer_ftol", 1.0e-11)),
        gtol=float(config.get("optimizer_gtol", 1.0e-11)),
        max_nfev=max_nfev,
        verbose=0,
    )

    residual = objective(result.x)
    cr, ci = objective.unpack(result.x)
    norm = float(np.linalg.norm(residual))
    lower_distance = float(result.x[1] - bounds[0][1])
    upper_distance = float(bounds[1][1] - result.x[1])
    active_lower = lower_distance <= 1.0e-7 * max(1.0, abs(bounds[0][1]))
    active_upper = upper_distance <= 1.0e-7 * max(1.0, abs(bounds[1][1]))
    accepted = bool(
        np.isfinite(norm)
        and norm <= root_tolerance
        and ci > ci_floor
        and not active_lower
        and not active_upper
    )

    return {
        "Mach": Mach,
        "alpha": alpha,
        "cr": cr,
        "ci": ci,
        "omega_i": alpha * ci,
        "delta_kappa": float(residual[0]),
        "delta_q": float(residual[1]),
        "residual_norm": norm,
        "accepted": accepted,
        "parameterization": parameterization,
        "settings_name": settings.name,
        "spectral_extent": settings.extent,
        "matching_y": settings.matching_y,
        "max_step": settings.max_step,
        "rtol": settings.rtol,
        "atol": settings.atol,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "objective_calls": int(objective.count),
        "integration_failures": int(objective.integration_failures),
        "active_lower_ci": active_lower,
        "active_upper_ci": active_upper,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "seed_cr": seed_cr,
        "seed_ci": seed_ci,
    }


def fallback_settings(config: dict[str, Any]) -> list[SolverSettings]:
    custom = config.get("fallbacks")
    if not custom:
        return list(DEFAULT_FALLBACKS)
    return [
        SolverSettings(
            str(item["name"]),
            float(item["extent"]),
            float(item["matching_y"]),
            float(item["max_step"]),
            float(item["rtol"]),
            float(item["atol"]),
        )
        for item in custom
    ]


def solve_with_fallbacks(
    *,
    Mach: float,
    alpha: float,
    seed_cr: float,
    seed_ci: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    preferred = solver_parameterization(seed_ci, config)
    parameterizations = [preferred]
    if preferred == "log_ci" and seed_ci <= 5.0 * float(config.get("direct_ci_switch", 1.0e-3)):
        parameterizations.append("linear_ci")

    for settings in fallback_settings(config):
        for parameterization in parameterizations:
            result = solve_once(
                Mach=Mach,
                alpha=alpha,
                seed_cr=seed_cr,
                seed_ci=seed_ci,
                settings=settings,
                config=config,
                parameterization=parameterization,
            )
            result["attempt_timestamp"] = utc_now()
            attempts.append(result)
            if result["accepted"]:
                return result, attempts
    return None, attempts


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_existing_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def quality_score(frame: pd.DataFrame) -> pd.Series:
    score = pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)
    for column in ("trusted_spectral", "best_spectral_success"):
        if column in frame.columns:
            values = frame[column].astype(str).str.lower()
            score += values.isin(("true", "1", "yes")).astype(float) * 20.0
    for column in ("validation_status", "best_status", "campaign_target_status"):
        if column in frame.columns:
            values = frame[column].astype(str).str.lower()
            score += values.str.contains("valid|accept|converg", na=False).astype(float) * 10.0
            score -= values.str.contains("reject|fail|unresolved", na=False).astype(float) * 20.0
    mismatch_column = first_existing_column(
        frame,
        ("best_stage1_mismatch", "stage1_mismatch", "max_stage1", "canonical_stage1_mismatch"),
    )
    if mismatch_column is not None:
        mismatch = numeric(frame[mismatch_column])
        finite = np.isfinite(mismatch)
        score.loc[finite] += np.maximum(0.0, -np.log10(np.maximum(mismatch.loc[finite], 1.0e-30)))
    return score


def anchor_candidates(
    *,
    repo: Path,
    Mach: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    paths: list[Path] = []
    manual = config.get("anchor_table")
    if manual:
        manual_path = Path(str(manual))
        paths.append(manual_path if manual_path.is_absolute() else repo / manual_path)
    paths.extend(repo / relative for relative in SPECTRAL_CANDIDATES)

    records: list[dict[str, Any]] = []
    max_mach_distance = float(config.get("anchor_mach_max_distance", 0.11))
    for path in paths:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        mach_col = first_existing_column(frame, ("Mach", "M", "mach"))
        alpha_col = first_existing_column(frame, ("alpha", "Alpha"))
        cr_col = first_existing_column(frame, ("cr", "reference_cr", "best_cr"))
        ci_col = first_existing_column(frame, ("ci", "reference_ci", "best_ci"))
        if None in (mach_col, alpha_col, cr_col, ci_col):
            continue

        work = frame.copy()
        work["__Mach"] = numeric(work[mach_col])
        work["__alpha"] = numeric(work[alpha_col])
        work["__cr"] = numeric(work[cr_col])
        work["__ci"] = numeric(work[ci_col])
        work["__quality"] = quality_score(work)
        work = work[
            np.isfinite(work["__Mach"])
            & np.isfinite(work["__alpha"])
            & np.isfinite(work["__cr"])
            & np.isfinite(work["__ci"])
            & (work["__ci"] > 0.0)
        ].copy()
        if work.empty:
            continue
        work["__mach_distance"] = np.abs(work["__Mach"] - Mach)
        work = work[work["__mach_distance"] <= max_mach_distance].copy()
        if work.empty:
            continue
        # Reliable anchors should be comfortably unstable, not close to neutrality.
        work["__anchor_score"] = (
            work["__quality"]
            + 8.0 * np.log10(1.0 + 1000.0 * work["__ci"])
            - 100.0 * work["__mach_distance"]
        )
        for _, row in work.nlargest(int(config.get("anchor_candidates", 12)), "__anchor_score").iterrows():
            records.append(
                {
                    "source_path": str(path),
                    "source_Mach": float(row["__Mach"]),
                    "source_alpha": float(row["__alpha"]),
                    "source_cr": float(row["__cr"]),
                    "source_ci": float(row["__ci"]),
                    "mach_distance": float(row["__mach_distance"]),
                    "anchor_score": float(row["__anchor_score"]),
                }
            )
    records.sort(key=lambda item: item["anchor_score"], reverse=True)
    unique: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for record in records:
        key = (round(record["source_Mach"], 8), round(record["source_alpha"], 10), round(record["source_cr"], 10))
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def append_rows(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = load_frame(path)
    addition = pd.DataFrame(rows)
    combined = pd.concat([frame, addition], ignore_index=True, sort=False)
    atomic_write_csv(path, combined)
    return combined


def replace_point_row(path: Path, row: dict[str, Any]) -> pd.DataFrame:
    frame = load_frame(path)
    if not frame.empty:
        mask = (
            np.isclose(numeric(frame["alpha"]), float(row["alpha"]), rtol=0.0, atol=5.0e-13)
            & (frame["direction"].astype(str) == str(row["direction"]))
            & (frame["is_target"].astype(str).str.lower().isin(("true", "1")) == bool(row["is_target"]))
        )
        frame = frame.loc[~mask].copy()
    combined = pd.concat([frame, pd.DataFrame([row])], ignore_index=True, sort=False)
    combined = combined.sort_values(["alpha", "direction", "is_target"], kind="stable")
    atomic_write_csv(path, combined)
    return combined


def predictor(history: list[dict[str, Any]], trial_alpha: float) -> tuple[float, float]:
    last = history[-1]
    if len(history) < 2:
        return float(last["cr"]), float(last["ci"])
    previous = history[-2]
    denominator = float(last["alpha"] - previous["alpha"])
    if abs(denominator) < 1.0e-15:
        return float(last["cr"]), float(last["ci"])
    ratio = (trial_alpha - float(last["alpha"])) / denominator
    cr = float(last["cr"]) + ratio * (float(last["cr"]) - float(previous["cr"]))
    if min(float(last["ci"]), float(previous["ci"])) <= 5.0e-3:
        ci = float(last["ci"]) + ratio * (float(last["ci"]) - float(previous["ci"]))
    else:
        log_ci = math.log(float(last["ci"])) + ratio * (
            math.log(float(last["ci"])) - math.log(float(previous["ci"]))
        )
        ci = math.exp(log_ci)
    ci = max(ci, 1.0e-12)
    return cr, ci


def neutral_fit(history: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any] | None:
    threshold = float(config.get("neutral_fit_ci_max", 5.0e-4))
    points = [row for row in history if 0.0 < float(row["ci"]) <= threshold]
    count = int(config.get("neutral_fit_points", 5))
    if len(points) < max(3, count - 1):
        return None
    points = points[-count:]
    alpha = np.asarray([float(row["alpha"]) for row in points], dtype=float)
    ci = np.asarray([float(row["ci"]) for row in points], dtype=float)
    slope, intercept = np.polyfit(alpha, ci, 1)
    predicted = slope * alpha + intercept
    ss_res = float(np.sum((ci - predicted) ** 2))
    ss_tot = float(np.sum((ci - np.mean(ci)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    if slope >= 0.0:
        return None
    alpha_neutral = float(-intercept / slope)
    if not np.isfinite(alpha_neutral) or alpha_neutral <= alpha[-1]:
        return None
    return {
        "alpha_neutral": alpha_neutral,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "n_points": len(points),
        "alpha_min_fit": float(alpha[0]),
        "alpha_max_fit": float(alpha[-1]),
    }


def point_row(
    *,
    result: dict[str, Any],
    direction: str,
    is_target: bool,
    status: str,
    seed_alpha: float,
    requested_alpha: float,
    alpha_step: float,
    anchor: dict[str, Any],
    neutral: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        **result,
        "direction": direction,
        "is_target": bool(is_target),
        "status": status,
        "requested_alpha": requested_alpha,
        "seed_alpha": seed_alpha,
        "alpha_step": alpha_step,
        "anchor_source_path": anchor.get("source_path", ""),
        "anchor_source_Mach": anchor.get("source_Mach", np.nan),
        "anchor_source_alpha": anchor.get("source_alpha", np.nan),
        "neutral_alpha_estimate": neutral.get("alpha_neutral", np.nan) if neutral else np.nan,
        "neutral_fit_r2": neutral.get("r2", np.nan) if neutral else np.nan,
        "timestamp": utc_now(),
    }


def status_row(
    *,
    Mach: float,
    alpha: float,
    direction: str,
    status: str,
    requested_alpha: float,
    anchor: dict[str, Any],
    neutral: dict[str, Any] | None,
    message: str,
) -> dict[str, Any]:
    return {
        "Mach": Mach,
        "alpha": alpha,
        "cr": np.nan,
        "ci": np.nan,
        "omega_i": np.nan,
        "delta_kappa": np.nan,
        "delta_q": np.nan,
        "residual_norm": np.nan,
        "accepted": False,
        "parameterization": "",
        "settings_name": "",
        "spectral_extent": np.nan,
        "matching_y": np.nan,
        "max_step": np.nan,
        "rtol": np.nan,
        "atol": np.nan,
        "optimizer_success": False,
        "optimizer_status": np.nan,
        "optimizer_message": message,
        "optimizer_nfev": 0,
        "objective_calls": 0,
        "integration_failures": 0,
        "active_lower_ci": False,
        "active_upper_ci": False,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "seed_cr": np.nan,
        "seed_ci": np.nan,
        "direction": direction,
        "is_target": True,
        "status": status,
        "requested_alpha": requested_alpha,
        "seed_alpha": np.nan,
        "alpha_step": np.nan,
        "anchor_source_path": anchor.get("source_path", ""),
        "anchor_source_Mach": anchor.get("source_Mach", np.nan),
        "anchor_source_alpha": anchor.get("source_alpha", np.nan),
        "neutral_alpha_estimate": neutral.get("alpha_neutral", np.nan) if neutral else np.nan,
        "neutral_fit_r2": neutral.get("r2", np.nan) if neutral else np.nan,
        "timestamp": utc_now(),
    }


def find_or_create_anchor(
    *,
    repo: Path,
    Mach: float,
    config: dict[str, Any],
    points_path: Path,
    attempts_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing = load_frame(points_path)
    if not existing.empty:
        anchors = existing[existing["status"].astype(str) == "anchor_converged"]
        if not anchors.empty:
            row = anchors.sort_values("timestamp").iloc[-1].to_dict()
            source = {
                "source_path": row.get("anchor_source_path", "resume"),
                "source_Mach": row.get("anchor_source_Mach", Mach),
                "source_alpha": row.get("anchor_source_alpha", row["alpha"]),
                "source_cr": row["cr"],
                "source_ci": row["ci"],
            }
            return row, source

    candidates = anchor_candidates(repo=repo, Mach=Mach, config=config)
    if not candidates:
        raise RuntimeError(f"No anchor candidate found for Mach={Mach}.")

    all_attempts: list[dict[str, Any]] = []
    for candidate in candidates:
        result, attempts = solve_with_fallbacks(
            Mach=Mach,
            alpha=float(candidate["source_alpha"]),
            seed_cr=float(candidate["source_cr"]),
            seed_ci=float(candidate["source_ci"]),
            config=config,
        )
        for attempt in attempts:
            attempt.update({"purpose": "anchor", **candidate})
        all_attempts.extend(attempts)
        append_rows(attempts_path, attempts)
        if result is None:
            continue
        row = point_row(
            result=result,
            direction="anchor",
            is_target=True,
            status="anchor_converged",
            seed_alpha=float(candidate["source_alpha"]),
            requested_alpha=float(candidate["source_alpha"]),
            alpha_step=0.0,
            anchor=candidate,
            neutral=None,
        )
        replace_point_row(points_path, row)
        return row, candidate

    raise RuntimeError(
        f"All {len(candidates)} anchor candidates failed for Mach={Mach}."
    )


def accepted_history(points: pd.DataFrame, direction: str, anchor_row: dict[str, Any]) -> list[dict[str, Any]]:
    history = [anchor_row]
    if points.empty:
        return history
    subset = points[
        (points["direction"].astype(str) == direction)
        & points["status"].astype(str).isin(("converged",))
    ].copy()
    if subset.empty:
        return history
    ascending = direction == "high"
    subset = subset.sort_values("alpha", ascending=ascending)
    for _, row in subset.iterrows():
        history.append(row.to_dict())
    # Keep chronological direction order from the anchor.
    history = sorted(history, key=lambda item: float(item["alpha"]), reverse=not ascending)
    anchor_alpha = float(anchor_row["alpha"])
    if ascending:
        history = [item for item in history if float(item["alpha"]) >= anchor_alpha - 1.0e-12]
    else:
        history = [item for item in history if float(item["alpha"]) <= anchor_alpha + 1.0e-12]
        history.sort(key=lambda item: float(item["alpha"]), reverse=True)
    if not np.isclose(float(history[0]["alpha"]), anchor_alpha, atol=1.0e-12):
        history.insert(0, anchor_row)
    return history


def existing_target_status(points: pd.DataFrame, direction: str, alpha: float) -> str | None:
    if points.empty:
        return None
    mask = (
        (points["direction"].astype(str) == direction)
        & points["is_target"].astype(str).str.lower().isin(("true", "1"))
        & np.isclose(numeric(points["alpha"]), alpha, rtol=0.0, atol=5.0e-12)
    )
    subset = points.loc[mask]
    if subset.empty:
        return None
    return str(subset.iloc[-1]["status"])


def advance_to_target(
    *,
    Mach: float,
    target_alpha: float,
    direction: str,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    anchor: dict[str, Any],
    points_path: Path,
    attempts_path: Path,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any] | None]:
    sign = 1.0 if direction == "high" else -1.0
    minimum_step = float(config.get("minimum_bridge_step", 1.0e-5))
    growth = float(config.get("bridge_growth", 1.5))
    current_alpha = float(history[-1]["alpha"])
    remaining = target_alpha - current_alpha
    if sign * remaining <= 1.0e-13:
        return True, history, neutral_fit(history, config) if direction == "high" else None

    step = remaining
    max_loops = int(config.get("max_bridge_attempts_per_target", 100))
    loops = 0
    last_failure: dict[str, Any] | None = None

    while sign * (target_alpha - current_alpha) > 1.0e-13:
        loops += 1
        if loops > max_loops:
            break
        remaining = target_alpha - current_alpha
        if abs(step) > abs(remaining):
            step = remaining
        trial_alpha = current_alpha + step
        seed_cr, seed_ci = predictor(history, trial_alpha)
        result, attempts = solve_with_fallbacks(
            Mach=Mach,
            alpha=trial_alpha,
            seed_cr=seed_cr,
            seed_ci=seed_ci,
            config=config,
        )
        for attempt in attempts:
            attempt.update(
                {
                    "purpose": "continuation",
                    "direction": direction,
                    "requested_alpha": target_alpha,
                    "seed_alpha": current_alpha,
                    "trial_alpha": trial_alpha,
                    "trial_step": step,
                }
            )
        append_rows(attempts_path, attempts)

        if result is None:
            last_failure = attempts[-1] if attempts else None
            step *= 0.5
            if abs(step) < minimum_step:
                break
            continue

        is_target = abs(trial_alpha - target_alpha) <= 5.0e-12
        neutral = neutral_fit(history + [result], config) if direction == "high" else None
        row = point_row(
            result=result,
            direction=direction,
            is_target=is_target,
            status="converged",
            seed_alpha=current_alpha,
            requested_alpha=target_alpha,
            alpha_step=step,
            anchor=anchor,
            neutral=neutral,
        )
        replace_point_row(points_path, row)
        history.append(row)
        current_alpha = trial_alpha

        if is_target:
            return True, history, neutral

        remaining = target_alpha - current_alpha
        step = sign * min(abs(remaining), abs(step) * growth)

    neutral = neutral_fit(history, config) if direction == "high" else None
    message = "Continuation could not reach target above minimum bridge step."
    if last_failure is not None:
        message += f" Last residual={last_failure.get('residual_norm', np.nan)}"
    row = status_row(
        Mach=Mach,
        alpha=target_alpha,
        direction=direction,
        status="rejected",
        requested_alpha=target_alpha,
        anchor=anchor,
        neutral=neutral,
        message=message,
    )
    replace_point_row(points_path, row)
    return False, history, neutral


def mark_remaining(
    *,
    Mach: float,
    alphas: Iterable[float],
    direction: str,
    status: str,
    anchor: dict[str, Any],
    neutral: dict[str, Any] | None,
    message: str,
    points_path: Path,
) -> None:
    points = load_frame(points_path)
    for alpha in alphas:
        if existing_target_status(points, direction, float(alpha)) in TERMINAL_STATUSES:
            continue
        row = status_row(
            Mach=Mach,
            alpha=float(alpha),
            direction=direction,
            status=status,
            requested_alpha=float(alpha),
            anchor=anchor,
            neutral=neutral,
            message=message,
        )
        points = replace_point_row(points_path, row)


def run_direction(
    *,
    Mach: float,
    direction: str,
    targets: list[float],
    anchor_row: dict[str, Any],
    anchor: dict[str, Any],
    config: dict[str, Any],
    points_path: Path,
    attempts_path: Path,
    state_path: Path,
) -> None:
    points = load_frame(points_path)
    history = accepted_history(points, direction, anchor_row)
    minimum_r2 = float(config.get("neutral_min_r2", 0.98))
    neutral_margin = float(config.get("neutral_stop_margin", 0.0))

    for index, target_alpha in enumerate(targets):
        points = load_frame(points_path)
        status = existing_target_status(points, direction, target_alpha)
        if status in TERMINAL_STATUSES:
            continue

        neutral = neutral_fit(history, config) if direction == "high" else None
        if (
            direction == "high"
            and neutral is not None
            and neutral["r2"] >= minimum_r2
            and target_alpha >= neutral["alpha_neutral"] - neutral_margin
        ):
            mark_remaining(
                Mach=Mach,
                alphas=targets[index:],
                direction=direction,
                status="stable_beyond_neutral",
                anchor=anchor,
                neutral=neutral,
                message="Target lies at or beyond extrapolated neutral boundary.",
                points_path=points_path,
            )
            atomic_write_json(
                state_path,
                {
                    "Mach": Mach,
                    "direction": direction,
                    "complete": True,
                    "reason": "neutral_boundary",
                    "neutral_fit": neutral,
                    "updated": utc_now(),
                },
            )
            return

        print(
            f"\n[{direction}] target alpha={target_alpha:.12g}; "
            f"last accepted={float(history[-1]['alpha']):.12g}, "
            f"ci={float(history[-1]['ci']):.6e}",
            flush=True,
        )
        success, history, neutral = advance_to_target(
            Mach=Mach,
            target_alpha=target_alpha,
            direction=direction,
            history=history,
            config=config,
            anchor=anchor,
            points_path=points_path,
            attempts_path=attempts_path,
        )
        atomic_write_json(
            state_path,
            {
                "Mach": Mach,
                "direction": direction,
                "last_target": target_alpha,
                "last_success": success,
                "last_accepted_alpha": float(history[-1]["alpha"]),
                "last_accepted_cr": float(history[-1]["cr"]),
                "last_accepted_ci": float(history[-1]["ci"]),
                "neutral_fit": neutral,
                "updated": utc_now(),
            },
        )

        if not success:
            if direction == "high" and neutral is not None:
                mark_remaining(
                    Mach=Mach,
                    alphas=targets[index + 1 :],
                    direction=direction,
                    status="stable_beyond_neutral",
                    anchor=anchor,
                    neutral=neutral,
                    message="Continuation stopped close to extrapolated neutral boundary.",
                    points_path=points_path,
                )
                return
            mark_remaining(
                Mach=Mach,
                alphas=targets[index + 1 :],
                direction=direction,
                status="not_reached_after_failure",
                anchor=anchor,
                neutral=neutral,
                message="Direction stopped after an unrecoverable continuation failure.",
                points_path=points_path,
            )
            return

    atomic_write_json(
        state_path,
        {
            "Mach": Mach,
            "direction": direction,
            "complete": True,
            "reason": "grid_complete",
            "neutral_fit": neutral_fit(history, config) if direction == "high" else None,
            "updated": utc_now(),
        },
    )


def campaign_summary(points: pd.DataFrame, Mach: float) -> dict[str, Any]:
    converged = points[points["status"].astype(str).isin(("anchor_converged", "converged"))]
    targets = points[points["is_target"].astype(str).str.lower().isin(("true", "1"))] if not points.empty else points
    neutral_values = numeric(points.get("neutral_alpha_estimate", pd.Series(dtype=float)))
    neutral_values = neutral_values[np.isfinite(neutral_values)]
    return {
        "Mach": Mach,
        "n_rows": int(len(points)),
        "n_target_rows": int(len(targets)),
        "n_converged": int(len(converged)),
        "n_rejected": int(np.sum(points["status"].astype(str) == "rejected")),
        "n_stable_beyond_neutral": int(np.sum(points["status"].astype(str) == "stable_beyond_neutral")),
        "alpha_min_converged": float(numeric(converged["alpha"]).min()) if not converged.empty else None,
        "alpha_max_converged": float(numeric(converged["alpha"]).max()) if not converged.empty else None,
        "neutral_alpha_estimate": float(neutral_values.iloc[-1]) if len(neutral_values) else None,
        "updated": utc_now(),
    }


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = repo / config_path
    config = load_config(config_path)
    Mach = resolve_mach(args, config)

    output_root = Path(str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = repo / output_root
    mach_dir = output_root / f"M{Mach:.6f}".replace(".", "p")
    points_path = mach_dir / "spectral_points.csv"
    attempts_path = mach_dir / "solver_attempts.csv"
    state_low_path = mach_dir / "state_low.json"
    state_high_path = mach_dir / "state_high.json"
    summary_path = mach_dir / "campaign_summary.json"
    mach_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for path in (points_path, attempts_path, state_low_path, state_high_path, summary_path):
            path.unlink(missing_ok=True)

    alpha_grid = decimal_grid(
        float(config["alpha_min"]),
        float(config["alpha_max"]),
        float(config["alpha_step"]),
    )

    print("=== Dense supersonic spectral campaign ===")
    print(f"Mach         : {Mach}")
    print(f"alpha grid   : {alpha_grid[0]} .. {alpha_grid[-1]} ({len(alpha_grid)} targets)")
    print(f"output       : {mach_dir}")
    print(f"resume       : {not args.overwrite}")

    if args.dry_run:
        candidates = anchor_candidates(repo=repo, Mach=Mach, config=config)
        print(pd.DataFrame(candidates[:10]).to_string(index=False))
        return 0

    anchor_row, anchor = find_or_create_anchor(
        repo=repo,
        Mach=Mach,
        config=config,
        points_path=points_path,
        attempts_path=attempts_path,
    )
    anchor_alpha = float(anchor_row["alpha"])
    print(
        f"Anchor       : alpha={anchor_alpha:.12g}, "
        f"c={float(anchor_row['cr']):.12g}+{float(anchor_row['ci']):.6e}i, "
        f"source={anchor.get('source_path')}",
        flush=True,
    )

    low_targets = [float(value) for value in alpha_grid[alpha_grid < anchor_alpha - 5.0e-12][::-1]]
    high_targets = [float(value) for value in alpha_grid[alpha_grid > anchor_alpha + 5.0e-12]]

    run_direction(
        Mach=Mach,
        direction="low",
        targets=low_targets,
        anchor_row=anchor_row,
        anchor=anchor,
        config=config,
        points_path=points_path,
        attempts_path=attempts_path,
        state_path=state_low_path,
    )
    run_direction(
        Mach=Mach,
        direction="high",
        targets=high_targets,
        anchor_row=anchor_row,
        anchor=anchor,
        config=config,
        points_path=points_path,
        attempts_path=attempts_path,
        state_path=state_high_path,
    )

    points = load_frame(points_path)
    summary = campaign_summary(points, Mach)
    atomic_write_json(summary_path, summary)
    print("\n=== CAMPAIGN SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Written to: {mach_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
