#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.evaluation.run_dense_supersonic_campaign as campaign


ACCEPTED_STATUSES = {"accepted_root", "accepted", "converged", "success"}


def robust_config(path: Path) -> dict[str, Any]:
    config = campaign.load_config(path)
    config = dict(config)
    config["root_tolerance"] = min(float(config.get("root_tolerance", 1e-8)), 1e-8)
    config["ci_floor"] = min(float(config.get("ci_floor", 1e-12)), 1e-12)
    config["ci_upper"] = max(float(config.get("ci_upper", 0.2)), 0.25)
    config["cr_half_width"] = max(float(config.get("cr_half_width", 0.08)), 0.12)
    config["ci_factor"] = max(float(config.get("ci_factor", 100.0)), 1000.0)
    config["max_nfev"] = max(int(config.get("max_nfev", 120)), 180)
    return config


def numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def accepted_mask(frame: pd.DataFrame) -> pd.Series:
    status_ok = frame["status"].astype(str).str.strip().isin(ACCEPTED_STATUSES)
    return (
        status_ok
        & np.isfinite(pd.to_numeric(frame["classical_cr"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["classical_ci"], errors="coerce"))
    )


def load_targets(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "blumen_row_id",
        "curve_key",
        "Mach",
        "alpha",
        "blumen_ci",
        "status",
        "classical_cr",
        "classical_ci",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing pointwise columns: {missing}")

    frame = numeric(
        frame,
        (
            "task_index",
            "blumen_row_id",
            "source_row_id",
            "Mach",
            "alpha",
            "blumen_ci",
            "classical_cr",
            "classical_ci",
            "residual_norm",
        ),
    )
    frame = frame.loc[
        np.isfinite(frame["Mach"])
        & np.isfinite(frame["alpha"])
        & np.isfinite(frame["blumen_ci"])
        & (frame["blumen_ci"] > 0.0)
    ].copy()
    order_column = "source_row_id" if "source_row_id" in frame.columns else "blumen_row_id"
    frame = frame.sort_values(order_column, kind="stable").reset_index(drop=True)
    frame["true_isoline_task_index"] = np.arange(len(frame), dtype=int)
    return frame


def load_seed_pool(reference_path: Path, pointwise: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(reference_path)
    required = {"Mach", "alpha", "cr", "ci"}
    missing = sorted(required.difference(reference.columns))
    if missing:
        raise KeyError(f"Missing reference columns: {missing}")
    reference = numeric(reference, ("Mach", "alpha", "cr", "ci", "residual_norm"))
    reference = reference.dropna(subset=["Mach", "alpha", "cr", "ci"]).copy()
    if "accepted" in reference.columns:
        mask = reference["accepted"].astype(str).str.lower().isin({"true", "1", "yes"})
        reference = reference.loc[mask].copy()
    reference = reference.rename(columns={"cr": "seed_cr", "ci": "seed_ci"})
    reference["seed_source"] = "canonical_reference"

    pointwise_ok = pointwise.loc[accepted_mask(pointwise)].copy()
    pointwise_ok = pointwise_ok.rename(
        columns={"classical_cr": "seed_cr", "classical_ci": "seed_ci"}
    )
    pointwise_ok["seed_source"] = "pointwise_blumen"

    columns = ["Mach", "alpha", "seed_cr", "seed_ci", "seed_source"]
    pool = pd.concat(
        [reference[columns], pointwise_ok[columns]],
        ignore_index=True,
        sort=False,
    )
    pool = pool.dropna(subset=["Mach", "alpha", "seed_cr", "seed_ci"])
    pool = pool.loc[pool["seed_ci"] > 0.0].reset_index(drop=True)
    return pool


def seed_candidates(
    *,
    pool: pd.DataFrame,
    Mach: float,
    alpha: float,
    target_ci: float,
    preferred: dict[str, Any] | None,
    target_row: pd.Series,
    max_candidates: int = 5,
) -> list[tuple[float, float, str]]:
    candidates: list[tuple[float, float, str]] = []

    if preferred is not None:
        candidates.append(
            (float(preferred["cr"]), float(preferred["ci"]), "continuation")
        )

    target_status = str(target_row.get("status", "")).strip()
    target_cr = pd.to_numeric(pd.Series([target_row.get("classical_cr")]), errors="coerce").iloc[0]
    target_classical_ci = pd.to_numeric(
        pd.Series([target_row.get("classical_ci")]), errors="coerce"
    ).iloc[0]
    if (
        target_status in ACCEPTED_STATUSES
        and np.isfinite(target_cr)
        and np.isfinite(target_classical_ci)
        and target_classical_ci > 0.0
    ):
        candidates.append(
            (float(target_cr), float(target_classical_ci), "target_pointwise_root")
        )

    if not pool.empty:
        work = pool.copy()
        work["score"] = (
            ((work["Mach"] - Mach) / 0.04) ** 2
            + ((work["alpha"] - alpha) / 0.025) ** 2
            + 0.15 * (np.log(work["seed_ci"] / target_ci)) ** 2
        )
        for _, row in work.nsmallest(max_candidates, "score").iterrows():
            candidates.append(
                (float(row["seed_cr"]), float(row["seed_ci"]), str(row["seed_source"]))
            )

    # Last-resort target-ci seeds using the best available cr values.
    for cr, _, source in list(candidates)[:3]:
        candidates.append((float(cr), float(target_ci), f"{source}_target_ci"))

    unique: list[tuple[float, float, str]] = []
    seen: set[tuple[float, float]] = set()
    for cr, ci, source in candidates:
        if not (np.isfinite(cr) and np.isfinite(ci) and ci > 0.0):
            continue
        key = (round(cr, 10), round(ci, 12))
        if key in seen:
            continue
        seen.add(key)
        unique.append((cr, ci, source))
    return unique[: max_candidates + 3]


class AlphaEvaluator:
    def __init__(
        self,
        *,
        Mach: float,
        target_ci: float,
        pool: pd.DataFrame,
        target_row: pd.Series,
        config: dict[str, Any],
    ) -> None:
        self.Mach = Mach
        self.target_ci = target_ci
        self.pool = pool
        self.target_row = target_row
        self.config = config
        self.cache: dict[float, dict[str, Any]] = {}
        self.attempt_rows: list[dict[str, Any]] = []

    def evaluate(
        self,
        alpha: float,
        preferred: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        alpha = float(alpha)
        key = round(alpha, 12)
        if key in self.cache:
            return self.cache[key]

        candidates = seed_candidates(
            pool=self.pool,
            Mach=self.Mach,
            alpha=alpha,
            target_ci=self.target_ci,
            preferred=preferred,
            target_row=self.target_row,
        )

        for seed_index, (seed_cr, seed_ci, seed_source) in enumerate(candidates):
            try:
                root, attempts = campaign.solve_with_fallbacks(
                    Mach=self.Mach,
                    alpha=alpha,
                    seed_cr=seed_cr,
                    seed_ci=max(seed_ci, 1e-12),
                    config=self.config,
                )
            except Exception as exc:
                self.attempt_rows.append(
                    {
                        "Mach": self.Mach,
                        "alpha": alpha,
                        "target_ci": self.target_ci,
                        "seed_index": seed_index,
                        "seed_source": seed_source,
                        "seed_cr": seed_cr,
                        "seed_ci": seed_ci,
                        "accepted": False,
                        "exception": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            for attempt in attempts:
                record = dict(attempt)
                record.update(
                    {
                        "target_ci": self.target_ci,
                        "seed_index": seed_index,
                        "seed_source": seed_source,
                    }
                )
                self.attempt_rows.append(record)

            if root is not None:
                value = dict(root)
                value["f_ci"] = float(value["ci"] - self.target_ci)
                value["seed_source_selected"] = seed_source
                self.cache[key] = value
                return value

        return None

    def successful(self) -> list[dict[str, Any]]:
        return sorted(self.cache.values(), key=lambda row: float(row["alpha"]))


def nearest_exact(
    rows: list[dict[str, Any]],
    alpha_blumen: float,
    ci_tolerance: float,
) -> dict[str, Any] | None:
    candidates = [row for row in rows if abs(float(row["f_ci"])) <= ci_tolerance]
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(float(row["alpha"]) - alpha_blumen))


def nearest_bracket(
    rows: list[dict[str, Any]],
    alpha_blumen: float,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for left, right in zip(rows[:-1], rows[1:], strict=False):
        f_left = float(left["f_ci"])
        f_right = float(right["f_ci"])
        if not (np.isfinite(f_left) and np.isfinite(f_right)):
            continue
        if f_left == 0.0 or f_right == 0.0 or f_left * f_right < 0.0:
            midpoint = 0.5 * (float(left["alpha"]) + float(right["alpha"]))
            candidates.append((abs(midpoint - alpha_blumen), left, right))
    if not candidates:
        return None
    _, left, right = min(candidates, key=lambda item: item[0])
    return left, right


def refine_bracket(
    *,
    evaluator: AlphaEvaluator,
    left: dict[str, Any],
    right: dict[str, Any],
    alpha_tolerance: float,
    ci_tolerance: float,
    max_iterations: int,
) -> tuple[dict[str, Any], float, float]:
    if float(left["alpha"]) > float(right["alpha"]):
        left, right = right, left

    best = min((left, right), key=lambda row: abs(float(row["f_ci"])))

    for _ in range(max_iterations):
        a = float(left["alpha"])
        b = float(right["alpha"])
        fa = float(left["f_ci"])
        fb = float(right["f_ci"])

        if abs(float(best["f_ci"])) <= ci_tolerance or (b - a) <= alpha_tolerance:
            break

        if np.isfinite(fa) and np.isfinite(fb) and abs(fb - fa) > 1e-15:
            trial_alpha = b - fb * (b - a) / (fb - fa)
        else:
            trial_alpha = 0.5 * (a + b)

        inner_lo = a + 0.15 * (b - a)
        inner_hi = b - 0.15 * (b - a)
        if not (inner_lo <= trial_alpha <= inner_hi):
            trial_alpha = 0.5 * (a + b)

        preferred = left if abs(trial_alpha - a) <= abs(trial_alpha - b) else right
        trial = evaluator.evaluate(trial_alpha, preferred=preferred)
        if trial is None:
            midpoint = 0.5 * (a + b)
            if abs(midpoint - trial_alpha) > 1e-12:
                trial = evaluator.evaluate(midpoint, preferred=preferred)
        if trial is None:
            break

        if abs(float(trial["f_ci"])) < abs(float(best["f_ci"])):
            best = trial

        ft = float(trial["f_ci"])
        if fa == 0.0:
            best = left
            break
        if fb == 0.0:
            best = right
            break
        if fa * ft <= 0.0:
            right = trial
        else:
            left = trial

    lower = float(left["alpha"])
    upper = float(right["alpha"])
    best = min(
        evaluator.successful(),
        key=lambda row: (
            abs(float(row["f_ci"])),
            abs(float(row["alpha"]) - 0.5 * (lower + upper)),
        ),
    )
    return best, lower, upper


def result_from_root(
    *,
    target: pd.Series,
    root: dict[str, Any],
    status: str,
    alpha_lower: float | None = None,
    alpha_upper: float | None = None,
    n_spectral_solves: int,
) -> dict[str, Any]:
    alpha_blumen = float(target["alpha"])
    alpha_classical = float(root["alpha"])
    out = target.to_dict()
    out.update(
        {
            "status": status,
            "alpha_blumen": alpha_blumen,
            "target_ci": float(target["blumen_ci"]),
            "alpha_classical": alpha_classical,
            "delta_alpha": alpha_classical - alpha_blumen,
            "classical_cr": float(root["cr"]),
            "classical_ci": float(root["ci"]),
            "delta_ci_at_root": float(root["ci"] - float(target["blumen_ci"])),
            "residual_norm": float(root["residual_norm"]),
            "settings_name": root.get("settings_name"),
            "parameterization": root.get("parameterization"),
            "seed_source_selected": root.get("seed_source_selected"),
            "alpha_bracket_lower": alpha_lower,
            "alpha_bracket_upper": alpha_upper,
            "alpha_bracket_width": (
                None if alpha_lower is None or alpha_upper is None else alpha_upper - alpha_lower
            ),
            "n_spectral_solves": n_spectral_solves,
        }
    )
    return out


def failure_result(
    *,
    target: pd.Series,
    status: str,
    message: str,
    n_spectral_solves: int,
) -> dict[str, Any]:
    out = target.to_dict()
    out.update(
        {
            "status": status,
            "alpha_blumen": float(target["alpha"]),
            "target_ci": float(target["blumen_ci"]),
            "alpha_classical": np.nan,
            "delta_alpha": np.nan,
            "classical_cr": np.nan,
            "classical_ci": np.nan,
            "delta_ci_at_root": np.nan,
            "residual_norm": np.nan,
            "alpha_bracket_lower": np.nan,
            "alpha_bracket_upper": np.nan,
            "alpha_bracket_width": np.nan,
            "n_spectral_solves": n_spectral_solves,
            "message": message,
        }
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alpha-min", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=0.50)
    parser.add_argument("--scan-step", type=float, default=0.0025)
    parser.add_argument("--max-halfwidth", type=float, default=0.18)
    parser.add_argument("--alpha-tolerance", type=float, default=2e-5)
    parser.add_argument("--ci-tolerance", type=float, default=2e-5)
    parser.add_argument("--max-refine-iterations", type=int, default=24)
    args = parser.parse_args()

    targets = load_targets(args.pointwise_csv.resolve())
    if args.task_index < 0 or args.task_index >= len(targets):
        raise IndexError(f"Task {args.task_index} outside [0, {len(targets)-1}]")
    target = targets.iloc[args.task_index]

    Mach = float(target["Mach"])
    alpha_blumen = float(target["alpha"])
    target_ci = float(target["blumen_ci"])
    config = robust_config(args.config.resolve())
    pool = load_seed_pool(args.reference_csv.resolve(), targets)

    evaluator = AlphaEvaluator(
        Mach=Mach,
        target_ci=target_ci,
        pool=pool,
        target_row=target,
        config=config,
    )

    point_root = args.output_root.resolve() / f"point_{args.task_index:03d}"
    point_root.mkdir(parents=True, exist_ok=True)

    try:
        center = evaluator.evaluate(alpha_blumen)
        rows = evaluator.successful()
        exact = nearest_exact(rows, alpha_blumen, args.ci_tolerance)

        if exact is not None:
            result = result_from_root(
                target=target,
                root=exact,
                status="converged_center",
                n_spectral_solves=len(rows),
            )
        else:
            max_steps = int(math.ceil(args.max_halfwidth / args.scan_step))
            bracket = nearest_bracket(rows, alpha_blumen)

            for step_index in range(1, max_steps + 1):
                if bracket is not None:
                    break
                radius = step_index * args.scan_step
                for trial_alpha in (alpha_blumen - radius, alpha_blumen + radius):
                    if trial_alpha < args.alpha_min or trial_alpha > args.alpha_max:
                        continue
                    preferred = center
                    evaluator.evaluate(trial_alpha, preferred=preferred)
                rows = evaluator.successful()
                exact = nearest_exact(rows, alpha_blumen, args.ci_tolerance)
                if exact is not None:
                    break
                bracket = nearest_bracket(rows, alpha_blumen)

            if exact is not None:
                result = result_from_root(
                    target=target,
                    root=exact,
                    status="converged_scan_point",
                    n_spectral_solves=len(rows),
                )
            elif bracket is not None:
                root, lower, upper = refine_bracket(
                    evaluator=evaluator,
                    left=bracket[0],
                    right=bracket[1],
                    alpha_tolerance=args.alpha_tolerance,
                    ci_tolerance=args.ci_tolerance,
                    max_iterations=args.max_refine_iterations,
                )
                result = result_from_root(
                    target=target,
                    root=root,
                    status="converged_root",
                    alpha_lower=lower,
                    alpha_upper=upper,
                    n_spectral_solves=len(evaluator.successful()),
                )
            elif rows:
                nearest = min(rows, key=lambda row: abs(float(row["f_ci"])))
                result = failure_result(
                    target=target,
                    status="no_alpha_bracket",
                    message=(
                        "No sign bracket found. Best sampled value: "
                        f"alpha={float(nearest['alpha']):.12g}, "
                        f"ci={float(nearest['ci']):.12g}, "
                        f"ci-target={float(nearest['f_ci']):.6e}."
                    ),
                    n_spectral_solves=len(rows),
                )
            else:
                result = failure_result(
                    target=target,
                    status="no_successful_spectral_solve",
                    message="No accepted classical root was found at any scanned alpha.",
                    n_spectral_solves=0,
                )
    except Exception as exc:
        result = failure_result(
            target=target,
            status="exception",
            message=f"{type(exc).__name__}: {exc}",
            n_spectral_solves=len(evaluator.successful()),
        )

    result["true_isoline_task_index"] = args.task_index
    pd.DataFrame([result]).to_csv(point_root / "result.csv", index=False)
    pd.DataFrame(evaluator.successful()).to_csv(point_root / "evaluations.csv", index=False)
    pd.DataFrame(evaluator.attempt_rows).to_csv(point_root / "solver_attempts.csv", index=False)
    (point_root / "metadata.json").write_text(
        json.dumps(campaign.json_safe(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== TRUE POSITIVE BLUMEN ISOLINE POINT ===")
    print(f"Task index       : {args.task_index}")
    print(f"Blumen row       : {target['blumen_row_id']}")
    print(f"Curve            : {target['curve_key']}")
    print(f"Mach             : {Mach}")
    print(f"Target ci        : {target_ci}")
    print(f"Blumen alpha     : {alpha_blumen}")
    print(f"Status           : {result['status']}")
    print(f"Classical alpha  : {result.get('alpha_classical')}")
    print(f"Delta alpha      : {result.get('delta_alpha')}")
    print(f"Spectral solves  : {result.get('n_spectral_solves')}")

    # Numerical misses are recorded in result.csv and audited by the asset job.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
