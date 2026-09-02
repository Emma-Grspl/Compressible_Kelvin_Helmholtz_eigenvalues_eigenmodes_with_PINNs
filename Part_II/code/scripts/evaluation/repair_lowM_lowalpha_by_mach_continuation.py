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


CONVERGED = {"converged", "anchor_converged"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair the low-M/low-alpha extension by continuing each fixed-alpha "
            "eigenvalue from M=1.10 downwards in Mach."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-mach", type=float, default=None)
    parser.add_argument("--minimum-mach-step", type=float, default=2.5e-4)
    parser.add_argument("--bridge-growth", type=float, default=1.5)
    parser.add_argument("--max-bridge-attempts", type=int, default=120)
    return parser.parse_args()


def resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def mach_dir(output_root: Path, Mach: float) -> Path:
    return output_root / f"M{Mach:.6f}".replace(".", "p")


def alpha_grid(config: dict[str, Any]) -> np.ndarray:
    return campaign.decimal_grid(
        float(config["alpha_min"]),
        float(config["alpha_max"]),
        float(config["alpha_step"]),
    )


def latest_target_row(path: Path, alpha: float, converged_only: bool) -> dict[str, Any] | None:
    frame = campaign.load_frame(path)
    if frame.empty:
        return None
    mask = (
        frame["is_target"].astype(str).str.lower().isin(("true", "1"))
        & np.isclose(
            pd.to_numeric(frame["alpha"], errors="coerce"),
            float(alpha),
            rtol=0.0,
            atol=5.0e-12,
        )
    )
    if converged_only:
        cr = pd.to_numeric(frame["cr"], errors="coerce")
        ci = pd.to_numeric(frame["ci"], errors="coerce")
        mask &= (
            frame["status"].astype(str).isin(CONVERGED)
            & np.isfinite(cr)
            & np.isfinite(ci)
            & (ci > 0.0)
        )
    subset = frame.loc[mask].copy()
    if subset.empty:
        return None
    if "timestamp" in subset.columns:
        subset = subset.sort_values("timestamp", kind="stable")
    return subset.iloc[-1].to_dict()


def predictor(history: list[dict[str, Any]], trial_mach: float) -> tuple[float, float]:
    last = history[-1]
    if len(history) < 2:
        return float(last["cr"]), float(last["ci"])
    previous = history[-2]
    denominator = float(last["Mach"]) - float(previous["Mach"])
    if abs(denominator) < 1.0e-14:
        return float(last["cr"]), float(last["ci"])
    weight = (float(trial_mach) - float(last["Mach"])) / denominator
    cr = float(last["cr"]) + weight * (float(last["cr"]) - float(previous["cr"]))
    ci = float(last["ci"]) + weight * (float(last["ci"]) - float(previous["ci"]))
    return cr, max(ci, 10.0 * np.finfo(float).eps)


def solve_to_target_mach(
    *,
    alpha: float,
    target_mach: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    attempts_path: Path,
    minimum_step: float,
    growth: float,
    max_attempts: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    current_mach = float(history[-1]["Mach"])
    if target_mach >= current_mach - 1.0e-13:
        raise ValueError("Mach continuation must proceed downwards.")

    step = target_mach - current_mach
    loops = 0
    last_failure: dict[str, Any] | None = None

    while current_mach - target_mach > 1.0e-13:
        loops += 1
        if loops > max_attempts:
            break
        remaining = target_mach - current_mach
        if abs(step) > abs(remaining):
            step = remaining
        trial_mach = current_mach + step
        seed_cr, seed_ci = predictor(history, trial_mach)
        result, attempts = campaign.solve_with_fallbacks(
            Mach=float(trial_mach),
            alpha=float(alpha),
            seed_cr=float(seed_cr),
            seed_ci=float(seed_ci),
            config=config,
        )
        for attempt in attempts:
            attempt.update(
                {
                    "purpose": "mach_continuation_repair",
                    "continuation_axis": "Mach",
                    "requested_Mach": float(target_mach),
                    "seed_Mach": float(current_mach),
                    "trial_Mach": float(trial_mach),
                    "Mach_step": float(step),
                    "requested_alpha": float(alpha),
                }
            )
        campaign.append_rows(attempts_path, attempts)

        if result is None:
            last_failure = attempts[-1] if attempts else None
            step *= 0.5
            if abs(step) < minimum_step:
                break
            continue

        history.append(result)
        current_mach = float(trial_mach)
        if abs(current_mach - target_mach) <= 1.0e-13:
            return result, history, last_failure

        remaining = target_mach - current_mach
        step = -min(abs(remaining), abs(step) * growth)

    return None, history, last_failure


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    config_path = resolve(repo, args.config.expanduser())
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve(repo, Path(str(config["output_root"])))

    mach_values = sorted({float(value) for value in config["mach_values"]}, reverse=True)
    source_mach = float(args.source_mach) if args.source_mach is not None else max(mach_values)
    if source_mach not in mach_values:
        raise ValueError(f"Source Mach {source_mach} is not in configured mach_values.")
    targets = [value for value in mach_values if value < source_mach - 1.0e-13]
    alphas = alpha_grid(config)

    source_points = mach_dir(output_root, source_mach) / "spectral_points.csv"
    missing_source = [
        float(alpha)
        for alpha in alphas
        if latest_target_row(source_points, float(alpha), converged_only=True) is None
    ]
    if missing_source:
        raise RuntimeError(
            f"Source M={source_mach} is missing converged roots at alpha={missing_source}."
        )

    print("=== MACH-CONTINUATION REPAIR ===", flush=True)
    print(f"Source Mach       : {source_mach}", flush=True)
    print(f"Target Machs      : {targets}", flush=True)
    print(f"Alpha points      : {len(alphas)}", flush=True)

    failures: list[dict[str, Any]] = []
    repaired: list[dict[str, Any]] = []

    for alpha_index, alpha_value in enumerate(alphas, start=1):
        alpha = float(alpha_value)
        source = latest_target_row(source_points, alpha, converged_only=True)
        assert source is not None
        history: list[dict[str, Any]] = [source]
        print(
            f"\n[alpha {alpha_index}/{len(alphas)}] alpha={alpha:.12g}; "
            f"source c={float(source['cr']):.12g}+{float(source['ci']):.6e}i",
            flush=True,
        )

        for target_mach in targets:
            target_dir = mach_dir(output_root, target_mach)
            target_dir.mkdir(parents=True, exist_ok=True)
            points_path = target_dir / "spectral_points.csv"
            attempts_path = target_dir / "solver_attempts.csv"

            existing = latest_target_row(points_path, alpha, converged_only=True)
            if existing is not None:
                history.append(existing)
                print(
                    f"  M={target_mach:.6f}: already converged, "
                    f"c={float(existing['cr']):.12g}+{float(existing['ci']):.6e}i",
                    flush=True,
                )
                continue

            seed_mach = float(history[-1]["Mach"])
            result, history, last_failure = solve_to_target_mach(
                alpha=alpha,
                target_mach=float(target_mach),
                history=history,
                config=config,
                attempts_path=attempts_path,
                minimum_step=float(args.minimum_mach_step),
                growth=float(args.bridge_growth),
                max_attempts=int(args.max_bridge_attempts),
            )

            anchor = {
                "source_path": str(source_points),
                "source_Mach": float(source_mach),
                "source_alpha": float(alpha),
                "source_cr": float(source["cr"]),
                "source_ci": float(source["ci"]),
            }

            if result is None:
                message = "Mach continuation could not reach target above minimum step."
                if last_failure is not None:
                    message += f" Last residual={last_failure.get('residual_norm', np.nan)}"
                row = campaign.status_row(
                    Mach=float(target_mach),
                    alpha=float(alpha),
                    direction="low",
                    status="rejected",
                    requested_alpha=float(alpha),
                    anchor=anchor,
                    neutral=None,
                    message=message,
                )
                row.update(
                    {
                        "continuation_axis": "Mach",
                        "seed_Mach": seed_mach,
                        "requested_Mach": float(target_mach),
                        "Mach_step": float(target_mach - seed_mach),
                    }
                )
                campaign.replace_point_row(points_path, row)
                failures.append(
                    {
                        "Mach": float(target_mach),
                        "alpha": float(alpha),
                        "last_residual": (
                            last_failure.get("residual_norm") if last_failure else np.nan
                        ),
                    }
                )
                print(f"  M={target_mach:.6f}: FAILED", flush=True)
                break

            row = campaign.point_row(
                result=result,
                direction="low",
                is_target=True,
                status="converged",
                seed_alpha=float(alpha),
                requested_alpha=float(alpha),
                alpha_step=0.0,
                anchor=anchor,
                neutral=None,
            )
            row.update(
                {
                    "continuation_axis": "Mach",
                    "seed_Mach": seed_mach,
                    "requested_Mach": float(target_mach),
                    "Mach_step": float(target_mach - seed_mach),
                }
            )
            campaign.replace_point_row(points_path, row)
            repaired.append(
                {
                    "Mach": float(target_mach),
                    "alpha": float(alpha),
                    "cr": float(result["cr"]),
                    "ci": float(result["ci"]),
                    "residual_norm": float(result["residual_norm"]),
                }
            )
            print(
                f"  M={target_mach:.6f}: c={float(result['cr']):.12g}"
                f"+{float(result['ci']):.6e}i, "
                f"res={float(result['residual_norm']):.3e}",
                flush=True,
            )

    repaired_frame = pd.DataFrame(repaired)
    failures_frame = pd.DataFrame(failures)
    campaign.atomic_write_csv(output_root / "mach_continuation_repaired_points.csv", repaired_frame)
    campaign.atomic_write_csv(output_root / "mach_continuation_failures.csv", failures_frame)

    expected_repairs = len(targets) * len(alphas)
    print("\n=== REPAIR SUMMARY ===", flush=True)
    print(f"Expected repairs : {expected_repairs}", flush=True)
    print(f"Converged repairs: {len(repaired_frame)}", flush=True)
    print(f"Failures         : {len(failures_frame)}", flush=True)
    print(f"Written to       : {output_root}", flush=True)

    if failures or len(repaired_frame) != expected_repairs:
        return 2
    print("MACH-CONTINUATION REPAIR: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
