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
            "Repair the low-M/low-alpha extension by detecting the available "
            "converged seed for each alpha and continuing in Mach in either direction."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
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


def latest_target_row(
    path: Path,
    alpha: float,
    *,
    converged_only: bool,
) -> dict[str, Any] | None:
    frame = campaign.load_frame(path)
    if frame.empty:
        return None

    alpha_values = pd.to_numeric(frame.get("alpha"), errors="coerce")
    target_mask = frame.get("is_target", pd.Series(False, index=frame.index))
    target_mask = target_mask.astype(str).str.lower().isin(("true", "1", "yes"))
    mask = target_mask & np.isclose(
        alpha_values,
        float(alpha),
        rtol=0.0,
        atol=5.0e-12,
    )

    if converged_only:
        cr = pd.to_numeric(frame.get("cr"), errors="coerce")
        ci = pd.to_numeric(frame.get("ci"), errors="coerce")
        status = frame.get("status", pd.Series("", index=frame.index)).astype(str)
        mask &= status.isin(CONVERGED) & np.isfinite(cr) & np.isfinite(ci) & (ci > 0.0)

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

    ratio = (float(trial_mach) - float(last["Mach"])) / denominator
    cr = float(last["cr"]) + ratio * (float(last["cr"]) - float(previous["cr"]))

    last_ci = float(last["ci"])
    previous_ci = float(previous["ci"])
    if min(last_ci, previous_ci) > 5.0e-3:
        log_ci = math.log(last_ci) + ratio * (math.log(last_ci) - math.log(previous_ci))
        ci = math.exp(log_ci)
    else:
        ci = last_ci + ratio * (last_ci - previous_ci)
    return cr, max(float(ci), 10.0 * np.finfo(float).eps)


def solve_between_machs(
    *,
    alpha: float,
    target_mach: float,
    source_row: dict[str, Any],
    second_row: dict[str, Any] | None,
    config: dict[str, Any],
    attempts_path: Path,
    minimum_step: float,
    growth: float,
    max_attempts: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source_mach = float(source_row["Mach"])
    target_mach = float(target_mach)
    total = target_mach - source_mach
    if abs(total) <= 1.0e-13:
        return source_row, None

    history: list[dict[str, Any]] = []
    if second_row is not None:
        history.append(second_row)
    history.append(source_row)

    current_mach = source_mach
    step = total
    loops = 0
    last_failure: dict[str, Any] | None = None

    while abs(target_mach - current_mach) > 1.0e-13:
        loops += 1
        if loops > max_attempts:
            break

        remaining = target_mach - current_mach
        if abs(step) > abs(remaining):
            step = remaining
        if step * remaining <= 0.0:
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
                    "purpose": "mach_continuation_repair_auto_source",
                    "continuation_axis": "Mach",
                    "requested_Mach": float(target_mach),
                    "seed_Mach": float(current_mach),
                    "trial_Mach": float(trial_mach),
                    "Mach_step": float(step),
                    "requested_alpha": float(alpha),
                    "source_Mach": float(source_mach),
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
            return result, last_failure

        remaining = target_mach - current_mach
        step = math.copysign(min(abs(remaining), abs(step) * growth), remaining)

    return None, last_failure


def choose_second_source(
    known: dict[float, dict[str, Any]],
    primary_mach: float,
    target_mach: float,
) -> dict[str, Any] | None:
    alternatives = [m for m in known if not np.isclose(m, primary_mach, atol=1.0e-13)]
    if not alternatives:
        return None
    # Prefer a second point on the continuation side opposite the target so that
    # the last two history entries are ordered toward the target.
    direction = math.copysign(1.0, target_mach - primary_mach)
    behind = [m for m in alternatives if (primary_mach - m) * direction > 0.0]
    pool = behind if behind else alternatives
    second_mach = min(pool, key=lambda m: abs(m - primary_mach))
    return known[second_mach]


def source_coverage(
    output_root: Path,
    mach_values: list[float],
    alphas: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for Mach in mach_values:
        path = mach_dir(output_root, Mach) / "spectral_points.csv"
        count = sum(
            latest_target_row(path, float(alpha), converged_only=True) is not None
            for alpha in alphas
        )
        rows.append({"Mach": Mach, "converged_targets": int(count)})
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    config_path = resolve(repo, args.config.expanduser())
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve(repo, Path(str(config["output_root"])))
    output_root.mkdir(parents=True, exist_ok=True)

    mach_values = sorted({float(value) for value in config["mach_values"]})
    alphas = alpha_grid(config)

    coverage = source_coverage(output_root, mach_values, alphas)
    campaign.atomic_write_csv(output_root / "pre_repair_source_coverage.csv", coverage)

    print("=== AUTO-SOURCE MACH-CONTINUATION REPAIR ===", flush=True)
    print(coverage.to_string(index=False), flush=True)
    print(f"Alpha points: {len(alphas)}", flush=True)

    repaired: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for alpha_index, alpha_value in enumerate(alphas, start=1):
        alpha = float(alpha_value)
        known: dict[float, dict[str, Any]] = {}
        for Mach in mach_values:
            row = latest_target_row(
                mach_dir(output_root, Mach) / "spectral_points.csv",
                alpha,
                converged_only=True,
            )
            if row is not None:
                known[Mach] = row

        print(
            f"\n[alpha {alpha_index}/{len(alphas)}] alpha={alpha:.12g}; "
            f"available seeds={sorted(known)}",
            flush=True,
        )

        if not known:
            failures.append(
                {
                    "Mach": np.nan,
                    "alpha": alpha,
                    "reason": "no converged source at this alpha",
                }
            )
            print("  FAILED: no converged source", flush=True)
            continue

        missing = [Mach for Mach in mach_values if Mach not in known]
        while missing:
            # Grow from the existing solved set to the nearest unresolved grid node.
            target_mach = min(
                missing,
                key=lambda target: min(abs(target - source) for source in known),
            )
            ordered_sources = sorted(known, key=lambda source: abs(target_mach - source))
            solved = False
            last_failure: dict[str, Any] | None = None

            for source_mach in ordered_sources:
                source_row = known[source_mach]
                second_row = choose_second_source(known, source_mach, target_mach)
                target_directory = mach_dir(output_root, target_mach)
                target_directory.mkdir(parents=True, exist_ok=True)
                points_path = target_directory / "spectral_points.csv"
                attempts_path = target_directory / "solver_attempts.csv"

                result, last_failure = solve_between_machs(
                    alpha=alpha,
                    target_mach=target_mach,
                    source_row=source_row,
                    second_row=second_row,
                    config=config,
                    attempts_path=attempts_path,
                    minimum_step=float(args.minimum_mach_step),
                    growth=float(args.bridge_growth),
                    max_attempts=int(args.max_bridge_attempts),
                )
                if result is None:
                    continue

                anchor = {
                    "source_path": str(mach_dir(output_root, source_mach) / "spectral_points.csv"),
                    "source_Mach": float(source_mach),
                    "source_alpha": float(alpha),
                    "source_cr": float(source_row["cr"]),
                    "source_ci": float(source_row["ci"]),
                }
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
                        "seed_Mach": float(source_mach),
                        "requested_Mach": float(target_mach),
                        "Mach_step": float(target_mach - source_mach),
                        "repair_strategy": "nearest_available_source_bidirectional",
                    }
                )
                campaign.replace_point_row(points_path, row)
                known[target_mach] = row
                repaired.append(
                    {
                        "Mach": float(target_mach),
                        "alpha": float(alpha),
                        "source_Mach": float(source_mach),
                        "cr": float(result["cr"]),
                        "ci": float(result["ci"]),
                        "residual_norm": float(result["residual_norm"]),
                    }
                )
                missing.remove(target_mach)
                solved = True
                print(
                    f"  M={target_mach:.6f} from M={source_mach:.6f}: "
                    f"c={float(result['cr']):.12g}+{float(result['ci']):.6e}i, "
                    f"res={float(result['residual_norm']):.3e}",
                    flush=True,
                )
                break

            if not solved:
                failures.append(
                    {
                        "Mach": float(target_mach),
                        "alpha": float(alpha),
                        "reason": "all available Mach seeds failed",
                        "last_residual": (
                            last_failure.get("residual_norm") if last_failure else np.nan
                        ),
                    }
                )
                print(f"  M={target_mach:.6f}: FAILED from all sources", flush=True)
                break

    repaired_frame = pd.DataFrame(repaired)
    failures_frame = pd.DataFrame(failures)
    campaign.atomic_write_csv(
        output_root / "mach_continuation_repaired_points.csv",
        repaired_frame,
    )
    campaign.atomic_write_csv(
        output_root / "mach_continuation_failures.csv",
        failures_frame,
    )

    final_coverage = source_coverage(output_root, mach_values, alphas)
    campaign.atomic_write_csv(output_root / "post_repair_source_coverage.csv", final_coverage)
    complete = bool((final_coverage["converged_targets"] == len(alphas)).all())

    print("\n=== REPAIR SUMMARY ===", flush=True)
    print(f"Newly repaired points: {len(repaired_frame)}", flush=True)
    print(f"Failures             : {len(failures_frame)}", flush=True)
    print("Final coverage:", flush=True)
    print(final_coverage.to_string(index=False), flush=True)
    print(f"REPAIR STATUS        : {'PASS' if complete else 'FAIL'}", flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
