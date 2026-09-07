#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path.cwd().resolve()

PHASE6_WORKER = (
    REPO
    / "scripts/pinn_supersonic/reviewer_runs/"
      "run_budget_matched_v64.py"
)

PREDICTIONS = (
    REPO
    / "assets/pinn_supersonic/"
      "atlas2d_v1_continuousM/N76/"
      "test401/N76_test401_predictions_401.csv"
)

SHOOTING = (
    REPO
    / "assets/pinn_supersonic/"
      "atlas2d_v1_continuousM/N76/"
      "shooting_T401/N76_T401_shooting_401.csv"
)

FAIL_MACH = 1.15
FAIL_ALPHA = 0.06

CR_MIN = 0.0
CR_MAX = 0.12
N_CR = 25

CI_MIN = 0.06
CI_MAX = 0.24
N_CI = 31

ROOT_TOL = 1.0e-4


def load_phase6():
    spec = importlib.util.spec_from_file_location(
        "phase6_worker",
        PHASE6_WORKER,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {PHASE6_WORKER}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def atomic_csv(
    path: Path,
    frame: pd.DataFrame,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(path)


def find_unique(
    frame: pd.DataFrame,
    *,
    mach: float,
    alpha: float,
):
    mask = (
        np.isclose(
            pd.to_numeric(
                frame["Mach"]
            ),
            mach,
            rtol=0.0,
            atol=1.0e-12,
        )
        &
        np.isclose(
            pd.to_numeric(
                frame["alpha"]
            ),
            alpha,
            rtol=0.0,
            atol=1.0e-12,
        )
    )

    subset = frame.loc[mask]

    if len(subset) != 1:
        raise RuntimeError(
            f"Expected one row for "
            f"M={mach}, alpha={alpha}; "
            f"found {len(subset)}"
        )

    return subset.iloc[0]


def classify_root(
    *,
    cr: float,
    ci: float,
    cr_target: float,
    ci_target: float,
    cr_failure: float,
    ci_failure: float,
):
    d_target = float(
        np.hypot(
            cr - cr_target,
            ci - ci_target,
        )
    )

    d_failure = float(
        np.hypot(
            cr - cr_failure,
            ci - ci_failure,
        )
    )

    if d_target <= ROOT_TOL:
        label = "target"

    elif d_failure <= ROOT_TOL:
        label = "failure_branch"

    else:
        label = "other"

    return (
        label,
        d_target,
        d_failure,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--start-index",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--stop-index",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    if not PREDICTIONS.is_file():
        raise FileNotFoundError(
            PREDICTIONS
        )

    if not SHOOTING.is_file():
        raise FileNotFoundError(
            SHOOTING
        )

    pred = pd.read_csv(
        PREDICTIONS
    )

    shoot = pd.read_csv(
        SHOOTING
    )

    source = find_unique(
        pred,
        mach=FAIL_MACH,
        alpha=FAIL_ALPHA,
    )

    historical = find_unique(
        shoot,
        mach=FAIL_MACH,
        alpha=FAIL_ALPHA,
    )

    cr_target = float(
        source["cr"]
    )

    ci_target = float(
        source["ci"]
    )

    cr_pinn = float(
        source["cr_pred"]
    )

    ci_pinn = float(
        source["ci_pred"]
    )

    cr_failure = float(
        historical["shoot_cr"]
    )

    ci_failure = float(
        historical["shoot_ci"]
    )

    print("=" * 100)
    print("PHASE 8 — T401 FAILURE BASIN")
    print("=" * 100)

    print(
        f"M, alpha       = "
        f"{FAIL_MACH}, {FAIL_ALPHA}"
    )

    print(
        f"target         = "
        f"{cr_target:.10f} "
        f"+ {ci_target:.10f} i"
    )

    print(
        f"PINN seed      = "
        f"{cr_pinn:.10f} "
        f"+ {ci_pinn:.10f} i"
    )

    print(
        f"failure root   = "
        f"{cr_failure:.10f} "
        f"+ {ci_failure:.10f} i"
    )

    phase6 = load_phase6()

    shooting_module, defaults = (
        phase6.load_shooting()
    )

    cr_values = np.linspace(
        CR_MIN,
        CR_MAX,
        N_CR,
    )

    ci_values = np.linspace(
        CI_MIN,
        CI_MAX,
        N_CI,
    )

    grid = [
        (
            i_cr,
            i_ci,
            float(cr),
            float(ci),
        )
        for i_ci, ci
        in enumerate(ci_values)
        for i_cr, cr
        in enumerate(cr_values)
    ]

    n_total = len(grid)

    if n_total != N_CR * N_CI:
        raise RuntimeError(
            "Grid construction failed"
        )

    if not (
        0
        <= args.start_index
        < args.stop_index
        <= n_total
    ):
        raise ValueError(
            f"Invalid range "
            f"[{args.start_index},"
            f"{args.stop_index}) "
            f"for n={n_total}"
        )

    print(
        f"grid           = "
        f"{N_CR} x {N_CI} "
        f"= {n_total}"
    )

    print(
        f"cr range       = "
        f"[{CR_MIN}, {CR_MAX}]"
    )

    print(
        f"ci range       = "
        f"[{CI_MIN}, {CI_MAX}]"
    )

    print(
        "local box      = "
        "(0.015, 0.008)"
    )

    print(
        "max_retries    = 0"
    )

    print(
        f"chunk          = "
        f"[{args.start_index},"
        f"{args.stop_index})"
    )

    if args.dry_run:
        for item in grid[
            args.start_index:
            min(
                args.stop_index,
                args.start_index + 10,
            )
        ]:
            print(item)

        return

    rows = []

    for grid_index in range(
        args.start_index,
        args.stop_index,
    ):

        (
            i_cr,
            i_ci,
            cr_center,
            ci_center,
        ) = grid[
            grid_index
        ]

        t0 = time.perf_counter()

        row = {
            "grid_index":
                grid_index,

            "i_cr":
                i_cr,

            "i_ci":
                i_ci,

            "Mach":
                FAIL_MACH,

            "alpha":
                FAIL_ALPHA,

            "cr_center":
                cr_center,

            "ci_center":
                ci_center,

            "cr_target":
                cr_target,

            "ci_target":
                ci_target,

            "cr_pinn":
                cr_pinn,

            "ci_pinn":
                ci_pinn,

            "cr_failure_reference":
                cr_failure,

            "ci_failure_reference":
                ci_failure,

            "status":
                "FAILED",

            "technical_success":
                False,

            "root_class":
                "technical_failure",

            "shoot_cr":
                np.nan,

            "shoot_ci":
                np.nan,

            "stage1_mismatch":
                np.nan,

            "stage2_mismatch":
                np.nan,

            "total_mismatch":
                np.nan,

            "distance_target":
                np.nan,

            "distance_failure":
                np.nan,

            "error":
                "",
        }

        try:
            (
                solver,
                result,
                retry_idx,
                used_cr_half,
                used_ci_half,
            ) = (
                shooting_module
                .multistart_single_box(
                    alpha=FAIL_ALPHA,
                    mach=FAIL_MACH,

                    match_y=float(
                        defaults[
                            "match_y"
                        ]
                    ),

                    use_mapping=True,
                    mapping_scale=5.0,

                    min_y_limit=float(
                        defaults[
                            "min_y_limit"
                        ]
                    ),

                    max_y_limit=float(
                        defaults[
                            "max_y_limit"
                        ]
                    ),

                    y_limit_factor=float(
                        defaults[
                            "y_limit_factor"
                        ]
                    ),

                    amp_lower_bound=float(
                        defaults[
                            "amp_lower_bound"
                        ]
                    ),

                    amp_upper_bound=float(
                        defaults[
                            "amp_upper_bound"
                        ]
                    ),

                    cr_center=
                        cr_center,

                    ci_center=max(
                        1.0e-4,
                        ci_center,
                    ),

                    # Smallest production box only.
                    cr_half_window=
                        0.015,

                    ci_half_window=
                        0.008,

                    retry_growth=float(
                        defaults[
                            "retry_growth"
                        ]
                    ),

                    # Critical for a local basin map:
                    # no adaptive widening.
                    max_retries=0,

                    max_iter=int(
                        defaults[
                            "max_iter"
                        ]
                    ),

                    grid_size=int(
                        defaults[
                            "grid_size"
                        ]
                    ),
                )
            )

            cr = float(
                result.cr
            )

            ci = float(
                result.ci
            )

            technical = bool(
                result.spectral_success
                and
                result.mode_success
            )

            (
                root_class,
                d_target,
                d_failure,
            ) = classify_root(
                cr=cr,
                ci=ci,
                cr_target=cr_target,
                ci_target=ci_target,
                cr_failure=cr_failure,
                ci_failure=ci_failure,
            )

            if not technical:
                root_class = (
                    "technical_failure"
                )

            row.update({
                "status":
                    "COMPLETED",

                "technical_success":
                    technical,

                "root_class":
                    root_class,

                "shoot_cr":
                    cr,

                "shoot_ci":
                    ci,

                "stage1_mismatch":
                    float(
                        result
                        .stage1_mismatch
                    ),

                "stage2_mismatch":
                    float(
                        result
                        .stage2_mismatch
                    ),

                "total_mismatch":
                    float(
                        result
                        .stage1_mismatch
                        +
                        result
                        .stage2_mismatch
                    ),

                "distance_target":
                    d_target,

                "distance_failure":
                    d_failure,
            })

        except Exception as exc:
            row["error"] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        row["seconds"] = (
            time.perf_counter()
            - t0
        )

        rows.append(
            row
        )

        print(
            f"{grid_index:04d}/"
            f"{n_total:04d} "
            f"center=("
            f"{cr_center:.5f},"
            f"{ci_center:.5f}) "
            f"class="
            f"{row['root_class']} "
            f"root=("
            f"{row['shoot_cr']},"
            f"{row['shoot_ci']}) "
            f"time="
            f"{row['seconds']:.1f}s",
            flush=True,
        )

    output_root = (
        args.output_root.resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        output_root
        / (
            f"basin_"
            f"{args.start_index:04d}_"
            f"{args.stop_index:04d}.csv"
        )
    )

    atomic_csv(
        output,
        pd.DataFrame(rows),
    )

    print()
    print(
        "WROTE:",
        output,
    )


if __name__ == "__main__":
    main()
