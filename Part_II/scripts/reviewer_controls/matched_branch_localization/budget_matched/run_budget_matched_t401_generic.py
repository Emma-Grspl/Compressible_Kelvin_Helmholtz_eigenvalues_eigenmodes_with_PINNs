#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path.cwd().resolve()

PHASE6_WORKER = (
    REPO
    / "scripts/pinn_supersonic/reviewer_runs/"
      "run_budget_matched_v64.py"
)


def load_phase6():
    if not PHASE6_WORKER.is_file():
        raise FileNotFoundError(PHASE6_WORKER)

    spec = importlib.util.spec_from_file_location(
        "phase6_budget_matched_worker",
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--stop",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    predictions = args.predictions.resolve()
    output_root = args.output_root.resolve()

    if not predictions.is_file():
        raise FileNotFoundError(
            predictions
        )

    frame = (
        pd.read_csv(predictions)
        .reset_index(drop=True)
    )

    if len(frame) != 401:
        raise RuntimeError(
            f"Expected T401 with 401 rows, "
            f"found {len(frame)}"
        )

    required = {
        "test_id",
        "Mach",
        "alpha",
        "cr",
        "ci",
        "cr_pred",
        "ci_pred",
        "atlas_chart",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    if frame["test_id"].nunique() != 401:
        raise RuntimeError(
            "test_id is not globally unique"
        )

    if (
        frame[
            ["Mach", "alpha"]
        ]
        .drop_duplicates()
        .shape[0]
        != 401
    ):
        raise RuntimeError(
            "(Mach,alpha) is not globally unique"
        )

    if not (
        0
        <= args.start
        < args.stop
        <= 401
    ):
        raise ValueError(
            f"Invalid range [{args.start},{args.stop})"
        )

    phase6 = load_phase6()

    shooting, defaults = (
        phase6.load_shooting()
    )

    pairs = phase6.window_pairs(
        defaults
    )

    if len(pairs) != 12:
        raise RuntimeError(
            f"Expected 12 window pairs; got {len(pairs)}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    part = frame.iloc[
        args.start:args.stop
    ].copy()

    # --------------------------------------------------------------
    # DRY RUN
    # --------------------------------------------------------------

    if args.dry_run:

        print(
            f"T401 rows = {len(frame)}"
        )
        print(
            f"range = [{args.start},{args.stop})"
        )
        print(
            f"window pairs = {len(pairs)}"
        )

        for global_index, source in (
            part.head(2).iterrows()
        ):

            specs = phase6.candidate_specs(
                condition="generic",
                source=source,
                # Global frozen-row index controls only
                # deterministic rotation of window/center
                # association.
                benchmark_id=int(
                    global_index
                ),
                defaults=defaults,
            )

            print()
            print(
                "global_index:",
                global_index,
            )
            print(
                "test_id:",
                source["test_id"],
            )
            print(
                "M,alpha:",
                float(source["Mach"]),
                float(source["alpha"]),
            )

            for spec in specs:
                print(spec)

        return

    # --------------------------------------------------------------
    # RUN
    # --------------------------------------------------------------

    candidate_rows = []
    summary_rows = []

    for global_index, source in part.iterrows():

        global_index = int(
            global_index
        )

        test_id = source[
            "test_id"
        ]

        mach = float(
            source["Mach"]
        )

        alpha = float(
            source["alpha"]
        )

        cr_ref = float(
            source["cr"]
        )

        ci_ref = float(
            source["ci"]
        )

        print()
        print("=" * 110)
        print(
            f"GENERIC T401 "
            f"row={global_index:03d} "
            f"test_id={test_id} "
            f"M={mach:.3f} "
            f"alpha={alpha:.3f}",
            flush=True,
        )

        specs = phase6.candidate_specs(
            condition="generic",
            source=source,
            benchmark_id=global_index,
            defaults=defaults,
        )

        if len(specs) != 12:
            raise RuntimeError(
                f"row={global_index}: "
                f"expected 12 specs"
            )

        local = []

        for spec in specs:

            result = phase6.solve_candidate(
                shooting=shooting,
                defaults=defaults,
                source=source,
                benchmark_id=global_index,
                condition="generic",
                spec=spec,
            )

            result[
                "t401_row"
            ] = global_index

            result[
                "test_id"
            ] = test_id

            local.append(
                result
            )

            candidate_rows.append(
                result
            )

            print(
                f"  box="
                f"{result['candidate_index']:02d} "
                f"center=("
                f"{result['cr_center']:.4f},"
                f"{result['ci_center']:.4g}) "
                f"retry="
                f"{result['retry_index']} "
                f"tech="
                f"{result['technical_success']} "
                f"mismatch="
                f"{result['total_mismatch']}",
                flush=True,
            )

        best = phase6.choose_best(
            local
        )

        if best is None:

            summary = {
                "t401_row":
                    global_index,

                "test_id":
                    test_id,

                "atlas_chart":
                    str(
                        source[
                            "atlas_chart"
                        ]
                    ),

                "Mach":
                    mach,

                "alpha":
                    alpha,

                "cr_reference":
                    cr_ref,

                "ci_reference":
                    ci_ref,

                "cr_pinn":
                    float(
                        source["cr_pred"]
                    ),

                "ci_pinn":
                    float(
                        source["ci_pred"]
                    ),

                "status":
                    "FAILED",

                "technical_success":
                    False,

                "target_error":
                    np.nan,

                "target_recovery":
                    False,

                "n_boxes":
                    len(local),

                "n_completed_boxes":
                    int(
                        sum(
                            r["status"]
                            == "COMPLETED"
                            for r in local
                        )
                    ),

                "n_technical_boxes":
                    int(
                        sum(
                            bool(
                                r[
                                    "technical_success"
                                ]
                            )
                            for r in local
                        )
                    ),

                "total_seconds":
                    float(
                        sum(
                            float(
                                r["seconds"]
                            )
                            for r in local
                        )
                    ),
            }

        else:

            target_error = float(
                np.hypot(
                    float(
                        best[
                            "shoot_cr"
                        ]
                    )
                    - cr_ref,
                    float(
                        best[
                            "shoot_ci"
                        ]
                    )
                    - ci_ref,
                )
            )

            retries = [
                int(
                    r["retry_index"]
                )
                for r in local
                if (
                    r["status"]
                    == "COMPLETED"
                    and
                    np.isfinite(
                        float(
                            r[
                                "retry_index"
                            ]
                        )
                    )
                )
            ]

            summary = {
                "t401_row":
                    global_index,

                "test_id":
                    test_id,

                "atlas_chart":
                    str(
                        source[
                            "atlas_chart"
                        ]
                    ),

                "Mach":
                    mach,

                "alpha":
                    alpha,

                "cr_reference":
                    cr_ref,

                "ci_reference":
                    ci_ref,

                "cr_pinn":
                    float(
                        source["cr_pred"]
                    ),

                "ci_pinn":
                    float(
                        source["ci_pred"]
                    ),

                "status":
                    "COMPLETED",

                "technical_success":
                    bool(
                        best[
                            "technical_success"
                        ]
                    ),

                "target_error":
                    target_error,

                "target_recovery":
                    bool(
                        target_error
                        <= 1.0e-4
                    ),

                "best_candidate_index":
                    int(
                        best[
                            "candidate_index"
                        ]
                    ),

                "best_center_index":
                    int(
                        best[
                            "center_index"
                        ]
                    ),

                "best_cr_center":
                    float(
                        best[
                            "cr_center"
                        ]
                    ),

                "best_ci_center":
                    float(
                        best[
                            "ci_center"
                        ]
                    ),

                "best_cr":
                    float(
                        best[
                            "shoot_cr"
                        ]
                    ),

                "best_ci":
                    float(
                        best[
                            "shoot_ci"
                        ]
                    ),

                "best_stage1_mismatch":
                    float(
                        best[
                            "stage1_mismatch"
                        ]
                    ),

                "best_stage2_mismatch":
                    float(
                        best[
                            "stage2_mismatch"
                        ]
                    ),

                "best_total_mismatch":
                    float(
                        best[
                            "total_mismatch"
                        ]
                    ),

                "best_retry_index":
                    int(
                        best[
                            "retry_index"
                        ]
                    ),

                "n_boxes":
                    len(local),

                "n_completed_boxes":
                    int(
                        sum(
                            r["status"]
                            == "COMPLETED"
                            for r in local
                        )
                    ),

                "n_technical_boxes":
                    int(
                        sum(
                            bool(
                                r[
                                    "technical_success"
                                ]
                            )
                            for r in local
                        )
                    ),

                # Retry 0 = one actual solve.
                "total_box_solves":
                    int(
                        sum(
                            retry + 1
                            for retry
                            in retries
                        )
                    ),

                "total_seconds":
                    float(
                        sum(
                            float(
                                r["seconds"]
                            )
                            for r in local
                        )
                    ),
            }

        summary_rows.append(
            summary
        )

        print(
            "BEST:",
            f"technical="
            f"{summary['technical_success']} "
            f"target_error="
            f"{summary['target_error']} "
            f"target="
            f"{summary['target_recovery']}",
            flush=True,
        )

    stem = (
        f"{args.start:03d}_"
        f"{args.stop:03d}"
    )

    atomic_csv(
        output_root
        / f"generic_candidates_{stem}.csv",
        pd.DataFrame(
            candidate_rows
        ),
    )

    atomic_csv(
        output_root
        / f"generic_summary_{stem}.csv",
        pd.DataFrame(
            summary_rows
        ),
    )

    metadata = {
        "condition":
            "generic",

        "dataset":
            "T401 branch-labelled",

        "start":
            args.start,

        "stop":
            args.stop,

        "predictions":
            str(predictions),

        "phase6_worker":
            str(PHASE6_WORKER),

        "budget":
            (
                "12 multistart_single_box "
                "calls per T401 point"
            ),

        "reference_used_for_selection":
            False,

        "target_tolerance":
            1.0e-4,

        "generic_centers":
            phase6.GENERIC_CENTERS,

        "cr_half_windows":
            defaults[
                "cr_half_windows"
            ],

        "ci_half_windows":
            defaults[
                "ci_half_windows"
            ],

        "retry_growth":
            defaults[
                "retry_growth"
            ],

        "max_retries":
            defaults[
                "max_retries"
            ],

        "max_iter":
            defaults[
                "max_iter"
            ],

        "grid_size":
            defaults[
                "grid_size"
            ],
    }

    (
        output_root
        / f"generic_metadata_{stem}.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
