#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path.cwd().resolve()

BASE_CORRECTORS = (
    REPO
    / "scripts/pinn_supersonic/"
      "benchmark_atlas2d_correctors.py"
)

GENERIC_CENTERS = [
    (0.10, 1.0e-4),
    (0.50, 1.0e-4),
    (0.90, 1.0e-4),
    (0.10, 2.0e-3),
    (0.50, 2.0e-3),
    (0.90, 2.0e-3),
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_csv(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def load_shooting():
    if not BASE_CORRECTORS.is_file():
        raise FileNotFoundError(BASE_CORRECTORS)

    base = load_module(
        "budget_matched_correctors_base",
        BASE_CORRECTORS,
    )

    # Preserve the same repository-specific import side effect
    # as the production reviewer shooting.
    base.load_module(
        "budget_matched_gep_path_setup_only",
        base.COMPARE_GEP_SCRIPT,
    )

    shooting = base.load_module(
        "budget_matched_shooting",
        base.SHOOTING_SCRIPT,
    )

    defaults = base.parser_defaults(
        shooting.build_parser()
    )

    return shooting, defaults


def window_pairs(defaults):
    cr_windows = [
        float(x)
        for x in defaults["cr_half_windows"]
    ]
    ci_windows = [
        float(x)
        for x in defaults["ci_half_windows"]
    ]

    pairs = [
        (cr, ci)
        for cr in cr_windows
        for ci in ci_windows
    ]

    if len(pairs) != 12:
        raise RuntimeError(
            "Expected exactly 12 production window pairs; "
            f"found {len(pairs)}: {pairs}"
        )

    return pairs


def candidate_specs(
    *,
    condition: str,
    source: pd.Series,
    benchmark_id: int,
    defaults,
):
    pairs = window_pairs(defaults)

    specs = []

    if condition == "seeded":
        cr_center = max(
            0.0,
            float(source["cr_pred"]),
        )
        ci_center = max(
            1.0e-4,
            float(source["ci_pred"]),
        )

        for idx, (cr_half, ci_half) in enumerate(pairs):
            specs.append({
                "candidate_index": idx,
                "center_kind": "PINN",
                "center_index": 0,
                "cr_center": cr_center,
                "ci_center": ci_center,
                "cr_half": cr_half,
                "ci_half": ci_half,
            })

    elif condition == "generic":

        # Same 12 box shapes as the PINN condition.
        #
        # Six historical generic centers are used exactly twice
        # per point. Rotation by benchmark_id prevents any one
        # center from being systematically associated with a
        # particular box size across V64.
        for idx, (cr_half, ci_half) in enumerate(pairs):

            center_index = (
                idx + benchmark_id
            ) % len(GENERIC_CENTERS)

            cr_center, ci_center = (
                GENERIC_CENTERS[center_index]
            )

            specs.append({
                "candidate_index": idx,
                "center_kind": "generic",
                "center_index": center_index,
                "cr_center": float(cr_center),
                "ci_center": float(ci_center),
                "cr_half": cr_half,
                "ci_half": ci_half,
            })

    else:
        raise ValueError(condition)

    if len(specs) != 12:
        raise RuntimeError(
            f"{condition}: expected 12 candidate specs"
        )

    return specs


def solve_candidate(
    *,
    shooting,
    defaults,
    source,
    benchmark_id,
    condition,
    spec,
):
    mach = float(source["Mach"])
    alpha = float(source["alpha"])

    t0 = time.perf_counter()

    row = {
        "condition": condition,
        "benchmark_id": benchmark_id,
        "atlas_chart": str(
            source.get("atlas_chart", "")
        ),
        "Mach": mach,
        "alpha": alpha,

        "candidate_index":
            int(spec["candidate_index"]),

        "center_kind":
            str(spec["center_kind"]),

        "center_index":
            int(spec["center_index"]),

        "cr_center":
            float(spec["cr_center"]),

        "ci_center":
            float(spec["ci_center"]),

        "requested_cr_half_window":
            float(spec["cr_half"]),

        "requested_ci_half_window":
            float(spec["ci_half"]),

        "status": "FAILED",
        "error": "",

        "retry_index": np.nan,

        "used_cr_half_window": np.nan,
        "used_ci_half_window": np.nan,

        "shoot_cr": np.nan,
        "shoot_ci": np.nan,

        "stage1_mismatch": np.nan,
        "stage2_mismatch": np.nan,
        "total_mismatch": np.nan,

        "spectral_success": False,
        "mode_success": False,
        "technical_success": False,

        "delta_center": np.nan,
        "seconds": np.nan,
    }

    try:
        (
            solver,
            result,
            retry_idx,
            used_cr_half,
            used_ci_half,
        ) = shooting.multistart_single_box(
            alpha=alpha,
            mach=mach,

            match_y=float(
                defaults["match_y"]
            ),

            # Match production reviewer convention exactly.
            use_mapping=True,
            mapping_scale=5.0,

            min_y_limit=float(
                defaults["min_y_limit"]
            ),
            max_y_limit=float(
                defaults["max_y_limit"]
            ),
            y_limit_factor=float(
                defaults["y_limit_factor"]
            ),

            amp_lower_bound=float(
                defaults["amp_lower_bound"]
            ),
            amp_upper_bound=float(
                defaults["amp_upper_bound"]
            ),

            cr_center=float(
                spec["cr_center"]
            ),
            ci_center=float(
                spec["ci_center"]
            ),

            cr_half_window=float(
                spec["cr_half"]
            ),
            ci_half_window=float(
                spec["ci_half"]
            ),

            retry_growth=float(
                defaults["retry_growth"]
            ),
            max_retries=int(
                defaults["max_retries"]
            ),

            max_iter=int(
                defaults["max_iter"]
            ),
            grid_size=int(
                defaults["grid_size"]
            ),
        )

        cr = float(result.cr)
        ci = float(result.ci)

        stage1 = float(
            result.stage1_mismatch
        )
        stage2 = float(
            result.stage2_mismatch
        )

        spectral_success = bool(
            result.spectral_success
        )
        mode_success = bool(
            result.mode_success
        )

        row.update({
            "status": "COMPLETED",

            "retry_index":
                int(retry_idx),

            "used_cr_half_window":
                float(used_cr_half),

            "used_ci_half_window":
                float(used_ci_half),

            "shoot_cr": cr,
            "shoot_ci": ci,

            "stage1_mismatch": stage1,
            "stage2_mismatch": stage2,
            "total_mismatch":
                stage1 + stage2,

            "spectral_success":
                spectral_success,

            "mode_success":
                mode_success,

            "technical_success":
                bool(
                    spectral_success
                    and mode_success
                ),

            "delta_center":
                float(
                    np.hypot(
                        cr
                        - float(
                            spec["cr_center"]
                        ),
                        ci
                        - float(
                            spec["ci_center"]
                        ),
                    )
                ),
        })

    except Exception as exc:
        row["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    row["seconds"] = (
        time.perf_counter() - t0
    )

    return row


def choose_best(candidates):
    valid = [
        row
        for row in candidates
        if (
            row["status"] == "COMPLETED"
            and np.isfinite(
                float(
                    row["total_mismatch"]
                )
            )
        )
    ]

    if not valid:
        return None

    # Identical reference-independent selection
    # in both conditions.
    return min(
        valid,
        key=lambda row: (
            not bool(
                row["technical_success"]
            ),
            float(
                row["total_mismatch"]
            ),
            float(
                row["delta_center"]
            ),
            int(
                row["candidate_index"]
            ),
        ),
    )


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
        "--condition",
        choices=[
            "seeded",
            "generic",
        ],
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

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.read_csv(
        predictions
    ).reset_index(drop=True)

    required = {
        "Mach",
        "alpha",
        "cr",
        "ci",
        "cr_pred",
        "ci_pred",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    n = len(frame)

    if n != 64:
        raise RuntimeError(
            f"Expected V64 with 64 rows; found {n}"
        )

    if not (
        0 <= args.start < args.stop <= n
    ):
        raise ValueError(
            f"Invalid range [{args.start}, {args.stop})"
        )

    shooting, defaults = (
        load_shooting()
    )

    part = frame.iloc[
        args.start:args.stop
    ].copy()

    if args.dry_run:
        print(
            "condition =",
            args.condition,
        )
        print(
            "range     =",
            args.start,
            args.stop,
        )

        for global_idx, source in part.head(2).iterrows():

            benchmark_id = (
                int(source["benchmark_id"])
                if "benchmark_id" in frame.columns
                else int(global_idx)
            )

            print()
            print(
                "benchmark_id =",
                benchmark_id,
            )
            print(
                "M, alpha =",
                float(source["Mach"]),
                float(source["alpha"]),
            )

            specs = candidate_specs(
                condition=args.condition,
                source=source,
                benchmark_id=benchmark_id,
                defaults=defaults,
            )

            for spec in specs:
                print(spec)

        return

    candidate_rows = []
    summary_rows = []

    for global_idx, source in part.iterrows():

        benchmark_id = (
            int(source["benchmark_id"])
            if "benchmark_id" in frame.columns
            else int(global_idx)
        )

        mach = float(source["Mach"])
        alpha = float(source["alpha"])

        cr_ref = float(source["cr"])
        ci_ref = float(source["ci"])

        cr_pinn = float(
            source["cr_pred"]
        )
        ci_pinn = float(
            source["ci_pred"]
        )

        print()
        print("=" * 100)
        print(
            f"{args.condition.upper()} "
            f"id={benchmark_id:02d} "
            f"M={mach:.3f} "
            f"alpha={alpha:.3f}",
            flush=True,
        )

        specs = candidate_specs(
            condition=args.condition,
            source=source,
            benchmark_id=benchmark_id,
            defaults=defaults,
        )

        local = []

        for spec in specs:

            result = solve_candidate(
                shooting=shooting,
                defaults=defaults,
                source=source,
                benchmark_id=benchmark_id,
                condition=args.condition,
                spec=spec,
            )

            local.append(result)
            candidate_rows.append(result)

            print(
                f"  box={result['candidate_index']:02d} "
                f"center=({result['cr_center']:.4f},"
                f"{result['ci_center']:.4g}) "
                f"window=({result['requested_cr_half_window']:.3f},"
                f"{result['requested_ci_half_window']:.3f}) "
                f"retry={result['retry_index']} "
                f"tech={result['technical_success']} "
                f"mismatch={result['total_mismatch']}",
                flush=True,
            )

        best = choose_best(local)

        ambiguous = bool(
            np.isclose(
                mach,
                1.10,
                rtol=0.0,
                atol=1.0e-12,
            )
            and
            np.isclose(
                alpha,
                0.09,
                rtol=0.0,
                atol=1.0e-12,
            )
        )

        if best is None:
            summary = {
                "condition":
                    args.condition,
                "benchmark_id":
                    benchmark_id,
                "atlas_chart":
                    str(
                        source.get(
                            "atlas_chart",
                            "",
                        )
                    ),
                "Mach": mach,
                "alpha": alpha,

                "cr_reference":
                    cr_ref,
                "ci_reference":
                    ci_ref,

                "cr_pinn":
                    cr_pinn,
                "ci_pinn":
                    ci_pinn,

                "ambiguous":
                    ambiguous,

                "status":
                    "FAILED",

                "technical_success":
                    False,

                "target_error":
                    np.nan,

                "target_recovery":
                    False,

                "n_boxes": 12,

                "n_completed_boxes":
                    0,

                "n_technical_boxes":
                    0,

                "total_retries_used":
                    np.nan,

                "total_seconds":
                    float(
                        sum(
                            r["seconds"]
                            for r in local
                        )
                    ),
            }

        else:
            target_error = float(
                np.hypot(
                    float(
                        best["shoot_cr"]
                    )
                    - cr_ref,
                    float(
                        best["shoot_ci"]
                    )
                    - ci_ref,
                )
            )

            retries = [
                int(r["retry_index"])
                for r in local
                if (
                    r["status"]
                    == "COMPLETED"
                    and np.isfinite(
                        r["retry_index"]
                    )
                )
            ]

            summary = {
                "condition":
                    args.condition,
                "benchmark_id":
                    benchmark_id,
                "atlas_chart":
                    str(
                        source.get(
                            "atlas_chart",
                            "",
                        )
                    ),

                "Mach": mach,
                "alpha": alpha,

                "cr_reference":
                    cr_ref,
                "ci_reference":
                    ci_ref,

                "cr_pinn":
                    cr_pinn,
                "ci_pinn":
                    ci_pinn,

                "ambiguous":
                    ambiguous,

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
                        (not ambiguous)
                        and
                        target_error
                        <= 1.0e-4
                    ),

                "best_candidate_index":
                    int(
                        best[
                            "candidate_index"
                        ]
                    ),

                "best_center_kind":
                    best[
                        "center_kind"
                    ],

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
                        best["shoot_cr"]
                    ),

                "best_ci":
                    float(
                        best["shoot_ci"]
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

                "best_cr_half_window":
                    float(
                        best[
                            "used_cr_half_window"
                        ]
                    ),

                "best_ci_half_window":
                    float(
                        best[
                            "used_ci_half_window"
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

                # retry_index=0 means one actual solve;
                # retry_index=3 means four solves.
                "total_box_solves":
                    int(
                        sum(
                            r + 1
                            for r in retries
                        )
                    ),

                "total_seconds":
                    float(
                        sum(
                            r["seconds"]
                            for r in local
                        )
                    ),
            }

        summary_rows.append(
            summary
        )

        print(
            "BEST:",
            summary,
            flush=True,
        )

    stem = (
        f"{args.condition}_"
        f"{args.start:02d}_"
        f"{args.stop:02d}"
    )

    atomic_csv(
        output_root
        / f"candidates_{stem}.csv",
        pd.DataFrame(
            candidate_rows
        ),
    )

    atomic_csv(
        output_root
        / f"summary_{stem}.csv",
        pd.DataFrame(
            summary_rows
        ),
    )

    metadata = {
        "condition":
            args.condition,

        "start":
            args.start,

        "stop":
            args.stop,

        "predictions":
            str(predictions),

        "budget":
            "12 multistart_single_box calls per V64 point",

        "selection":
            (
                "technical success; "
                "minimum stage1+stage2 mismatch; "
                "minimum displacement from own center; "
                "candidate index tie-break"
            ),

        "reference_used_for_selection":
            False,

        "generic_centers":
            GENERIC_CENTERS,

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
        / f"metadata_{stem}.json"
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
