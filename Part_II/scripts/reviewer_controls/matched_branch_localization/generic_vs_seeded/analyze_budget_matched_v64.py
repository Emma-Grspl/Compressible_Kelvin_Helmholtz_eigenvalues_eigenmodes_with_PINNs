#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_RANGES = [
    (0, 32),
    (32, 64),
]

CONDITIONS = [
    "seeded",
    "generic",
]

TARGET_TOL = 1.0e-4

GENERIC_CENTERS = {
    (0.10, 1.0e-4),
    (0.50, 1.0e-4),
    (0.90, 1.0e-4),
    (0.10, 2.0e-3),
    (0.50, 2.0e-3),
    (0.90, 2.0e-3),
}


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        })
        .fillna(False)
        .astype(bool)
    )


def exact_paired_binomial_p(
    a_only: int,
    b_only: int,
) -> float:
    """
    Exact two-sided McNemar/binomial test on discordant pairs.
    """
    n = int(a_only + b_only)

    if n == 0:
        return 1.0

    k = min(
        int(a_only),
        int(b_only),
    )

    tail = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2.0 ** n)

    return min(
        1.0,
        2.0 * tail,
    )


def q95(values) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )
    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.quantile(
            values,
            0.95,
        )
    )


def load_condition(
    root: Path,
    condition: str,
):
    summaries = []
    candidates = []

    for start, stop in EXPECTED_RANGES:

        summary_path = (
            root
            / (
                f"summary_{condition}_"
                f"{start:02d}_{stop:02d}.csv"
            )
        )

        candidate_path = (
            root
            / (
                f"candidates_{condition}_"
                f"{start:02d}_{stop:02d}.csv"
            )
        )

        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Missing summary file:\n{summary_path}"
            )

        if not candidate_path.is_file():
            raise FileNotFoundError(
                f"Missing candidate file:\n{candidate_path}"
            )

        s = pd.read_csv(
            summary_path
        )

        c = pd.read_csv(
            candidate_path
        )

        s["source_file"] = str(
            summary_path
        )

        c["source_file"] = str(
            candidate_path
        )

        summaries.append(s)
        candidates.append(c)

    summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    candidate = pd.concat(
        candidates,
        ignore_index=True,
    )

    return summary, candidate


def validate_condition(
    condition: str,
    summary: pd.DataFrame,
    candidate: pd.DataFrame,
):
    print()
    print("=" * 110)
    print(
        f"INTEGRITY — {condition.upper()}"
    )
    print("=" * 110)

    # --------------------------------------------------------------
    # Summary structure
    # --------------------------------------------------------------

    if len(summary) != 64:
        raise RuntimeError(
            f"{condition}: expected 64 summary rows, "
            f"found {len(summary)}"
        )

    if summary["benchmark_id"].nunique() != 64:
        raise RuntimeError(
            f"{condition}: benchmark_id is not unique"
        )

    # --------------------------------------------------------------
    # Candidate structure
    # --------------------------------------------------------------

    expected_candidates = 64 * 12

    if len(candidate) != expected_candidates:
        raise RuntimeError(
            f"{condition}: expected {expected_candidates} "
            f"candidate rows, found {len(candidate)}"
        )

    counts = (
        candidate
        .groupby("benchmark_id")
        .size()
    )

    if not (
        (counts == 12).all()
        and len(counts) == 64
    ):
        bad = counts[
            counts != 12
        ]

        raise RuntimeError(
            f"{condition}: not exactly 12 boxes per point:\n"
            f"{bad}"
        )

    if set(
        summary["benchmark_id"].astype(int)
    ) != set(
        candidate["benchmark_id"].astype(int)
    ):
        raise RuntimeError(
            f"{condition}: summary/candidate IDs differ"
        )

    # --------------------------------------------------------------
    # Window-pair integrity
    # --------------------------------------------------------------

    pair_counts = (
        candidate
        .groupby("benchmark_id")
        .apply(
            lambda g: len(
                set(
                    zip(
                        np.round(
                            pd.to_numeric(
                                g[
                                    "requested_cr_half_window"
                                ]
                            ),
                            12,
                        ),
                        np.round(
                            pd.to_numeric(
                                g[
                                    "requested_ci_half_window"
                                ]
                            ),
                            12,
                        ),
                    )
                )
            )
        )
    )

    if not (
        pair_counts == 12
    ).all():
        raise RuntimeError(
            f"{condition}: duplicated/missing window pairs"
        )

    # --------------------------------------------------------------
    # Condition-specific center validation
    # --------------------------------------------------------------

    if condition == "generic":

        for benchmark_id, g in candidate.groupby(
            "benchmark_id"
        ):
            centers = [
                (
                    round(
                        float(cr),
                        12,
                    ),
                    round(
                        float(ci),
                        12,
                    ),
                )
                for cr, ci in zip(
                    g["cr_center"],
                    g["ci_center"],
                )
            ]

            unique = set(
                centers
            )

            expected = {
                (
                    round(cr, 12),
                    round(ci, 12),
                )
                for cr, ci
                in GENERIC_CENTERS
            }

            if unique != expected:
                raise RuntimeError(
                    f"generic id={benchmark_id}: "
                    "incorrect center set"
                )

            center_counts = (
                pd.Series(
                    centers
                )
                .value_counts()
            )

            if not (
                center_counts == 2
            ).all():
                raise RuntimeError(
                    f"generic id={benchmark_id}: "
                    "each center is not used exactly twice"
                )

    elif condition == "seeded":

        s_by_id = (
            summary
            .set_index("benchmark_id")
        )

        for benchmark_id, g in candidate.groupby(
            "benchmark_id"
        ):

            if (
                g["cr_center"].nunique()
                != 1
                or
                g["ci_center"].nunique()
                != 1
            ):
                raise RuntimeError(
                    f"seeded id={benchmark_id}: "
                    "boxes do not share one PINN center"
                )

            row = s_by_id.loc[
                benchmark_id
            ]

            expected_cr = max(
                0.0,
                float(row["cr_pinn"]),
            )

            expected_ci = max(
                1.0e-4,
                float(row["ci_pinn"]),
            )

            actual_cr = float(
                g["cr_center"].iloc[0]
            )

            actual_ci = float(
                g["ci_center"].iloc[0]
            )

            if not np.isclose(
                actual_cr,
                expected_cr,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    f"seeded id={benchmark_id}: "
                    "incorrect cr center"
                )

            if not np.isclose(
                actual_ci,
                expected_ci,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    f"seeded id={benchmark_id}: "
                    "incorrect ci center"
                )

    print(
        f"summary rows     : {len(summary)}/64"
    )
    print(
        f"candidate boxes  : {len(candidate)}/768"
    )
    print(
        "boxes per point  : 12/12"
    )
    print(
        "window pairs     : 12 unique/point"
    )
    print(
        "center policy    : OK"
    )


def prepare(
    summary: pd.DataFrame,
    candidate: pd.DataFrame,
):
    summary = summary.copy()
    candidate = candidate.copy()

    # --------------------------------------------------------------
    # Summary booleans and metrics
    # --------------------------------------------------------------

    summary[
        "technical_success"
    ] = as_bool(
        summary[
            "technical_success"
        ]
    )

    mach = pd.to_numeric(
        summary["Mach"],
        errors="raise",
    ).to_numpy(dtype=float)

    alpha = pd.to_numeric(
        summary["alpha"],
        errors="raise",
    ).to_numpy(dtype=float)

    ambiguous = (
        np.isclose(
            mach,
            1.10,
            rtol=0.0,
            atol=1.0e-12,
        )
        &
        np.isclose(
            alpha,
            0.09,
            rtol=0.0,
            atol=1.0e-12,
        )
    )

    if ambiguous.sum() != 1:
        raise RuntimeError(
            "Expected exactly one predeclared "
            f"ambiguous V64 point; found {ambiguous.sum()}"
        )

    summary[
        "ambiguous_recomputed"
    ] = ambiguous

    summary[
        "target_error"
    ] = pd.to_numeric(
        summary[
            "target_error"
        ],
        errors="coerce",
    )

    summary[
        "target_recovery_recomputed"
    ] = (
        (~summary[
            "ambiguous_recomputed"
        ])
        &
        np.isfinite(
            summary[
                "target_error"
            ]
        )
        &
        (
            summary[
                "target_error"
            ]
            <= TARGET_TOL
        )
    )

    # --------------------------------------------------------------
    # Candidate booleans / cost
    # --------------------------------------------------------------

    candidate[
        "technical_success"
    ] = as_bool(
        candidate[
            "technical_success"
        ]
    )

    candidate[
        "status_completed"
    ] = (
        candidate[
            "status"
        ]
        .astype(str)
        .str.upper()
        .eq("COMPLETED")
    )

    candidate[
        "retry_index"
    ] = pd.to_numeric(
        candidate[
            "retry_index"
        ],
        errors="coerce",
    )

    candidate[
        "seconds"
    ] = pd.to_numeric(
        candidate[
            "seconds"
        ],
        errors="coerce",
    )

    # retry=0 => one solve
    # retry=1 => two solves
    # ...
    candidate[
        "actual_box_solves"
    ] = np.where(
        (
            candidate[
                "status_completed"
            ]
            &
            np.isfinite(
                candidate[
                    "retry_index"
                ]
            )
        ),
        candidate[
            "retry_index"
        ]
        + 1.0,
        np.nan,
    )

    return summary, candidate


def point_costs(
    candidate: pd.DataFrame,
):
    rows = []

    for benchmark_id, g in candidate.groupby(
        "benchmark_id"
    ):

        failed = int(
            (
                ~g[
                    "status_completed"
                ]
            ).sum()
        )

        solves = pd.to_numeric(
            g[
                "actual_box_solves"
            ],
            errors="coerce",
        )

        seconds = pd.to_numeric(
            g["seconds"],
            errors="coerce",
        )

        rows.append({
            "benchmark_id":
                int(benchmark_id),

            "n_boxes":
                len(g),

            "n_completed_boxes":
                int(
                    g[
                        "status_completed"
                    ].sum()
                ),

            "n_failed_boxes":
                failed,

            "n_technical_boxes":
                int(
                    g[
                        "technical_success"
                    ].sum()
                ),

            "total_box_solves":
                float(
                    np.nansum(
                        solves.to_numpy(
                            dtype=float
                        )
                    )
                ),

            "total_seconds":
                float(
                    np.nansum(
                        seconds.to_numpy(
                            dtype=float
                        )
                    )
                ),
        })

    return pd.DataFrame(
        rows
    )


def aggregate_condition(
    condition: str,
    summary: pd.DataFrame,
    candidate: pd.DataFrame,
    costs: pd.DataFrame,
):
    nonamb = summary[
        ~summary[
            "ambiguous_recomputed"
        ]
    ].copy()

    if len(nonamb) != 63:
        raise RuntimeError(
            f"{condition}: expected 63 non-ambiguous points"
        )

    target_error = (
        pd.to_numeric(
            nonamb[
                "target_error"
            ],
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    finite_error = target_error[
        np.isfinite(
            target_error
        )
    ]

    failed_boxes = int(
        (
            ~candidate[
                "status_completed"
            ]
        ).sum()
    )

    return {
        "condition":
            condition,

        "technical_success":
            int(
                summary[
                    "technical_success"
                ].sum()
            ),

        "technical_total":
            64,

        "target_recovery":
            int(
                nonamb[
                    "target_recovery_recomputed"
                ].sum()
            ),

        "target_total":
            63,

        "target_error_median":
            float(
                np.median(
                    finite_error
                )
            )
            if len(finite_error)
            else np.nan,

        "target_error_p95":
            q95(
                finite_error
            ),

        "target_error_max":
            float(
                np.max(
                    finite_error
                )
            )
            if len(finite_error)
            else np.nan,

        "target_error_le_1e5":
            int(
                (
                    finite_error
                    <= 1.0e-5
                ).sum()
            ),

        "target_error_le_1e4":
            int(
                (
                    finite_error
                    <= 1.0e-4
                ).sum()
            ),

        "n_boxes":
            len(candidate),

        "n_technical_boxes":
            int(
                candidate[
                    "technical_success"
                ].sum()
            ),

        "n_failed_boxes":
            failed_boxes,

        "total_box_solves":
            float(
                costs[
                    "total_box_solves"
                ].sum()
            ),

        "box_solves_per_point_median":
            float(
                costs[
                    "total_box_solves"
                ].median()
            ),

        "box_solves_per_point_p95":
            q95(
                costs[
                    "total_box_solves"
                ]
            ),

        "total_seconds":
            float(
                costs[
                    "total_seconds"
                ].sum()
            ),

        "seconds_per_point_median":
            float(
                costs[
                    "total_seconds"
                ].median()
            ),

        "seconds_per_point_p95":
            q95(
                costs[
                    "total_seconds"
                ]
            ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "assets/pinn_supersonic/"
            "reviewer_runs/"
            "budget_matched_v64"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    root = args.input_root.resolve()

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else root / "analysis"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summary = {}
    all_candidates = {}
    all_costs = {}

    # ==============================================================
    # LOAD + VALIDATE
    # ==============================================================

    for condition in CONDITIONS:

        summary, candidate = (
            load_condition(
                root,
                condition,
            )
        )

        validate_condition(
            condition,
            summary,
            candidate,
        )

        summary, candidate = prepare(
            summary,
            candidate,
        )

        costs = point_costs(
            candidate
        )

        all_summary[
            condition
        ] = summary

        all_candidates[
            condition
        ] = candidate

        all_costs[
            condition
        ] = costs

    # ==============================================================
    # PAIRED COORDINATE INTEGRITY
    # ==============================================================

    seeded = (
        all_summary[
            "seeded"
        ]
        .sort_values(
            "benchmark_id"
        )
        .reset_index(drop=True)
    )

    generic = (
        all_summary[
            "generic"
        ]
        .sort_values(
            "benchmark_id"
        )
        .reset_index(drop=True)
    )

    if not np.array_equal(
        seeded[
            "benchmark_id"
        ].to_numpy(dtype=int),
        generic[
            "benchmark_id"
        ].to_numpy(dtype=int),
    ):
        raise RuntimeError(
            "Seeded/generic benchmark IDs do not match"
        )

    for col in [
        "Mach",
        "alpha",
        "cr_reference",
        "ci_reference",
    ]:
        if not np.allclose(
            pd.to_numeric(
                seeded[col]
            ).to_numpy(dtype=float),
            pd.to_numeric(
                generic[col]
            ).to_numpy(dtype=float),
            rtol=0.0,
            atol=1.0e-12,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"Seeded/generic mismatch in {col}"
            )

    print()
    print("=" * 110)
    print("PAIRED COORDINATE INTEGRITY")
    print("=" * 110)
    print("64/64 coordinates identical: OK")
    print("references identical          : OK")
    print("one ambiguous point           : OK")

    # ==============================================================
    # AGGREGATE TABLE
    # ==============================================================

    aggregate_rows = []

    for condition in CONDITIONS:

        aggregate_rows.append(
            aggregate_condition(
                condition,
                all_summary[
                    condition
                ],
                all_candidates[
                    condition
                ],
                all_costs[
                    condition
                ],
            )
        )

    aggregate = pd.DataFrame(
        aggregate_rows
    )

    # ==============================================================
    # PAIRED TABLE
    # ==============================================================

    seeded_cost = (
        all_costs[
            "seeded"
        ]
        .set_index(
            "benchmark_id"
        )
    )

    generic_cost = (
        all_costs[
            "generic"
        ]
        .set_index(
            "benchmark_id"
        )
    )

    paired_rows = []

    for i in range(64):

        s = seeded.iloc[i]
        g = generic.iloc[i]

        benchmark_id = int(
            s[
                "benchmark_id"
            ]
        )

        sc = seeded_cost.loc[
            benchmark_id
        ]

        gc = generic_cost.loc[
            benchmark_id
        ]

        paired_rows.append({
            "benchmark_id":
                benchmark_id,

            "atlas_chart":
                s.get(
                    "atlas_chart",
                    "",
                ),

            "Mach":
                float(s["Mach"]),

            "alpha":
                float(s["alpha"]),

            "ambiguous":
                bool(
                    s[
                        "ambiguous_recomputed"
                    ]
                ),

            "seeded_technical":
                bool(
                    s[
                        "technical_success"
                    ]
                ),

            "generic_technical":
                bool(
                    g[
                        "technical_success"
                    ]
                ),

            "seeded_target_recovery":
                bool(
                    s[
                        "target_recovery_recomputed"
                    ]
                ),

            "generic_target_recovery":
                bool(
                    g[
                        "target_recovery_recomputed"
                    ]
                ),

            "seeded_target_error":
                float(
                    s[
                        "target_error"
                    ]
                ),

            "generic_target_error":
                float(
                    g[
                        "target_error"
                    ]
                ),

            "delta_generic_minus_seeded_error":
                float(
                    g[
                        "target_error"
                    ]
                    - s[
                        "target_error"
                    ]
                ),

            "seeded_box_solves":
                float(
                    sc[
                        "total_box_solves"
                    ]
                ),

            "generic_box_solves":
                float(
                    gc[
                        "total_box_solves"
                    ]
                ),

            "delta_generic_minus_seeded_box_solves":
                float(
                    gc[
                        "total_box_solves"
                    ]
                    - sc[
                        "total_box_solves"
                    ]
                ),

            "seeded_seconds":
                float(
                    sc[
                        "total_seconds"
                    ]
                ),

            "generic_seconds":
                float(
                    gc[
                        "total_seconds"
                    ]
                ),

            "delta_generic_minus_seeded_seconds":
                float(
                    gc[
                        "total_seconds"
                    ]
                    - sc[
                        "total_seconds"
                    ]
                ),
        })

    paired = pd.DataFrame(
        paired_rows
    )

    nonamb = paired[
        ~paired[
            "ambiguous"
        ]
    ].copy()

    if len(nonamb) != 63:
        raise RuntimeError(
            "Expected 63 paired non-ambiguous points"
        )

    # ==============================================================
    # PAIRED DISCORDANCES
    # ==============================================================

    seeded_only_target = int(
        (
            nonamb[
                "seeded_target_recovery"
            ]
            &
            ~nonamb[
                "generic_target_recovery"
            ]
        ).sum()
    )

    generic_only_target = int(
        (
            ~nonamb[
                "seeded_target_recovery"
            ]
            &
            nonamb[
                "generic_target_recovery"
            ]
        ).sum()
    )

    both_target = int(
        (
            nonamb[
                "seeded_target_recovery"
            ]
            &
            nonamb[
                "generic_target_recovery"
            ]
        ).sum()
    )

    neither_target = int(
        (
            ~nonamb[
                "seeded_target_recovery"
            ]
            &
            ~nonamb[
                "generic_target_recovery"
            ]
        ).sum()
    )

    target_exact_p = (
        exact_paired_binomial_p(
            seeded_only_target,
            generic_only_target,
        )
    )

    seeded_only_technical = int(
        (
            paired[
                "seeded_technical"
            ]
            &
            ~paired[
                "generic_technical"
            ]
        ).sum()
    )

    generic_only_technical = int(
        (
            ~paired[
                "seeded_technical"
            ]
            &
            paired[
                "generic_technical"
            ]
        ).sum()
    )

    technical_exact_p = (
        exact_paired_binomial_p(
            seeded_only_technical,
            generic_only_technical,
        )
    )

    # ==============================================================
    # CONTINUOUS PAIRED DESCRIPTIVES
    # ==============================================================

    error_delta = pd.to_numeric(
        nonamb[
            "delta_generic_minus_seeded_error"
        ],
        errors="coerce",
    )

    finite_delta = error_delta[
        np.isfinite(
            error_delta
        )
    ]

    seeded_lower_error = int(
        (
            finite_delta
            > 0.0
        ).sum()
    )

    generic_lower_error = int(
        (
            finite_delta
            < 0.0
        ).sum()
    )

    cost_delta = pd.to_numeric(
        paired[
            "delta_generic_minus_seeded_box_solves"
        ],
        errors="coerce",
    )

    seconds_delta = pd.to_numeric(
        paired[
            "delta_generic_minus_seeded_seconds"
        ],
        errors="coerce",
    )

    paired_summary = pd.DataFrame([
        {
            "metric":
                "target_recovery",

            "both_success":
                both_target,

            "seeded_only_success":
                seeded_only_target,

            "generic_only_success":
                generic_only_target,

            "neither_success":
                neither_target,

            "exact_paired_p":
                target_exact_p,
        },
        {
            "metric":
                "technical_success",

            "both_success":
                int(
                    (
                        paired[
                            "seeded_technical"
                        ]
                        &
                        paired[
                            "generic_technical"
                        ]
                    ).sum()
                ),

            "seeded_only_success":
                seeded_only_technical,

            "generic_only_success":
                generic_only_technical,

            "neither_success":
                int(
                    (
                        ~paired[
                            "seeded_technical"
                        ]
                        &
                        ~paired[
                            "generic_technical"
                        ]
                    ).sum()
                ),

            "exact_paired_p":
                technical_exact_p,
        },
    ])

    continuous_summary = pd.DataFrame([
        {
            "metric":
                "target_error_generic_minus_seeded",

            "n":
                len(finite_delta),

            "median_delta":
                float(
                    finite_delta.median()
                ),

            "mean_delta":
                float(
                    finite_delta.mean()
                ),

            "p95_delta":
                q95(
                    finite_delta
                ),

            "positive_count_seeded_better":
                seeded_lower_error,

            "negative_count_generic_better":
                generic_lower_error,
        },
        {
            "metric":
                "box_solves_generic_minus_seeded",

            "n":
                int(
                    np.isfinite(
                        cost_delta
                    ).sum()
                ),

            "median_delta":
                float(
                    cost_delta.median()
                ),

            "mean_delta":
                float(
                    cost_delta.mean()
                ),

            "p95_delta":
                q95(
                    cost_delta
                ),

            "positive_count_seeded_better":
                int(
                    (
                        cost_delta
                        > 0.0
                    ).sum()
                ),

            "negative_count_generic_better":
                int(
                    (
                        cost_delta
                        < 0.0
                    ).sum()
                ),
        },
        {
            "metric":
                "seconds_generic_minus_seeded",

            "n":
                int(
                    np.isfinite(
                        seconds_delta
                    ).sum()
                ),

            "median_delta":
                float(
                    seconds_delta.median()
                ),

            "mean_delta":
                float(
                    seconds_delta.mean()
                ),

            "p95_delta":
                q95(
                    seconds_delta
                ),

            "positive_count_seeded_better":
                int(
                    (
                        seconds_delta
                        > 0.0
                    ).sum()
                ),

            "negative_count_generic_better":
                int(
                    (
                        seconds_delta
                        < 0.0
                    ).sum()
                ),
        },
    ])

    # ==============================================================
    # PRINT
    # ==============================================================

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        220,
    )

    print()
    print("=" * 110)
    print("PHASE 6 — CONDITION SUMMARY")
    print("=" * 110)

    print(
        aggregate.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8e}",
        )
    )

    print()
    print("=" * 110)
    print("PHASE 6 — PAIRED DISCORDANCES")
    print("=" * 110)

    print(
        paired_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8e}",
        )
    )

    print()
    print("=" * 110)
    print("PHASE 6 — PAIRED CONTINUOUS DIFFERENCES")
    print("=" * 110)

    print(
        continuous_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8e}",
        )
    )

    print()
    print("=" * 110)
    print("NON-AMBIGUOUS TARGET FAILURES")
    print("=" * 110)

    failures = nonamb[
        (
            ~nonamb[
                "seeded_target_recovery"
            ]
        )
        |
        (
            ~nonamb[
                "generic_target_recovery"
            ]
        )
    ].copy()

    if failures.empty:
        print("None")
    else:
        print(
            failures[
                [
                    "benchmark_id",
                    "atlas_chart",
                    "Mach",
                    "alpha",
                    "seeded_target_recovery",
                    "generic_target_recovery",
                    "seeded_target_error",
                    "generic_target_error",
                    "seeded_technical",
                    "generic_technical",
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.8e}",
            )
        )

    # ==============================================================
    # AUTOMATED CAUTIOUS VERDICT
    # ==============================================================

    seeded_recovery = int(
        nonamb[
            "seeded_target_recovery"
        ].sum()
    )

    generic_recovery = int(
        nonamb[
            "generic_target_recovery"
        ].sum()
    )

    if seeded_recovery > generic_recovery:
        verdict = (
            "The matched-budget comparison descriptively "
            "supports a branch-localization advantage from "
            "PINN seeding: the seeded condition recovers more "
            "target branches than the generic condition under "
            "the same 12-box classical search budget."
        )

    elif seeded_recovery == generic_recovery:
        verdict = (
            "The matched-budget comparison does not show a "
            "target-branch recovery advantage from PINN seeding: "
            "both conditions recover the same number of target "
            "branches under the same 12-box classical search budget."
        )

    else:
        verdict = (
            "The matched-budget comparison does not support a "
            "branch-localization advantage from PINN seeding: "
            "the generic condition recovers more target branches "
            "under the same 12-box classical search budget."
        )

    # ==============================================================
    # SAVE
    # ==============================================================

    aggregate.to_csv(
        output_root
        / "condition_summary.csv",
        index=False,
    )

    paired.to_csv(
        output_root
        / "paired_point_results.csv",
        index=False,
    )

    paired_summary.to_csv(
        output_root
        / "paired_discordances.csv",
        index=False,
    )

    continuous_summary.to_csv(
        output_root
        / "paired_continuous_summary.csv",
        index=False,
    )

    failures.to_csv(
        output_root
        / "target_failures.csv",
        index=False,
    )

    for condition in CONDITIONS:

        all_summary[
            condition
        ].to_csv(
            output_root
            / f"{condition}_summary_combined.csv",
            index=False,
        )

        all_candidates[
            condition
        ].to_csv(
            output_root
            / f"{condition}_candidates_combined.csv",
            index=False,
        )

        all_costs[
            condition
        ].to_csv(
            output_root
            / f"{condition}_point_costs.csv",
            index=False,
        )

    text = f"""
PHASE 6 — BUDGET-MATCHED SEEDED VS GENERIC V64
================================================

Protocol
--------
64 identical V64 coordinates.
12 multistart_single_box calls per point and condition.
Same window set, solver, retry policy, grid size and optimizer settings.
Only the search-center policy differs.
Target recovery assessed on 63 predeclared non-ambiguous points.
Target tolerance: |c_shoot - c_ref| <= {TARGET_TOL:.1e}.

Target branch recovery
----------------------
Seeded : {seeded_recovery}/63
Generic: {generic_recovery}/63

Paired target discordances
--------------------------
Both recover       : {both_target}
Seeded only        : {seeded_only_target}
Generic only       : {generic_only_target}
Neither            : {neither_target}
Exact paired p     : {target_exact_p:.8e}

Technical success
-----------------
Seeded : {int(aggregate.loc[aggregate.condition == 'seeded', 'technical_success'].iloc[0])}/64
Generic: {int(aggregate.loc[aggregate.condition == 'generic', 'technical_success'].iloc[0])}/64

Cost
----
Seeded total box solves :
{float(aggregate.loc[aggregate.condition == 'seeded', 'total_box_solves'].iloc[0]):.0f}

Generic total box solves:
{float(aggregate.loc[aggregate.condition == 'generic', 'total_box_solves'].iloc[0]):.0f}

Seeded total measured seconds :
{float(aggregate.loc[aggregate.condition == 'seeded', 'total_seconds'].iloc[0]):.3f}

Generic total measured seconds:
{float(aggregate.loc[aggregate.condition == 'generic', 'total_seconds'].iloc[0]):.3f}

Interpretation
--------------
{verdict}

Important limitation
--------------------
If any candidate box failed by exception, the reconstructed
total_box_solves is a lower bound for that box because the worker
does not retain the retry index reached before the exception.
See n_failed_boxes in condition_summary.csv.
""".strip()

    (
        output_root
        / "phase6_verdict.txt"
    ).write_text(
        text + "\n"
    )

    print()
    print("=" * 110)
    print("VERDICT")
    print("=" * 110)
    print(verdict)

    print()
    print("=" * 110)
    print("WRITTEN")
    print("=" * 110)

    for p in sorted(
        output_root.glob("*")
    ):
        print(p)


if __name__ == "__main__":
    main()
