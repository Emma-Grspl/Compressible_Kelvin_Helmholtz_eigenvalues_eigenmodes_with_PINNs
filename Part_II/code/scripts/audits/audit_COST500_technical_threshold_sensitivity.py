from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "assets/classic_supersonic/csv/"
    "computational_cost/analysis/"
    "table_N76_COST500_shooting_timing_500.csv"
)

OUTPUT = Path(
    "experiments/"
    "technical_threshold_sensitivity/"
    "COST500_seeded"
)

STAGE1_NOMINAL = 5.0e-2
STAGE2_NOMINAL = 1.0e-2

MULTIPLIERS = [
    0.25,
    0.50,
    1.00,
    2.00,
]


def quantiles(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "median": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    return {
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT)

    if len(df) != 500:
        raise RuntimeError(
            f"Expected COST500 with 500 rows; got {len(df)}."
        )

    required = {
        "benchmark_id",
        "Mach",
        "alpha",
        "atlas_chart",
        "shoot_stage1_mismatch",
        "shoot_stage2_mismatch",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    stage1 = pd.to_numeric(
        df["shoot_stage1_mismatch"],
        errors="coerce",
    ).to_numpy(float)

    stage2 = pd.to_numeric(
        df["shoot_stage2_mismatch"],
        errors="coerce",
    ).to_numpy(float)

    # ------------------------------------------------------------
    # Sanity check against the stored nominal flags.
    # ------------------------------------------------------------

    nominal_stage1 = (
        np.isfinite(stage1)
        & (stage1 < STAGE1_NOMINAL)
    )

    nominal_stage2 = (
        np.isfinite(stage2)
        & (stage2 < STAGE2_NOMINAL)
    )

    nominal_both = (
        nominal_stage1
        & nominal_stage2
    )

    print(
        "Nominal recomputed:"
        f" stage1={nominal_stage1.sum()}/500,"
        f" stage2={nominal_stage2.sum()}/500,"
        f" both={nominal_both.sum()}/500"
    )

    # Compare against stored flags when available.
    if (
        "shoot_spectral_success" in df.columns
        and "shoot_mode_success" in df.columns
    ):
        stored1 = (
            df["shoot_spectral_success"]
            .astype(bool)
            .to_numpy()
        )

        stored2 = (
            df["shoot_mode_success"]
            .astype(bool)
            .to_numpy()
        )

        print(
            "Stored/recomputed disagreements:"
            f" stage1={np.sum(stored1 != nominal_stage1)},"
            f" stage2={np.sum(stored2 != nominal_stage2)}"
        )

    # ------------------------------------------------------------
    # 1. Main sensitivity: scale BOTH thresholds together.
    # ------------------------------------------------------------

    rows = []

    for multiplier in MULTIPLIERS:
        t1 = (
            STAGE1_NOMINAL
            * multiplier
        )

        t2 = (
            STAGE2_NOMINAL
            * multiplier
        )

        ok1 = (
            np.isfinite(stage1)
            & (stage1 < t1)
        )

        ok2 = (
            np.isfinite(stage2)
            & (stage2 < t2)
        )

        both = ok1 & ok2

        rows.append(
            {
                "multiplier": multiplier,
                "stage1_threshold": t1,
                "stage2_threshold": t2,
                "n_total": 500,
                "n_stage1_success": int(
                    ok1.sum()
                ),
                "stage1_success_rate": float(
                    ok1.mean()
                ),
                "n_stage2_success": int(
                    ok2.sum()
                ),
                "stage2_success_rate": float(
                    ok2.mean()
                ),
                "n_technical_success": int(
                    both.sum()
                ),
                "technical_success_rate": float(
                    both.mean()
                ),
                "n_fail_stage1_only": int(
                    np.sum(
                        (~ok1) & ok2
                    )
                ),
                "n_fail_stage2_only": int(
                    np.sum(
                        ok1 & (~ok2)
                    )
                ),
                "n_fail_both": int(
                    np.sum(
                        (~ok1) & (~ok2)
                    )
                ),
            }
        )

    scale = pd.DataFrame(rows)

    scale.to_csv(
        OUTPUT
        / "threshold_scale_sensitivity.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 2. Independent 2D sensitivity:
    #    Stage-1 and Stage-2 thresholds varied separately.
    # ------------------------------------------------------------

    grid_rows = []

    for m1 in MULTIPLIERS:
        for m2 in MULTIPLIERS:
            t1 = (
                STAGE1_NOMINAL
                * m1
            )

            t2 = (
                STAGE2_NOMINAL
                * m2
            )

            ok1 = (
                np.isfinite(stage1)
                & (stage1 < t1)
            )

            ok2 = (
                np.isfinite(stage2)
                & (stage2 < t2)
            )

            both = ok1 & ok2

            grid_rows.append(
                {
                    "stage1_multiplier": m1,
                    "stage2_multiplier": m2,
                    "stage1_threshold": t1,
                    "stage2_threshold": t2,
                    "n_technical_success": int(
                        both.sum()
                    ),
                    "technical_success_rate": float(
                        both.mean()
                    ),
                }
            )

    grid = pd.DataFrame(
        grid_rows
    )

    grid.to_csv(
        OUTPUT
        / "threshold_2D_sensitivity.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 3. Distribution summary / ECDF support.
    # ------------------------------------------------------------

    distribution_rows = []

    for name, values, nominal in [
        (
            "stage1_mismatch",
            stage1,
            STAGE1_NOMINAL,
        ),
        (
            "stage2_mismatch",
            stage2,
            STAGE2_NOMINAL,
        ),
    ]:
        stats = quantiles(values)

        distribution_rows.append(
            {
                "metric": name,
                "n_finite": int(
                    np.isfinite(values).sum()
                ),
                "nominal_threshold":
                    nominal,
                **stats,
                "max_over_nominal_threshold":
                    float(
                        np.nanmax(values)
                        / nominal
                    ),
            }
        )

    distribution = pd.DataFrame(
        distribution_rows
    )

    distribution.to_csv(
        OUTPUT
        / "mismatch_distribution_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 4. Explicit ECDF table.
    # ------------------------------------------------------------

    ecdf_frames = []

    for name, values in [
        ("stage1_mismatch", stage1),
        ("stage2_mismatch", stage2),
    ]:
        finite = np.sort(
            values[
                np.isfinite(values)
            ]
        )

        ecdf_frames.append(
            pd.DataFrame(
                {
                    "metric": name,
                    "mismatch": finite,
                    "ecdf": (
                        np.arange(
                            1,
                            len(finite) + 1,
                        )
                        / len(finite)
                    ),
                }
            )
        )

    ecdf = pd.concat(
        ecdf_frames,
        ignore_index=True,
    )

    ecdf.to_csv(
        OUTPUT
        / "mismatch_ecdf.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 5. Per-query critical multiplier.
    #
    # Smallest common multiplier m such that:
    #
    # stage1 < m * 0.05
    # stage2 < m * 0.01
    #
    # Thus:
    # m_critical = max(stage1/0.05, stage2/0.01)
    # ------------------------------------------------------------

    critical = np.maximum(
        stage1 / STAGE1_NOMINAL,
        stage2 / STAGE2_NOMINAL,
    )

    per_query = df[
        [
            "benchmark_id",
            "atlas_chart",
            "Mach",
            "alpha",
            "shoot_stage1_mismatch",
            "shoot_stage2_mismatch",
        ]
    ].copy()

    per_query[
        "stage1_fraction_of_nominal"
    ] = (
        stage1
        / STAGE1_NOMINAL
    )

    per_query[
        "stage2_fraction_of_nominal"
    ] = (
        stage2
        / STAGE2_NOMINAL
    )

    per_query[
        "critical_common_multiplier"
    ] = critical

    per_query.to_csv(
        OUTPUT
        / "per_query_threshold_margin.csv",
        index=False,
    )

    critical_finite = critical[
        np.isfinite(critical)
    ]

    critical_summary = {
        "n": int(
            len(critical_finite)
        ),
        "median": float(
            np.median(
                critical_finite
            )
        ),
        "p90": float(
            np.quantile(
                critical_finite,
                0.90,
            )
        ),
        "p95": float(
            np.quantile(
                critical_finite,
                0.95,
            )
        ),
        "p99": float(
            np.quantile(
                critical_finite,
                0.99,
            )
        ),
        "max": float(
            np.max(
                critical_finite
            )
        ),
    }

    (
        OUTPUT
        / "threshold_sensitivity_metadata.json"
    ).write_text(
        json.dumps(
            {
                "input": str(INPUT),
                "n_queries": 500,
                "stage1_nominal":
                    STAGE1_NOMINAL,
                "stage2_nominal":
                    STAGE2_NOMINAL,
                "multipliers":
                    MULTIPLIERS,
                "critical_multiplier_summary":
                    critical_summary,
            },
            indent=2,
        )
    )

    print()
    print("=" * 100)
    print("COMMON THRESHOLD SCALE")
    print("=" * 100)
    print(
        scale.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("MISMATCH DISTRIBUTIONS")
    print("=" * 100)
    print(
        distribution.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("CRITICAL COMMON MULTIPLIER")
    print("=" * 100)
    print(
        pd.Series(
            critical_summary
        ).to_string()
    )

    print()
    print("=" * 100)
    print("2D SUCCESS GRID")
    print("=" * 100)

    print(
        grid.pivot(
            index="stage1_multiplier",
            columns="stage2_multiplier",
            values="n_technical_success",
        ).to_string()
    )

    print()
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
