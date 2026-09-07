#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ASSET_ROOT = Path("assets/pinn_supersonic")

OUT = Path(
    "assets/pinn_supersonic/"
    "atlas2d_v1_continuousM/N76/"
    "t401_failure_basin_analysis"
)

EXPECTED_N = 775

TARGET_CR = -2.549736936302885e-09
TARGET_CI = 0.1998472244836177

HISTORICAL_FAILURE_CR = 0.0
HISTORICAL_FAILURE_CI = 0.102065065

PINN_CR = 0.026089
PINN_CI = 0.115581

TARGET_TOL = 1.0e-4


def pick(df, names, required=True):
    for name in names:
        if name in df.columns:
            return name

    if required:
        raise RuntimeError(
            f"None of {names} found.\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    return None


def q(x, p):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return float("nan")

    return float(np.quantile(x, p))


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Discover result CSVs.
    # ----------------------------------------------------------

    csvs = []

    for p in ASSET_ROOT.rglob("*.csv"):
        s = str(p).lower()

        if (
            "basin" not in s
            and "t401_failure" not in s
        ):
            continue

        try:
            df = pd.read_csv(p, nrows=2)
        except Exception:
            continue

        cols = set(df.columns)

        if any(
            c in cols
            for c in [
                "cr0",
                "cr_center",
                "initial_cr",
            ]
        ) and any(
            c in cols
            for c in [
                "ci0",
                "ci_center",
                "initial_ci",
            ]
        ):
            csvs.append(p)

    if not csvs:
        raise RuntimeError(
            "No Phase-8 basin result CSV found."
        )

    print("=" * 100)
    print("PHASE 8 — INPUT FILES")
    print("=" * 100)

    frames = []

    for p in sorted(csvs):
        df = pd.read_csv(p)

        print(
            f"{p}: {len(df)} rows"
        )

        df["_source_file"] = str(p)

        frames.append(df)

    raw = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    # ----------------------------------------------------------
    # Resolve schema.
    # ----------------------------------------------------------

    cr0_col = pick(
        raw,
        [
            "cr0",
            "cr_center",
            "initial_cr",
            "seed_cr",
        ],
    )

    ci0_col = pick(
        raw,
        [
            "ci0",
            "ci_center",
            "initial_ci",
            "seed_ci",
        ],
    )

    shoot_cr_col = pick(
        raw,
        [
            "shoot_cr",
            "best_cr",
            "final_cr",
            "cr_solution",
        ],
    )

    shoot_ci_col = pick(
        raw,
        [
            "shoot_ci",
            "best_ci",
            "final_ci",
            "ci_solution",
        ],
    )

    technical_col = pick(
        raw,
        [
            "technical_success",
            "success",
        ],
        required=False,
    )

    class_col = pick(
        raw,
        [
            "classification",
            "basin_class",
            "class",
            "outcome_class",
        ],
        required=False,
    )

    mismatch_col = pick(
        raw,
        [
            "total_mismatch",
            "best_total_mismatch",
            "stage1_mismatch",
        ],
        required=False,
    )

    # ----------------------------------------------------------
    # Normalize coordinates.
    # ----------------------------------------------------------

    raw["cr0"] = pd.to_numeric(
        raw[cr0_col],
        errors="coerce",
    )

    raw["ci0"] = pd.to_numeric(
        raw[ci0_col],
        errors="coerce",
    )

    raw["shoot_cr"] = pd.to_numeric(
        raw[shoot_cr_col],
        errors="coerce",
    )

    raw["shoot_ci"] = pd.to_numeric(
        raw[shoot_ci_col],
        errors="coerce",
    )

    raw["_cr_key"] = raw[
        "cr0"
    ].round(12)

    raw["_ci_key"] = raw[
        "ci0"
    ].round(12)

    duplicate_mask = raw.duplicated(
        subset=[
            "_cr_key",
            "_ci_key",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        print()
        print(
            "Duplicate center rows before dedup =",
            int(duplicate_mask.sum()),
        )

    data = (
        raw
        .sort_values(
            [
                "_cr_key",
                "_ci_key",
            ]
        )
        .drop_duplicates(
            subset=[
                "_cr_key",
                "_ci_key",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 100)
    print("PHASE 8 — GRID INTEGRITY")
    print("=" * 100)

    print(
        "unique centers =",
        len(data),
    )

    print(
        "unique cr0 =",
        data["cr0"].nunique(),
    )

    print(
        "unique ci0 =",
        data["ci0"].nunique(),
    )

    if len(data) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} centers, "
            f"found {len(data)}"
        )

    # Expected original design: 25 x 31.
    if (
        data["cr0"].nunique() != 25
        or data["ci0"].nunique() != 31
    ):
        raise RuntimeError(
            "Expected a 25 x 31 initialization grid."
        )

    print(
        "grid = 25 x 31 =",
        25 * 31,
    )

    # ----------------------------------------------------------
    # Recompute diagnostics independently.
    # ----------------------------------------------------------

    data[
        "target_error"
    ] = np.hypot(
        data["shoot_cr"]
        - TARGET_CR,
        data["shoot_ci"]
        - TARGET_CI,
    )

    data[
        "failure_branch_error"
    ] = np.hypot(
        data["shoot_cr"]
        - HISTORICAL_FAILURE_CR,
        data["shoot_ci"]
        - HISTORICAL_FAILURE_CI,
    )

    if technical_col is not None:
        data[
            "technical_success_norm"
        ] = data[
            technical_col
        ].astype(bool)

    else:
        data[
            "technical_success_norm"
        ] = (
            np.isfinite(
                data["shoot_cr"]
            )
            &
            np.isfinite(
                data["shoot_ci"]
            )
        )

    data[
        "target_recovery_recomputed"
    ] = (
        data[
            "technical_success_norm"
        ]
        &
        (
            data[
                "target_error"
            ]
            <= TARGET_TOL
        )
    )

    # If worker already classified results, preserve it.
    if class_col is not None:
        data[
            "worker_class"
        ] = (
            data[
                class_col
            ]
            .astype(str)
        )
    else:
        data[
            "worker_class"
        ] = "UNAVAILABLE"

    # ----------------------------------------------------------
    # Outcome statistics.
    # ----------------------------------------------------------

    print()
    print("=" * 100)
    print("PHASE 8 — OUTCOMES")
    print("=" * 100)

    print(
        "technical success =",
        int(
            data[
                "technical_success_norm"
            ].sum()
        ),
        "/",
        len(data),
    )

    print(
        "target recovery   =",
        int(
            data[
                "target_recovery_recomputed"
            ].sum()
        ),
        "/",
        len(data),
    )

    if class_col is not None:
        print()
        print(
            "WORKER CLASSIFICATION:"
        )
        print(
            data[
                "worker_class"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    # ----------------------------------------------------------
    # PINN seed diagnostic.
    # Nearest tested initialization to historical PINN seed.
    # ----------------------------------------------------------

    data[
        "distance_to_pinn_seed"
    ] = np.hypot(
        data["cr0"]
        - PINN_CR,
        data["ci0"]
        - PINN_CI,
    )

    nearest = (
        data
        .sort_values(
            "distance_to_pinn_seed"
        )
        .head(1)
    )

    print()
    print("=" * 100)
    print("PHASE 8 — PINN SEED LOCATION")
    print("=" * 100)

    print(
        f"historical PINN seed ~ "
        f"({PINN_CR:.6f}, {PINN_CI:.6f})"
    )

    cols = [
        "cr0",
        "ci0",
        "shoot_cr",
        "shoot_ci",
        "technical_success_norm",
        "target_error",
        "failure_branch_error",
        "target_recovery_recomputed",
        "worker_class",
        "distance_to_pinn_seed",
    ]

    if mismatch_col is not None:
        cols.append(
            mismatch_col
        )

    print(
        nearest[
            cols
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8e}",
        )
    )

    # ----------------------------------------------------------
    # Target basin geometry.
    # ----------------------------------------------------------

    target = data[
        data[
            "target_recovery_recomputed"
        ]
    ].copy()

    if not target.empty:
        print()
        print("=" * 100)
        print("PHASE 8 — TARGET BASIN EXTENT")
        print("=" * 100)

        print(
            "cr0 min/max =",
            float(
                target["cr0"].min()
            ),
            float(
                target["cr0"].max()
            ),
        )

        print(
            "ci0 min/max =",
            float(
                target["ci0"].min()
            ),
            float(
                target["ci0"].max()
            ),
        )

        print(
            "target fraction =",
            f"{len(target)/len(data):.6f}",
        )

    # ----------------------------------------------------------
    # Per-ci and per-cr basin fractions.
    # ----------------------------------------------------------

    by_ci = (
        data
        .groupby(
            "ci0",
            as_index=False,
        )
        .agg(
            n=(
                "ci0",
                "size",
            ),
            target_count=(
                "target_recovery_recomputed",
                "sum",
            ),
            technical_count=(
                "technical_success_norm",
                "sum",
            ),
        )
    )

    by_ci[
        "target_fraction"
    ] = (
        by_ci[
            "target_count"
        ]
        /
        by_ci["n"]
    )

    by_cr = (
        data
        .groupby(
            "cr0",
            as_index=False,
        )
        .agg(
            n=(
                "cr0",
                "size",
            ),
            target_count=(
                "target_recovery_recomputed",
                "sum",
            ),
            technical_count=(
                "technical_success_norm",
                "sum",
            ),
        )
    )

    by_cr[
        "target_fraction"
    ] = (
        by_cr[
            "target_count"
        ]
        /
        by_cr["n"]
    )

    # ----------------------------------------------------------
    # Summary.
    # ----------------------------------------------------------

    summary = {
        "coordinate": {
            "Mach":
                1.15,
            "alpha":
                0.06,
        },

        "grid": {
            "n":
                int(len(data)),
            "n_cr":
                int(
                    data[
                        "cr0"
                    ].nunique()
                ),
            "n_ci":
                int(
                    data[
                        "ci0"
                    ].nunique()
                ),
            "cr_min":
                float(
                    data[
                        "cr0"
                    ].min()
                ),
            "cr_max":
                float(
                    data[
                        "cr0"
                    ].max()
                ),
            "ci_min":
                float(
                    data[
                        "ci0"
                    ].min()
                ),
            "ci_max":
                float(
                    data[
                        "ci0"
                    ].max()
                ),
        },

        "technical_success": {
            "n":
                int(
                    data[
                        "technical_success_norm"
                    ].sum()
                ),
            "fraction":
                float(
                    data[
                        "technical_success_norm"
                    ].mean()
                ),
        },

        "target_recovery": {
            "n":
                int(
                    data[
                        "target_recovery_recomputed"
                    ].sum()
                ),
            "fraction":
                float(
                    data[
                        "target_recovery_recomputed"
                    ].mean()
                ),
        },

        "target_error": {
            "median":
                float(
                    data[
                        "target_error"
                    ].median()
                ),
            "p95":
                q(
                    data[
                        "target_error"
                    ],
                    0.95,
                ),
            "min":
                float(
                    data[
                        "target_error"
                    ].min()
                ),
        },

        "historical_seed": {
            "pinn_cr":
                PINN_CR,
            "pinn_ci":
                PINN_CI,
        },

        "historical_target": {
            "cr":
                TARGET_CR,
            "ci":
                TARGET_CI,
        },

        "historical_failure_solution": {
            "cr":
                HISTORICAL_FAILURE_CR,
            "ci":
                HISTORICAL_FAILURE_CI,
        },
    }

    # ----------------------------------------------------------
    # Save.
    # ----------------------------------------------------------

    data.to_csv(
        OUT
        / "phase8_basin_all_points.csv",
        index=False,
    )

    target.to_csv(
        OUT
        / "phase8_target_basin_points.csv",
        index=False,
    )

    by_ci.to_csv(
        OUT
        / "phase8_target_fraction_by_ci0.csv",
        index=False,
    )

    by_cr.to_csv(
        OUT
        / "phase8_target_fraction_by_cr0.csv",
        index=False,
    )

    nearest.to_csv(
        OUT
        / "phase8_nearest_pinn_seed.csv",
        index=False,
    )

    (
        OUT
        / "phase8_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 100)
    print("PHASE 8 SUMMARY")
    print("=" * 100)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        "WROTE:",
        OUT,
    )


if __name__ == "__main__":
    main()
