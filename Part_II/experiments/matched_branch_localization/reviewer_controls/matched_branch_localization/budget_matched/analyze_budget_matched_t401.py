#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "assets/pinn_supersonic/"
    "atlas2d_v1_continuousM/N76"
)

GENERIC_ROOT = (
    ROOT
    / "budget_matched_T401"
    / "generic"
)

OUT = (
    ROOT
    / "budget_matched_T401"
    / "analysis"
)

SEEDED_FILE = (
    ROOT
    / "shooting_T401"
    / "N76_T401_shooting_401.csv"
)

RANGES = [
    "000_051",
    "051_101",
    "101_151",
    "151_201",
    "201_251",
    "251_301",
    "301_351",
    "351_401",
]

TARGET_TOL = 1.0e-4


def pick(
    df: pd.DataFrame,
    names,
    *,
    required=True,
):
    for name in names:
        if name in df.columns:
            return name

    if required:
        raise RuntimeError(
            "Could not find any of columns "
            f"{names}. Available columns:\n"
            f"{df.columns.tolist()}"
        )

    return None


def exact_two_sided_discordant_p(
    seeded_only: int,
    generic_only: int,
):
    """
    Exact paired sign/McNemar test on discordant pairs,
    under p=0.5.
    """
    n = int(
        seeded_only
        + generic_only
    )

    if n == 0:
        return 1.0

    k = min(
        int(seeded_only),
        int(generic_only),
    )

    tail = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2.0 ** n)

    return min(
        1.0,
        2.0 * tail,
    )


def quantile(x, q):
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return float("nan")

    return float(
        np.quantile(x, q)
    )


def condition_summary(
    name: str,
    frame: pd.DataFrame,
    *,
    technical_col: str,
    recovery_col: str,
    error_col: str,
):
    error = frame[
        error_col
    ].to_numpy(float)

    return {
        "condition":
            name,

        "n":
            int(len(frame)),

        "technical_success":
            int(
                frame[
                    technical_col
                ].sum()
            ),

        "target_recovery":
            int(
                frame[
                    recovery_col
                ].sum()
            ),

        "target_error_median":
            float(
                np.median(error)
            ),

        "target_error_p90":
            quantile(
                error,
                0.90,
            ),

        "target_error_p95":
            quantile(
                error,
                0.95,
            ),

        "target_error_p99":
            quantile(
                error,
                0.99,
            ),

        "target_error_max":
            float(
                np.max(error)
            ),

        "error_le_1e-5":
            int(
                np.sum(
                    error <= 1.0e-5
                )
            ),

        "error_le_1e-4":
            int(
                np.sum(
                    error <= 1.0e-4
                )
            ),
    }


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SEEDED_FILE.is_file():
        raise FileNotFoundError(
            SEEDED_FILE
        )

    seeded = pd.read_csv(
        SEEDED_FILE
    )

    generic_parts = []

    for r in RANGES:
        p = (
            GENERIC_ROOT
            / f"generic_summary_{r}.csv"
        )

        if not p.is_file():
            raise FileNotFoundError(
                p
            )

        generic_parts.append(
            pd.read_csv(p)
        )

    generic = pd.concat(
        generic_parts,
        ignore_index=True,
    )

    print("=" * 100)
    print("PHASE 7 — INPUTS")
    print("=" * 100)

    print(
        "seeded rows  =",
        len(seeded),
    )
    print(
        "generic rows =",
        len(generic),
    )

    if len(seeded) != 401:
        raise RuntimeError(
            f"Expected seeded 401, got {len(seeded)}"
        )

    if len(generic) != 401:
        raise RuntimeError(
            f"Expected generic 401, got {len(generic)}"
        )

    # ------------------------------------------------------------
    # SEEDED
    # ------------------------------------------------------------

    for c in [
        "Mach",
        "alpha",
        "cr_reference",
        "ci_reference",
        "shoot_cr",
        "shoot_ci",
    ]:
        if c not in seeded.columns:
            raise RuntimeError(
                f"Seeded file missing {c}"
            )

    seeded = seeded.copy()

    if "technical_success" in seeded.columns:
        seeded[
            "seeded_technical"
        ] = seeded[
            "technical_success"
        ].astype(bool)

    else:
        seeded[
            "seeded_technical"
        ] = (
            seeded[
                "shoot_spectral_success"
            ].astype(bool)
            &
            seeded[
                "shoot_mode_success"
            ].astype(bool)
        )

    seeded[
        "seeded_error"
    ] = np.hypot(
        seeded[
            "shoot_cr"
        ].to_numpy(float)
        -
        seeded[
            "cr_reference"
        ].to_numpy(float),

        seeded[
            "shoot_ci"
        ].to_numpy(float)
        -
        seeded[
            "ci_reference"
        ].to_numpy(float),
    )

    seeded[
        "seeded_recovery"
    ] = (
        seeded[
            "seeded_error"
        ]
        <= TARGET_TOL
    )

    # ------------------------------------------------------------
    # GENERIC — robustly resolve actual output column names.
    # ------------------------------------------------------------

    g_cr_ref = pick(
        generic,
        [
            "cr_reference",
            "cr_ref",
            "reference_cr",
            "cr",
        ],
    )

    g_ci_ref = pick(
        generic,
        [
            "ci_reference",
            "ci_ref",
            "reference_ci",
            "ci",
        ],
    )

    g_cr = pick(
        generic,
        [
            "best_cr",
            "shoot_cr",
            "selected_cr",
            "cr_shoot",
            "cr_solution",
        ],
    )

    g_ci = pick(
        generic,
        [
            "best_ci",
            "shoot_ci",
            "selected_ci",
            "ci_shoot",
            "ci_solution",
        ],
    )

    generic = generic.copy()

    g_technical = pick(
        generic,
        [
            "technical_success",
            "best_technical_success",
            "success",
        ],
        required=False,
    )

    if g_technical is not None:
        generic[
            "generic_technical"
        ] = generic[
            g_technical
        ].astype(bool)

    else:
        g_spec = pick(
            generic,
            [
                "best_spectral_success",
                "spectral_success",
                "shoot_spectral_success",
            ],
        )

        g_mode = pick(
            generic,
            [
                "best_mode_success",
                "mode_success",
                "shoot_mode_success",
            ],
        )

        generic[
            "generic_technical"
        ] = (
            generic[
                g_spec
            ].astype(bool)
            &
            generic[
                g_mode
            ].astype(bool)
        )

    generic[
        "generic_error"
    ] = np.hypot(
        generic[
            g_cr
        ].to_numpy(float)
        -
        generic[
            g_cr_ref
        ].to_numpy(float),

        generic[
            g_ci
        ].to_numpy(float)
        -
        generic[
            g_ci_ref
        ].to_numpy(float),
    )

    generic[
        "generic_recovery"
    ] = (
        generic[
            "generic_error"
        ]
        <= TARGET_TOL
    )

    # ------------------------------------------------------------
    # Coordinate integrity + merge.
    # ------------------------------------------------------------

    if (
        seeded[
            ["Mach", "alpha"]
        ]
        .drop_duplicates()
        .shape[0]
        != 401
    ):
        raise RuntimeError(
            "Seeded coordinates are not unique"
        )

    if (
        generic[
            ["Mach", "alpha"]
        ]
        .drop_duplicates()
        .shape[0]
        != 401
    ):
        raise RuntimeError(
            "Generic coordinates are not unique"
        )

    seeded[
        "_M_key"
    ] = (
        seeded["Mach"]
        .astype(float)
        .round(12)
    )

    seeded[
        "_a_key"
    ] = (
        seeded["alpha"]
        .astype(float)
        .round(12)
    )

    generic[
        "_M_key"
    ] = (
        generic["Mach"]
        .astype(float)
        .round(12)
    )

    generic[
        "_a_key"
    ] = (
        generic["alpha"]
        .astype(float)
        .round(12)
    )

    seed_chart = pick(
        seeded,
        [
            "atlas_chart",
            "primary_chart",
        ],
        required=False,
    )

    seed_cols = [
        "_M_key",
        "_a_key",
        "Mach",
        "alpha",
        "cr_reference",
        "ci_reference",
        "shoot_cr",
        "shoot_ci",
        "seeded_technical",
        "seeded_error",
        "seeded_recovery",
    ]

    if seed_chart is not None:
        seed_cols.append(
            seed_chart
        )

    g_cols = [
        "_M_key",
        "_a_key",
        "Mach",
        "alpha",
        g_cr_ref,
        g_ci_ref,
        g_cr,
        g_ci,
        "generic_technical",
        "generic_error",
        "generic_recovery",
    ]

    S = seeded[
        seed_cols
    ].copy()

    G = generic[
        g_cols
    ].copy()

    # Rename before merge.
    rename_s = {
        "Mach":
            "Mach_seeded",

        "alpha":
            "alpha_seeded",

        "shoot_cr":
            "seeded_cr",

        "shoot_ci":
            "seeded_ci",
    }

    if seed_chart is not None:
        rename_s[
            seed_chart
        ] = "atlas_chart"

    S = S.rename(
        columns=rename_s
    )

    G = G.rename(
        columns={
            "Mach":
                "Mach_generic",

            "alpha":
                "alpha_generic",

            g_cr_ref:
                "generic_cr_reference",

            g_ci_ref:
                "generic_ci_reference",

            g_cr:
                "generic_cr",

            g_ci:
                "generic_ci",
        }
    )

    paired = S.merge(
        G,
        on=[
            "_M_key",
            "_a_key",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(paired) != 401:
        raise RuntimeError(
            f"Expected 401 paired rows, got {len(paired)}"
        )

    max_dm = float(
        np.max(
            np.abs(
                paired[
                    "Mach_seeded"
                ]
                -
                paired[
                    "Mach_generic"
                ]
            )
        )
    )

    max_da = float(
        np.max(
            np.abs(
                paired[
                    "alpha_seeded"
                ]
                -
                paired[
                    "alpha_generic"
                ]
            )
        )
    )

    max_dcr_ref = float(
        np.max(
            np.abs(
                paired[
                    "cr_reference"
                ]
                -
                paired[
                    "generic_cr_reference"
                ]
            )
        )
    )

    max_dci_ref = float(
        np.max(
            np.abs(
                paired[
                    "ci_reference"
                ]
                -
                paired[
                    "generic_ci_reference"
                ]
            )
        )
    )

    print()
    print("=" * 100)
    print("PAIRING INTEGRITY")
    print("=" * 100)

    print(
        "paired rows       =",
        len(paired),
    )
    print(
        "max |delta Mach|  =",
        f"{max_dm:.3e}",
    )
    print(
        "max |delta alpha| =",
        f"{max_da:.3e}",
    )
    print(
        "max |delta crref| =",
        f"{max_dcr_ref:.3e}",
    )
    print(
        "max |delta ciref| =",
        f"{max_dci_ref:.3e}",
    )

    if (
        max_dm > 1e-12
        or max_da > 1e-12
        or max_dcr_ref > 1e-12
        or max_dci_ref > 1e-12
    ):
        raise RuntimeError(
            "Seeded/generic pairing or references disagree"
        )

    paired[
        "Mach"
    ] = paired[
        "Mach_seeded"
    ]

    paired[
        "alpha"
    ] = paired[
        "alpha_seeded"
    ]

    # ------------------------------------------------------------
    # Condition summaries.
    # ------------------------------------------------------------

    seed_summary = condition_summary(
        "SEEDED",
        paired,
        technical_col=
            "seeded_technical",
        recovery_col=
            "seeded_recovery",
        error_col=
            "seeded_error",
    )

    generic_summary = condition_summary(
        "GENERIC",
        paired,
        technical_col=
            "generic_technical",
        recovery_col=
            "generic_recovery",
        error_col=
            "generic_error",
    )

    condition_table = pd.DataFrame(
        [
            seed_summary,
            generic_summary,
        ]
    )

    print()
    print("=" * 100)
    print("PHASE 7 — CONDITION SUMMARY")
    print("=" * 100)

    print(
        condition_table.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8e}",
        )
    )

    # ------------------------------------------------------------
    # Paired target recovery.
    # ------------------------------------------------------------

    sr = paired[
        "seeded_recovery"
    ].astype(bool)

    gr = paired[
        "generic_recovery"
    ].astype(bool)

    both = int(
        (sr & gr).sum()
    )

    seeded_only = int(
        (sr & ~gr).sum()
    )

    generic_only = int(
        (~sr & gr).sum()
    )

    neither = int(
        (~sr & ~gr).sum()
    )

    p_recovery = (
        exact_two_sided_discordant_p(
            seeded_only,
            generic_only,
        )
    )

    # Technical discordances separately.
    st = paired[
        "seeded_technical"
    ].astype(bool)

    gt = paired[
        "generic_technical"
    ].astype(bool)

    tech_both = int(
        (st & gt).sum()
    )

    tech_seeded_only = int(
        (st & ~gt).sum()
    )

    tech_generic_only = int(
        (~st & gt).sum()
    )

    tech_neither = int(
        (~st & ~gt).sum()
    )

    p_technical = (
        exact_two_sided_discordant_p(
            tech_seeded_only,
            tech_generic_only,
        )
    )

    print()
    print("=" * 100)
    print("PHASE 7 — PAIRED DISCORDANCES")
    print("=" * 100)

    print("TARGET BRANCH")
    print(
        "both recovered  =",
        both,
    )
    print(
        "seeded only     =",
        seeded_only,
    )
    print(
        "generic only    =",
        generic_only,
    )
    print(
        "neither         =",
        neither,
    )
    print(
        "exact paired p  =",
        f"{p_recovery:.16e}",
    )

    print()
    print("TECHNICAL SUCCESS")
    print(
        "both technical  =",
        tech_both,
    )
    print(
        "seeded only     =",
        tech_seeded_only,
    )
    print(
        "generic only    =",
        tech_generic_only,
    )
    print(
        "neither         =",
        tech_neither,
    )
    print(
        "exact paired p  =",
        f"{p_technical:.16e}",
    )

    # ------------------------------------------------------------
    # Continuous paired errors.
    # Positive = generic worse.
    # ------------------------------------------------------------

    paired[
        "generic_minus_seeded_error"
    ] = (
        paired[
            "generic_error"
        ]
        -
        paired[
            "seeded_error"
        ]
    )

    d = paired[
        "generic_minus_seeded_error"
    ].to_numpy(float)

    print()
    print("=" * 100)
    print("PHASE 7 — PAIRED TARGET ERROR")
    print("=" * 100)

    print(
        "generic - seeded median =",
        f"{np.median(d):.8e}",
    )
    print(
        "generic - seeded mean   =",
        f"{np.mean(d):.8e}",
    )
    print(
        "generic - seeded p95    =",
        f"{np.quantile(d, .95):.8e}",
    )

    print(
        "seeded lower error      =",
        int(
            np.sum(d > 0)
        ),
    )

    print(
        "generic lower error     =",
        int(
            np.sum(d < 0)
        ),
    )

    print(
        "ties                    =",
        int(
            np.sum(d == 0)
        ),
    )

    # ------------------------------------------------------------
    # Per-chart.
    # ------------------------------------------------------------

    by_chart_rows = []

    if "atlas_chart" in paired.columns:
        for chart, group in paired.groupby(
            "atlas_chart",
            sort=True,
        ):
            sr_c = group[
                "seeded_recovery"
            ].astype(bool)

            gr_c = group[
                "generic_recovery"
            ].astype(bool)

            by_chart_rows.append({
                "chart":
                    chart,

                "n":
                    int(len(group)),

                "seeded_technical":
                    int(
                        group[
                            "seeded_technical"
                        ].sum()
                    ),

                "generic_technical":
                    int(
                        group[
                            "generic_technical"
                        ].sum()
                    ),

                "seeded_recovery":
                    int(
                        sr_c.sum()
                    ),

                "generic_recovery":
                    int(
                        gr_c.sum()
                    ),

                "seeded_only":
                    int(
                        (
                            sr_c
                            & ~gr_c
                        ).sum()
                    ),

                "generic_only":
                    int(
                        (
                            ~sr_c
                            & gr_c
                        ).sum()
                    ),

                "seeded_error_median":
                    float(
                        group[
                            "seeded_error"
                        ].median()
                    ),

                "generic_error_median":
                    float(
                        group[
                            "generic_error"
                        ].median()
                    ),
            })

    by_chart = pd.DataFrame(
        by_chart_rows
    )

    if not by_chart.empty:
        print()
        print("=" * 100)
        print("PHASE 7 — BY CHART")
        print("=" * 100)

        print(
            by_chart.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.7e}",
            )
        )

    # ------------------------------------------------------------
    # Failure coordinates.
    # ------------------------------------------------------------

    failure_mask = (
        ~paired[
            "seeded_recovery"
        ]
        |
        ~paired[
            "generic_recovery"
        ]
    )

    failures = paired.loc[
        failure_mask
    ].copy()

    failure_cols = [
        "Mach",
        "alpha",
    ]

    if "atlas_chart" in failures:
        failure_cols.append(
            "atlas_chart"
        )

    failure_cols += [
        "cr_reference",
        "ci_reference",
        "seeded_cr",
        "seeded_ci",
        "seeded_technical",
        "seeded_error",
        "seeded_recovery",
        "generic_cr",
        "generic_ci",
        "generic_technical",
        "generic_error",
        "generic_recovery",
    ]

    print()
    print("=" * 100)
    print("PHASE 7 — FAILURES / DISCORDANCES")
    print("=" * 100)

    print(
        failures[
            failure_cols
        ]
        .sort_values(
            [
                "Mach",
                "alpha",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8e}",
        )
    )

    # ------------------------------------------------------------
    # Known historical seeded failure.
    # Pure diagnostic; not used in selection.
    # ------------------------------------------------------------

    known = paired[
        np.isclose(
            paired["Mach"],
            1.15,
            atol=1e-12,
            rtol=0,
        )
        &
        np.isclose(
            paired["alpha"],
            0.06,
            atol=1e-12,
            rtol=0,
        )
    ]

    print()
    print("=" * 100)
    print("KNOWN T401 SEEDED FAILURE M=1.15, alpha=0.06")
    print("=" * 100)

    if len(known) == 1:
        print(
            known[
                failure_cols
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.8e}",
            )
        )
    else:
        print(
            "Expected one known coordinate; found",
            len(known),
        )

    # ------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------

    paired.to_csv(
        OUT
        / "phase7_paired_T401.csv",
        index=False,
    )

    condition_table.to_csv(
        OUT
        / "phase7_condition_summary.csv",
        index=False,
    )

    if not by_chart.empty:
        by_chart.to_csv(
            OUT
            / "phase7_by_chart.csv",
            index=False,
        )

    failures[
        failure_cols
    ].to_csv(
        OUT
        / "phase7_failures.csv",
        index=False,
    )

    summary = {
        "n_paired":
            401,

        "target_tolerance":
            TARGET_TOL,

        "pairing": {
            "max_delta_Mach":
                max_dm,

            "max_delta_alpha":
                max_da,

            "max_delta_cr_reference":
                max_dcr_ref,

            "max_delta_ci_reference":
                max_dci_ref,
        },

        "seeded":
            seed_summary,

        "generic":
            generic_summary,

        "target_branch_discordances": {
            "both":
                both,

            "seeded_only":
                seeded_only,

            "generic_only":
                generic_only,

            "neither":
                neither,

            "exact_two_sided_p":
                p_recovery,
        },

        "technical_discordances": {
            "both":
                tech_both,

            "seeded_only":
                tech_seeded_only,

            "generic_only":
                tech_generic_only,

            "neither":
                tech_neither,

            "exact_two_sided_p":
                p_technical,
        },

        "paired_target_error_generic_minus_seeded": {
            "median":
                float(
                    np.median(d)
                ),

            "mean":
                float(
                    np.mean(d)
                ),

            "p95":
                float(
                    np.quantile(
                        d,
                        .95,
                    )
                ),

            "seeded_lower_error":
                int(
                    np.sum(
                        d > 0
                    )
                ),

            "generic_lower_error":
                int(
                    np.sum(
                        d < 0
                    )
                ),
        },

        "cost_comparison":
            "NOT PERFORMED: historical seeded T401 output "
            "does not retain full per-box retry counts.",
    }

    (
        OUT
        / "phase7_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 100)
    print("AUTOMATED VERDICT")
    print("=" * 100)

    ns = int(
        seed_summary[
            "target_recovery"
        ]
    )

    ng = int(
        generic_summary[
            "target_recovery"
        ]
    )

    if ns > ng:
        print(
            "PHASE 7: SUPPORTS PINN-INFORMED "
            "BRANCH-LOCALIZATION ADVANTAGE"
        )

        print(
            f"Seeded target recovery = "
            f"{ns}/401"
        )

        print(
            f"Generic target recovery = "
            f"{ng}/401"
        )

        print(
            "Interpretation: under the same external "
            "12-box local classical search budget, "
            "PINN-informed search centers recover the "
            "prescribed T401 branch more often than "
            "fixed generic centers."
        )

    elif ns == ng:
        print(
            "PHASE 7: NO TARGET-RECOVERY "
            "ADVANTAGE DETECTED"
        )

    else:
        print(
            "PHASE 7: GENERIC RECOVERY EXCEEDS SEEDED"
        )

    print()
    print(
        "IMPORTANT: this comparison tests the "
        "search-center / initialization information. "
        "It does NOT isolate a causal benefit from "
        "physics-informed losses."
    )

    print()
    print("WROTE:", OUT)


if __name__ == "__main__":
    main()
