from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path.cwd()

CAMPAIGN_PATH = (
    REPO
    / "classic_supersonic/scripts/"
      "validation/"
      "run_dense_supersonic_campaign.py"
)


GENERIC_SEEDS = [
    (0.10, 1.0e-4),
    (0.50, 1.0e-4),
    (0.90, 1.0e-4),
    (0.10, 2.0e-3),
    (0.50, 2.0e-3),
    (0.90, 2.0e-3),
]


CONFIG = {
    # Production root/integration settings.
    "method": "DOP853",
    "root_tolerance": 1.0e-8,
    "ci_floor": 1.0e-12,
    "ci_upper": 0.20,

    # Generic from-scratch search width.
    # The three cr seeds overlap and cover
    # approximately [0, 1.1].
    "cr_half_width": 0.20,

    # Both low-ci and ordinary unstable
    # solutions are covered by the two ci
    # seed families.
    "ci_factor": 100.0,
    "direct_ci_switch": 1.0e-3,
    "direct_ci_scale_floor": 1.0e-4,

    "log_ci_diff_step": 1.0e-4,
    "linear_ci_diff_step": 1.0e-3,

    "optimizer_xtol": 1.0e-11,
    "optimizer_ftol": 1.0e-11,
    "optimizer_gtol": 1.0e-11,

    "max_nfev": 100,
}


def load_campaign():
    # The production campaign performs sibling imports such as
    # ``test_kappa_q_modulus_reconstruction``.  When the campaign is
    # loaded dynamically, Python does not automatically add that script
    # directory to sys.path.
    validation_dir = CAMPAIGN_PATH.parent.resolve()

    search_dirs = [
        validation_dir,
        validation_dir.parent.resolve(),
        (REPO / "classic_supersonic/scripts").resolve(),
        (REPO / "classic_supersonic").resolve(),
    ]

    helper_matches = list(
        (REPO / "classic_supersonic").rglob(
            "test_kappa_q_modulus_reconstruction.py"
        )
    )

    for helper in helper_matches:
        search_dirs.append(
            helper.parent.resolve()
        )

    for directory in search_dirs:
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)

    if not helper_matches:
        raise FileNotFoundError(
            "Could not locate "
            "test_kappa_q_modulus_reconstruction.py"
        )

    print(
        "classical helper:",
        helper_matches[0],
        flush=True,
    )

    name = "final_classical_campaign"

    spec = importlib.util.spec_from_file_location(
        name,
        CAMPAIGN_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {CAMPAIGN_PATH}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    # Required for dataclasses defined in a
    # dynamically imported module.
    sys.modules[name] = module

    spec.loader.exec_module(
        module
    )

    return module


def key(m, a):
    return (
        round(float(m), 12),
        round(float(a), 12),
    )


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


def solve_point(
    campaign,
    *,
    Mach,
    alpha,
):

    start = time.perf_counter()

    accepted_results = []
    all_attempts = []

    for seed_index, (
        seed_cr,
        seed_ci,
    ) in enumerate(GENERIC_SEEDS):

        t_seed = time.perf_counter()

        result, attempts = (
            campaign.solve_with_fallbacks(
                Mach=float(Mach),
                alpha=float(alpha),
                seed_cr=float(seed_cr),
                seed_ci=float(seed_ci),
                config=CONFIG,
            )
        )

        seed_seconds = (
            time.perf_counter()
            - t_seed
        )

        for attempt in attempts:
            attempt = dict(attempt)

            attempt[
                "generic_seed_index"
            ] = seed_index

            attempt[
                "generic_seed_cr"
            ] = seed_cr

            attempt[
                "generic_seed_ci"
            ] = seed_ci

            attempt[
                "generic_seed_total_seconds"
            ] = seed_seconds

            all_attempts.append(
                attempt
            )

        if result is not None:

            candidate = dict(result)

            candidate[
                "generic_seed_index"
            ] = seed_index

            candidate[
                "generic_seed_cr"
            ] = seed_cr

            candidate[
                "generic_seed_ci"
            ] = seed_ci

            candidate[
                "generic_seed_total_seconds"
            ] = seed_seconds

            accepted_results.append(
                candidate
            )

    total_seconds = (
        time.perf_counter()
        - start
    )

    if accepted_results:

        # Reference-independent selection:
        # choose the accepted root with the
        # smallest Riccati matching residual.
        best = min(
            accepted_results,
            key=lambda r:
                float(
                    r[
                        "residual_norm"
                    ]
                ),
        )

        success = True

    else:

        finite_attempts = [
            r
            for r in all_attempts
            if np.isfinite(
                float(
                    r.get(
                        "residual_norm",
                        np.nan,
                    )
                )
            )
        ]

        if finite_attempts:

            best = min(
                finite_attempts,
                key=lambda r:
                    float(
                        r[
                            "residual_norm"
                        ]
                    ),
            )

        else:

            best = {}

        success = False

    objective_calls = int(
        sum(
            int(
                a.get(
                    "objective_calls",
                    0,
                )
            )
            for a in all_attempts
        )
    )

    integration_failures = int(
        sum(
            int(
                a.get(
                    "integration_failures",
                    0,
                )
            )
            for a in all_attempts
        )
    )

    return {
        "Mach": float(Mach),
        "alpha": float(alpha),

        "classical_success":
            bool(success),

        "classical_cr":
            float(
                best.get(
                    "cr",
                    np.nan,
                )
            ),

        "classical_ci":
            float(
                best.get(
                    "ci",
                    np.nan,
                )
            ),

        "classical_omega_i":
            float(alpha)
            * float(
                best.get(
                    "ci",
                    np.nan,
                )
            ),

        "classical_residual_norm":
            float(
                best.get(
                    "residual_norm",
                    np.nan,
                )
            ),

        "classical_delta_kappa":
            float(
                best.get(
                    "delta_kappa",
                    np.nan,
                )
            ),

        "classical_delta_q":
            float(
                best.get(
                    "delta_q",
                    np.nan,
                )
            ),

        "classical_parameterization":
            str(
                best.get(
                    "parameterization",
                    "",
                )
            ),

        "classical_settings_name":
            str(
                best.get(
                    "settings_name",
                    "",
                )
            ),

        "classical_best_seed_index":
            int(
                best.get(
                    "generic_seed_index",
                    -1,
                )
            ),

        "classical_best_seed_cr":
            float(
                best.get(
                    "generic_seed_cr",
                    np.nan,
                )
            ),

        "classical_best_seed_ci":
            float(
                best.get(
                    "generic_seed_ci",
                    np.nan,
                )
            ),

        "classical_n_generic_starts":
            len(GENERIC_SEEDS),

        "classical_n_accepted_starts":
            len(
                accepted_results
            ),

        "classical_n_solver_attempts":
            len(all_attempts),

        "classical_objective_calls":
            objective_calls,

        "classical_integration_failures":
            integration_failures,

        "classical_seconds":
            float(
                total_seconds
            ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    campaign = load_campaign()

    inp = pd.read_csv(
        args.input
    )

    if "cost_id" not in inp.columns:
        inp = inp.copy()
        inp["cost_id"] = np.arange(
            len(inp)
        )

    existing = pd.DataFrame()

    if args.output.is_file():
        existing = pd.read_csv(
            args.output
        )

    done = set()

    if (
        not existing.empty
        and "cost_id"
        in existing.columns
    ):
        done = set(
            int(x)
            for x in existing[
                "cost_id"
            ]
        )

    rows = (
        existing.to_dict(
            "records"
        )
        if not existing.empty
        else []
    )

    for i, row in inp.iterrows():

        cost_id = int(
            row["cost_id"]
        )

        if cost_id in done:

            print(
                f"SKIP cost_id={cost_id}",
                flush=True,
            )

            continue

        Mach = float(
            row["Mach"]
        )

        alpha = float(
            row["alpha"]
        )

        print(
            f"START cost_id={cost_id} "
            f"M={Mach:.6f} "
            f"alpha={alpha:.6f}",
            flush=True,
        )

        try:

            result = solve_point(
                campaign,
                Mach=Mach,
                alpha=alpha,
            )

            out = {
                "cost_id":
                    cost_id,
                **result,
                "status":
                    "COMPLETED",
            }

        except Exception as exc:

            out = {
                "cost_id":
                    cost_id,
                "Mach":
                    Mach,
                "alpha":
                    alpha,
                "classical_success":
                    False,
                "classical_seconds":
                    np.nan,
                "status":
                    "EXCEPTION",
                "exception":
                    repr(exc),
            }

        rows.append(out)

        frame = (
            pd.DataFrame(rows)
            .sort_values(
                "cost_id"
            )
            .drop_duplicates(
                "cost_id",
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        atomic_csv(
            args.output,
            frame,
        )

        print(
            "DONE "
            f"cost_id={cost_id} "
            f"success="
            f"{out.get('classical_success')} "
            f"seconds="
            f"{out.get('classical_seconds')}",
            flush=True,
        )

    final = pd.read_csv(
        args.output
    )

    expected = set(
        int(x)
        for x in inp["cost_id"]
    )

    found = set(
        int(x)
        for x in final[
            "cost_id"
        ]
    )

    missing = (
        expected - found
    )

    if missing:
        raise RuntimeError(
            f"Missing ids: {sorted(missing)}"
        )

    print()
    print("=" * 80)
    print("CLASSICAL CHUNK COMPLETE")
    print("=" * 80)

    print(
        final.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
