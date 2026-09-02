from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


REPO = Path.cwd()

PREDICTIONS = (
    REPO / 'assets/pinn_supersonic/csv/pinn_direct/validation/table_N76_validation_predictions_64.csv'
)

OUTPUT_ROOT = (
    REPO
    / "assets/pinn_supersonic/atlas2d_v1/N76/"
      "corrector_benchmark"
)

COMPARE_GEP_SCRIPT = (
    REPO
    / "code/scripts/evaluation/evaluate_compare_supersonic_gep_candidates_vs_shooting.py"
)

SHOOTING_SCRIPT = (
    REPO
    / "code/scripts/evaluation/track_supersonic_shooting_multistart.py"
)

POWELL_SCRIPT = (
    REPO
    / "scripts/dev/benchmark_supersonic_M150_pinn_powell_spectrum.py"
)


# ----------------------------------------------------------------------
# Dynamic module loading.
#
# Important:
# compare_supersonic_gep_candidates_vs_shooting.py already contains the
# repository-specific import logic required to locate classical_solver.
# We deliberately import that working module instead of hard-coding a
# guessed GEP source path.
# ----------------------------------------------------------------------

def load_module(
    name: str,
    path: Path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load module from {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module

    spec.loader.exec_module(
        module
    )

    return module


def parser_defaults(
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:

    result: dict[str, Any] = {}

    for action in parser._actions:

        if action.dest == "help":
            continue

        result[action.dest] = action.default

    return result


def source_argparse_defaults(
    path: Path,
) -> dict[str, Any]:
    """
    Read literal argparse defaults directly from a Python source file.

    This allows us to reuse the historical Powell optimizer defaults
    without importing/running its main().
    """

    tree = ast.parse(
        path.read_text()
    )

    defaults: dict[str, Any] = {}

    for node in ast.walk(tree):

        if not isinstance(node, ast.Call):
            continue

        func = node.func

        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "add_argument"
        ):
            continue

        if not node.args:
            continue

        try:
            option = ast.literal_eval(
                node.args[0]
            )
        except Exception:
            continue

        if not (
            isinstance(option, str)
            and option.startswith("--")
        ):
            continue

        dest = (
            option[2:]
            .replace("-", "_")
        )

        default_found = False

        for kw in node.keywords:

            if kw.arg != "default":
                continue

            try:
                defaults[dest] = (
                    ast.literal_eval(
                        kw.value
                    )
                )

                default_found = True

            except Exception:
                pass

        if not default_found:

            for kw in node.keywords:

                if kw.arg != "action":
                    continue

                try:
                    action = ast.literal_eval(
                        kw.value
                    )
                except Exception:
                    continue

                if action == "store_true":
                    defaults[dest] = False

                elif action == "store_false":
                    defaults[dest] = True

    return defaults


def unique_floats(
    values: Any,
) -> list[float]:

    if values is None:
        return []

    if isinstance(
        values,
        (float, int),
    ):
        values = [values]

    result = []

    for value in values:

        x = float(value)

        if not any(
            np.isclose(
                x,
                y,
                atol=1e-15,
                rtol=0.0,
            )
            for y in result
        ):
            result.append(x)

    return result


def spectral_error(
    cr: float,
    ci: float,
    cr_ref: float,
    ci_ref: float,
) -> float:

    return float(
        np.hypot(
            cr - cr_ref,
            ci - ci_ref,
        )
    )


def safe_float(
    value: Any,
    default: float = np.nan,
) -> float:

    try:
        result = float(value)
    except Exception:
        return float(default)

    return result


def finite(
    value: Any,
) -> bool:

    try:
        return bool(
            np.isfinite(
                float(value)
            )
        )
    except Exception:
        return False


# ----------------------------------------------------------------------
# GEP
# ----------------------------------------------------------------------

def run_gep(
    *,
    gep_module,
    gep_defaults: dict[str, Any],
    mach: float,
    alpha: float,
    cr_seed: float,
    ci_seed: float,
) -> dict[str, Any]:

    start = time.perf_counter()

    result: dict[str, Any] = {
        "gep_status": "FAILED",
        "gep_error": "",
        "gep_n_candidates": 0,
        "gep_cr": np.nan,
        "gep_ci": np.nan,
        "gep_score_to_pinn": np.nan,
        "gep_second_score_to_pinn": np.nan,
        "gep_score_gap": np.nan,
    }

    try:

        n_points = int(
            gep_defaults["n_points"]
        )

        mapping_kind = (
            gep_defaults["mapping_kind"]
        )

        mapping_scale = float(
            gep_defaults["mapping_scale"]
        )

        cubic_delta = float(
            gep_defaults["cubic_delta"]
        )

        xi_max = float(
            gep_defaults["xi_max"]
        )

        max_abs_c = float(
            gep_defaults["max_abs_c"]
        )

        ci_weight = float(
            gep_defaults["ci_weight"]
        )

        solver = (
            gep_module.NotebookStyleDenseGEPSolver(
                alpha=float(alpha),
                Mach=float(mach),
                n_points=n_points,
                mapping_kind=mapping_kind,
                mapping_scale=mapping_scale,
                cubic_delta=cubic_delta,
                xi_max=xi_max,
            )
        )

        # We are targeting the unstable KH branch.
        # ci > 0 is a physical constraint known before validation and
        # therefore does not constitute reference leakage.
        modes = (
            gep_module.extract_raw_modes_with_vectors(
                solver,
                max_abs_c=max_abs_c,
                positive_ci_only=True,
            )
        )

        result[
            "gep_n_candidates"
        ] = len(modes)

        if not modes:
            raise RuntimeError(
                "No admissible positive-ci GEP mode."
            )

        def score(
            mode: dict[str, Any],
        ) -> float:

            return float(
                np.hypot(
                    float(mode["cr"])
                    - cr_seed,

                    ci_weight
                    * (
                        float(mode["ci"])
                        - ci_seed
                    ),
                )
            )

        ranked = sorted(
            modes,
            key=score,
        )

        best = ranked[0]

        best_score = score(
            best
        )

        second_score = (
            score(ranked[1])
            if len(ranked) >= 2
            else np.nan
        )

        result.update(
            {
                "gep_status": "COMPLETED",
                "gep_cr": float(
                    best["cr"]
                ),
                "gep_ci": float(
                    best["ci"]
                ),
                "gep_score_to_pinn":
                    best_score,
                "gep_second_score_to_pinn":
                    second_score,
                "gep_score_gap":
                    (
                        second_score
                        - best_score
                        if np.isfinite(
                            second_score
                        )
                        else np.nan
                    ),
                "gep_n_points":
                    n_points,
            }
        )

    except Exception as exc:

        result[
            "gep_error"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

    result[
        "gep_seconds"
    ] = (
        time.perf_counter()
        - start
    )

    return result


# ----------------------------------------------------------------------
# Shooting multistart
# ----------------------------------------------------------------------

def run_shooting(
    *,
    shooting_module,
    shooting_defaults: dict[str, Any],
    mach: float,
    alpha: float,
    cr_seed: float,
    ci_seed: float,
) -> dict[str, Any]:

    start = time.perf_counter()

    result: dict[str, Any] = {
        "shoot_status": "FAILED",
        "shoot_error": "",
        "shoot_cr": np.nan,
        "shoot_ci": np.nan,
        "shoot_stage1_mismatch": np.nan,
        "shoot_stage2_mismatch": np.nan,
        "shoot_total_mismatch": np.nan,
        "shoot_spectral_success": False,
        "shoot_mode_success": False,
        "shoot_retry": np.nan,
        "shoot_cr_half_window": np.nan,
        "shoot_ci_half_window": np.nan,
        "shoot_n_candidates": 0,
    }

    try:

        cr_windows = unique_floats(
            shooting_defaults[
                "cr_half_windows"
            ]
        )

        ci_windows = unique_floats(
            shooting_defaults[
                "ci_half_windows"
            ]
        )

        if not cr_windows:
            raise RuntimeError(
                "No default cr_half_windows."
            )

        if not ci_windows:
            raise RuntimeError(
                "No default ci_half_windows."
            )

        candidates: list[
            dict[str, Any]
        ] = []

        for cr_half in cr_windows:

            for ci_half in ci_windows:

                try:

                    (
                        solver,
                        shooting_result,
                        retry_idx,
                        used_cr_half,
                        used_ci_half,
                    ) = (
                        shooting_module
                        .multistart_single_box(
                            alpha=float(alpha),
                            mach=float(mach),

                            match_y=float(
                                shooting_defaults[
                                    "match_y"
                                ]
                            ),

                            # Mapping is deliberately fixed on:
                            # this is the convention used in the
                            # existing direct GEP-vs-shooting script.
                            use_mapping=True,

                            mapping_scale=5.0,

                            min_y_limit=float(
                                shooting_defaults[
                                    "min_y_limit"
                                ]
                            ),

                            max_y_limit=float(
                                shooting_defaults[
                                    "max_y_limit"
                                ]
                            ),

                            y_limit_factor=float(
                                shooting_defaults[
                                    "y_limit_factor"
                                ]
                            ),

                            amp_lower_bound=float(
                                shooting_defaults[
                                    "amp_lower_bound"
                                ]
                            ),

                            amp_upper_bound=float(
                                shooting_defaults[
                                    "amp_upper_bound"
                                ]
                            ),

                            cr_center=max(
                                0.0,
                                float(cr_seed),
                            ),

                            ci_center=max(
                                1e-4,
                                float(ci_seed),
                            ),

                            cr_half_window=float(
                                cr_half
                            ),

                            ci_half_window=float(
                                ci_half
                            ),

                            retry_growth=float(
                                shooting_defaults[
                                    "retry_growth"
                                ]
                            ),

                            max_retries=int(
                                shooting_defaults[
                                    "max_retries"
                                ]
                            ),

                            max_iter=int(
                                shooting_defaults[
                                    "max_iter"
                                ]
                            ),

                            grid_size=int(
                                shooting_defaults[
                                    "grid_size"
                                ]
                            ),
                        )
                    )

                    stage1 = safe_float(
                        shooting_result
                        .stage1_mismatch
                    )

                    stage2 = safe_float(
                        shooting_result
                        .stage2_mismatch
                    )

                    total = (
                        stage1 + stage2
                    )

                    candidates.append(
                        {
                            "solver":
                                solver,

                            "result":
                                shooting_result,

                            "retry":
                                retry_idx,

                            "cr_half":
                                used_cr_half,

                            "ci_half":
                                used_ci_half,

                            "stage1":
                                stage1,

                            "stage2":
                                stage2,

                            "total":
                                total,

                            "technical_success":
                                bool(
                                    shooting_result
                                    .spectral_success
                                    and
                                    shooting_result
                                    .mode_success
                                ),

                            "delta_seed":
                                float(
                                    np.hypot(
                                        float(
                                            shooting_result.cr
                                        )
                                        - cr_seed,

                                        float(
                                            shooting_result.ci
                                        )
                                        - ci_seed,
                                    )
                                ),
                        }
                    )

                except Exception:
                    continue

        result[
            "shoot_n_candidates"
        ] = len(candidates)

        if not candidates:
            raise RuntimeError(
                "No shooting candidate returned."
            )

        # Selection does NOT use reference.
        #
        # 1. technical spectral+mode success
        # 2. smallest Riccati/amplitude mismatch
        # 3. smallest displacement from PINN
        best = min(
            candidates,
            key=lambda row: (
                not row[
                    "technical_success"
                ],
                row["total"],
                row["delta_seed"],
            ),
        )

        shooting_result = (
            best["result"]
        )

        result.update(
            {
                "shoot_status":
                    "COMPLETED",

                "shoot_cr":
                    float(
                        shooting_result.cr
                    ),

                "shoot_ci":
                    float(
                        shooting_result.ci
                    ),

                "shoot_stage1_mismatch":
                    float(
                        best["stage1"]
                    ),

                "shoot_stage2_mismatch":
                    float(
                        best["stage2"]
                    ),

                "shoot_total_mismatch":
                    float(
                        best["total"]
                    ),

                "shoot_spectral_success":
                    bool(
                        shooting_result
                        .spectral_success
                    ),

                "shoot_mode_success":
                    bool(
                        shooting_result
                        .mode_success
                    ),

                "shoot_retry":
                    int(
                        best["retry"]
                    ),

                "shoot_cr_half_window":
                    float(
                        best["cr_half"]
                    ),

                "shoot_ci_half_window":
                    float(
                        best["ci_half"]
                    ),
            }
        )

    except Exception as exc:

        result[
            "shoot_error"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

    result[
        "shoot_seconds"
    ] = (
        time.perf_counter()
        - start
    )

    return result


# ----------------------------------------------------------------------
# Powell stage-1 corrector
# ----------------------------------------------------------------------

def run_powell(
    *,
    solver_class,
    shooting_defaults: dict[str, Any],
    powell_defaults: dict[str, Any],
    mach: float,
    alpha: float,
    cr_seed: float,
    ci_seed: float,
) -> dict[str, Any]:

    start = time.perf_counter()

    result: dict[str, Any] = {
        "powell_status": "FAILED",
        "powell_error": "",
        "powell_optimizer_success": False,
        "powell_optimizer_message": "",
        "powell_cr": np.nan,
        "powell_ci": np.nan,
        "powell_initial_mismatch": np.nan,
        "powell_corrected_mismatch": np.nan,
        "powell_mismatch_reduction": np.nan,
        "powell_nfev": np.nan,
        "powell_nit": np.nan,
    }

    try:

        cr_windows = unique_floats(
            shooting_defaults[
                "cr_half_windows"
            ]
        )

        ci_windows = unique_floats(
            shooting_defaults[
                "ci_half_windows"
            ]
        )

        # Same PINN-centered search neighbourhood as the largest
        # predefined shooting box. This choice uses no validation
        # reference and gives Powell and shooting comparable reach.
        cr_half = max(
            cr_windows
        )

        ci_half = max(
            ci_windows
        )

        cr_lower = max(
            0.0,
            cr_seed - cr_half,
        )

        cr_upper = (
            cr_seed + cr_half
        )

        ci_lower = max(
            1e-4,
            ci_seed - ci_half,
        )

        ci_upper = (
            ci_seed + ci_half
        )

        solver_kwargs = {
            "alpha":
                float(alpha),

            "Mach":
                float(mach),

            "match_y":
                float(
                    powell_defaults.get(
                        "match_y",
                        shooting_defaults[
                            "match_y"
                        ],
                    )
                ),

            "use_mapping":
                True,

            "mapping_scale":
                float(
                    powell_defaults.get(
                        "mapping_scale",
                        5.0,
                    )
                ),

            "min_y_limit":
                float(
                    powell_defaults.get(
                        "min_y_limit",
                        shooting_defaults[
                            "min_y_limit"
                        ],
                    )
                ),

            "max_y_limit":
                float(
                    powell_defaults.get(
                        "max_y_limit",
                        shooting_defaults[
                            "max_y_limit"
                        ],
                    )
                ),

            "y_limit_factor":
                float(
                    powell_defaults.get(
                        "y_limit_factor",
                        shooting_defaults[
                            "y_limit_factor"
                        ],
                    )
                ),
        }

        max_step = (
            powell_defaults.get(
                "max_step"
            )
        )

        if (
            max_step is not None
            and finite(max_step)
        ):
            solver_kwargs[
                "max_step"
            ] = float(max_step)

        solver = solver_class(
            **solver_kwargs
        )

        evaluations = 0

        def objective(
            parameters: np.ndarray,
        ) -> float:

            nonlocal evaluations

            evaluations += 1

            cr = float(
                parameters[0]
            )

            ci = float(
                parameters[1]
            )

            if (
                not np.isfinite(cr)
                or not np.isfinite(ci)
                or cr < 0.0
                or ci <= 0.0
            ):
                return 1.0e12

            try:
                value = float(
                    solver.stage1_mismatch(
                        cr,
                        ci,
                    )
                )

            except Exception:
                return 1.0e12

            if not np.isfinite(
                value
            ):
                return 1.0e12

            return value

        x0 = np.array(
            [
                np.clip(
                    cr_seed,
                    cr_lower,
                    cr_upper,
                ),
                np.clip(
                    ci_seed,
                    ci_lower,
                    ci_upper,
                ),
            ],
            dtype=float,
        )

        initial_mismatch = (
            objective(
                x0
            )
        )

        options: dict[
            str,
            Any,
        ] = {
            "disp": False,
        }

        # Exact names observed in the historical benchmark.
        for name in [
            "maxiter",
            "maxfev",
            "xtol",
            "ftol",
        ]:

            value = (
                powell_defaults.get(
                    name
                )
            )

            if value is not None:
                options[name] = value

        optimization = minimize(
            objective,
            x0,
            method="Powell",
            bounds=[
                (
                    cr_lower,
                    cr_upper,
                ),
                (
                    ci_lower,
                    ci_upper,
                ),
            ],
            options=options,
        )

        corrected = np.asarray(
            optimization.x,
            dtype=float,
        )

        cr_corrected = float(
            corrected[0]
        )

        ci_corrected = float(
            corrected[1]
        )

        corrected_mismatch = float(
            optimization.fun
        )

        result.update(
            {
                "powell_status":
                    "COMPLETED",

                "powell_optimizer_success":
                    bool(
                        optimization.success
                    ),

                "powell_optimizer_message":
                    str(
                        optimization.message
                    ),

                "powell_cr":
                    cr_corrected,

                "powell_ci":
                    ci_corrected,

                "powell_initial_mismatch":
                    float(
                        initial_mismatch
                    ),

                "powell_corrected_mismatch":
                    corrected_mismatch,

                "powell_mismatch_reduction":
                    float(
                        initial_mismatch
                        / max(
                            corrected_mismatch,
                            1e-30,
                        )
                    ),

                "powell_nfev":
                    int(
                        getattr(
                            optimization,
                            "nfev",
                            evaluations,
                        )
                    ),

                "powell_nit":
                    int(
                        getattr(
                            optimization,
                            "nit",
                            -1,
                        )
                    ),

                "powell_cr_lower":
                    cr_lower,

                "powell_cr_upper":
                    cr_upper,

                "powell_ci_lower":
                    ci_lower,

                "powell_ci_upper":
                    ci_upper,
            }
        )

    except Exception as exc:

        result[
            "powell_error"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

    result[
        "powell_seconds"
    ] = (
        time.perf_counter()
        - start
    )

    return result


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

def method_summary(
    frame: pd.DataFrame,
    prefix: str,
) -> dict[str, Any]:

    error_col = (
        f"{prefix}_spectral_error"
    )

    status_col = (
        f"{prefix}_status"
    )

    valid = (
        frame[
            error_col
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .to_numpy(float)
    )

    if len(valid) == 0:

        return {
            "method": prefix,
            "n": len(frame),
            "n_completed": 0,
        }

    pinn = frame.loc[
        frame[
            error_col
        ].notna(),
        "pinn_spectral_error",
    ].to_numpy(float)

    corrected = frame.loc[
        frame[
            error_col
        ].notna(),
        error_col,
    ].to_numpy(float)

    return {
        "method":
            prefix,

        "n":
            int(len(frame)),

        "n_completed":
            int(
                frame[
                    status_col
                ].eq(
                    "COMPLETED"
                ).sum()
            ),

        "mean_error":
            float(
                np.mean(valid)
            ),

        "median_error":
            float(
                np.median(valid)
            ),

        "p95_error":
            float(
                np.quantile(
                    valid,
                    0.95,
                )
            ),

        "max_error":
            float(
                np.max(valid)
            ),

        "n_le_1e-4":
            int(
                np.sum(
                    valid <= 1e-4
                )
            ),

        "n_le_5e-4":
            int(
                np.sum(
                    valid <= 5e-4
                )
            ),

        "n_le_1e-3":
            int(
                np.sum(
                    valid <= 1e-3
                )
            ),

        "n_le_5e-3":
            int(
                np.sum(
                    valid <= 5e-3
                )
            ),

        "n_le_1e-2":
            int(
                np.sum(
                    valid <= 1e-2
                )
            ),

        "mean_improvement_factor":
            float(
                np.mean(
                    pinn
                    / np.maximum(
                        corrected,
                        1e-30,
                    )
                )
            ),

        "median_improvement_factor":
            float(
                np.median(
                    pinn
                    / np.maximum(
                        corrected,
                        1e-30,
                    )
                )
            ),

        "n_improved":
            int(
                np.sum(
                    corrected < pinn
                )
            ),

        "n_worsened":
            int(
                np.sum(
                    corrected > pinn
                )
            ),

        "mean_seconds":
            float(
                frame[
                    f"{prefix}_seconds"
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .mean()
            ),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of validation "
            "points to process."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not PREDICTIONS.is_file():
        raise FileNotFoundError(
            PREDICTIONS
        )

    frame = pd.read_csv(
        PREDICTIONS
    )

    required = {
        "Mach",
        "alpha",
        "cr",
        "ci",
        "cr_pred",
        "ci_pred",
        "atlas_chart",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    if len(frame) != 64:
        raise RuntimeError(
            f"Expected 64 points, got {len(frame)}."
        )

    frame = (
        frame
        .sort_values(
            [
                "atlas_chart",
                "Mach",
                "alpha",
            ]
        )
        .reset_index(drop=True)
    )

    frame[
        "benchmark_id"
    ] = np.arange(
        len(frame)
    )

    print(
        "Loading working solver modules..."
    )

    # Load GEP module FIRST because it contains the
    # repository-specific classical_solver path setup.
    gep_module = load_module(
        "atlas_gep_compare",
        COMPARE_GEP_SCRIPT,
    )

    shooting_module = load_module(
        "atlas_shooting_multistart",
        SHOOTING_SCRIPT,
    )

    gep_defaults = parser_defaults(
        gep_module
        .build_parser()
    )

    shooting_defaults = parser_defaults(
        shooting_module
        .build_parser()
    )

    powell_defaults = (
        source_argparse_defaults(
            POWELL_SCRIPT
        )
    )

    solver_class = (
        shooting_module
        .Mstab17SupersonicSolver
    )

    settings = {
        "gep_defaults":
            {
                k: v
                for k, v
                in gep_defaults.items()
                if isinstance(
                    v,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    ),
                )
            },

        "shooting_defaults":
            {
                k: v
                for k, v
                in shooting_defaults.items()
                if isinstance(
                    v,
                    (
                        str,
                        int,
                        float,
                        bool,
                        list,
                        tuple,
                        type(None),
                    ),
                )
            },

        "powell_defaults":
            powell_defaults,

        "selection_policy": {
            "GEP":
                (
                    "positive-ci finite mode "
                    "with smallest weighted "
                    "distance to PINN seed"
                ),

            "shooting":
                (
                    "technical-success candidate "
                    "with smallest stage1+stage2 "
                    "mismatch; seed distance as "
                    "tie-breaker"
                ),

            "Powell":
                (
                    "bounded stage1 mismatch "
                    "minimization initialized "
                    "exactly at PINN seed"
                ),

            "reference_used_for_selection":
                False,
        },
    }

    (
        OUTPUT_ROOT
        / "benchmark_settings.json"
    ).write_text(
        json.dumps(
            settings,
            indent=2,
            default=str,
        )
    )

    checkpoint_path = (
        OUTPUT_ROOT
        / "corrector_benchmark_64.csv"
    )

    existing: dict[
        int,
        dict[str, Any],
    ] = {}

    if (
        args.resume
        and checkpoint_path.is_file()
    ):

        old = pd.read_csv(
            checkpoint_path
        )

        for _, row in old.iterrows():

            existing[
                int(
                    row[
                        "benchmark_id"
                    ]
                )
            ] = (
                row.to_dict()
            )

        print(
            f"Resume: {len(existing)} "
            "points already available."
        )

    rows: list[
        dict[str, Any]
    ] = []

    n_target = (
        len(frame)
        if args.limit is None
        else min(
            len(frame),
            int(args.limit),
        )
    )

    for idx in range(
        n_target
    ):

        source = frame.iloc[
            idx
        ]

        benchmark_id = int(
            source[
                "benchmark_id"
            ]
        )

        if benchmark_id in existing:

            rows.append(
                existing[
                    benchmark_id
                ]
            )

            print(
                f"[{idx+1:02d}/{n_target:02d}] "
                f"id={benchmark_id:02d} "
                "SKIP existing"
            )

            continue

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

        cr_seed = float(
            source["cr_pred"]
        )

        ci_seed = float(
            source["ci_pred"]
        )

        chart = str(
            source[
                "atlas_chart"
            ]
        )

        pinn_error = (
            spectral_error(
                cr_seed,
                ci_seed,
                cr_ref,
                ci_ref,
            )
        )

        print()
        print("=" * 110)
        print(
            f"[{idx+1:02d}/{n_target:02d}] "
            f"id={benchmark_id:02d} "
            f"{chart} "
            f"M={mach:.3f} "
            f"alpha={alpha:.3f}"
        )
        print(
            f"PINN seed = "
            f"({cr_seed:.8f}, "
            f"{ci_seed:.8f})"
        )
        print(
            f"PINN validation error = "
            f"{pinn_error:.6e}"
        )

        row: dict[
            str,
            Any,
        ] = {
            "benchmark_id":
                benchmark_id,

            "atlas_chart":
                chart,

            "Mach":
                mach,

            "alpha":
                alpha,

            "cr_reference":
                cr_ref,

            "ci_reference":
                ci_ref,

            "cr_pinn":
                cr_seed,

            "ci_pinn":
                ci_seed,

            "pinn_spectral_error":
                pinn_error,
        }

        # ----------------------------------------------------------
        # GEP
        # ----------------------------------------------------------

        print("  GEP...")

        gep = run_gep(
            gep_module=gep_module,
            gep_defaults=gep_defaults,
            mach=mach,
            alpha=alpha,
            cr_seed=cr_seed,
            ci_seed=ci_seed,
        )

        row.update(
            gep
        )

        if (
            gep[
                "gep_status"
            ]
            == "COMPLETED"
        ):
            row[
                "gep_spectral_error"
            ] = spectral_error(
                float(
                    gep["gep_cr"]
                ),
                float(
                    gep["gep_ci"]
                ),
                cr_ref,
                ci_ref,
            )

        else:
            row[
                "gep_spectral_error"
            ] = np.nan

        print(
            "    status="
            f"{row['gep_status']} "
            "err="
            f"{row['gep_spectral_error']:.6e}"
            if finite(
                row[
                    "gep_spectral_error"
                ]
            )
            else
            "    status=FAILED"
        )

        # ----------------------------------------------------------
        # Shooting
        # ----------------------------------------------------------

        print("  shooting multistart...")

        shooting = run_shooting(
            shooting_module=
                shooting_module,

            shooting_defaults=
                shooting_defaults,

            mach=mach,
            alpha=alpha,
            cr_seed=cr_seed,
            ci_seed=ci_seed,
        )

        row.update(
            shooting
        )

        if (
            shooting[
                "shoot_status"
            ]
            == "COMPLETED"
        ):
            row[
                "shoot_spectral_error"
            ] = spectral_error(
                float(
                    shooting[
                        "shoot_cr"
                    ]
                ),
                float(
                    shooting[
                        "shoot_ci"
                    ]
                ),
                cr_ref,
                ci_ref,
            )

        else:
            row[
                "shoot_spectral_error"
            ] = np.nan

        if finite(
            row[
                "shoot_spectral_error"
            ]
        ):
            print(
                "    success="
                f"{row['shoot_spectral_success']}"
                "/"
                f"{row['shoot_mode_success']} "
                "err="
                f"{row['shoot_spectral_error']:.6e} "
                "mismatch="
                f"{row['shoot_total_mismatch']:.3e}"
            )

        else:
            print(
                "    shooting FAILED"
            )

        # ----------------------------------------------------------
        # Powell
        # ----------------------------------------------------------

        print("  Powell...")

        powell = run_powell(
            solver_class=
                solver_class,

            shooting_defaults=
                shooting_defaults,

            powell_defaults=
                powell_defaults,

            mach=mach,
            alpha=alpha,
            cr_seed=cr_seed,
            ci_seed=ci_seed,
        )

        row.update(
            powell
        )

        if (
            powell[
                "powell_status"
            ]
            == "COMPLETED"
        ):
            row[
                "powell_spectral_error"
            ] = spectral_error(
                float(
                    powell[
                        "powell_cr"
                    ]
                ),
                float(
                    powell[
                        "powell_ci"
                    ]
                ),
                cr_ref,
                ci_ref,
            )

        else:
            row[
                "powell_spectral_error"
            ] = np.nan

        if finite(
            row[
                "powell_spectral_error"
            ]
        ):
            print(
                "    optimizer_success="
                f"{row['powell_optimizer_success']} "
                "err="
                f"{row['powell_spectral_error']:.6e} "
                "mismatch="
                f"{row['powell_corrected_mismatch']:.3e}"
            )

        else:
            print(
                "    Powell FAILED"
            )

        rows.append(
            row
        )

        # Save after EVERY validation point.
        #
        # If Slurm times out, --resume continues from this file.
        pd.DataFrame(
            rows
        ).sort_values(
            "benchmark_id"
        ).to_csv(
            checkpoint_path,
            index=False,
        )

    results = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "benchmark_id"
        )
        .reset_index(drop=True)
    )

    if len(results) == 0:
        raise RuntimeError(
            "No benchmark result."
        )

    summaries = []

    for prefix in [
        "gep",
        "shoot",
        "powell",
    ]:

        summaries.append(
            method_summary(
                results,
                prefix,
            )
        )

    summary = pd.DataFrame(
        summaries
    )

    summary_path = (
        OUTPUT_ROOT
        / "corrector_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    worst_cols = [
        "benchmark_id",
        "atlas_chart",
        "Mach",
        "alpha",
        "pinn_spectral_error",
        "gep_spectral_error",
        "shoot_spectral_error",
        "powell_spectral_error",
        "gep_status",
        "shoot_status",
        "shoot_spectral_success",
        "shoot_mode_success",
        "powell_optimizer_success",
    ]

    worst = (
        results[
            [
                c
                for c in worst_cols
                if c in results.columns
            ]
        ]
        .sort_values(
            "pinn_spectral_error",
            ascending=False,
        )
    )

    worst.to_csv(
        OUTPUT_ROOT
        / "corrector_worst_pinn_points.csv",
        index=False,
    )

    print()
    print("=" * 130)
    print(
        "FINAL CORRECTOR SUMMARY"
    )
    print("=" * 130)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6e}",
        )
    )

    print()
    print("=" * 130)
    print(
        "WORST PINN POINTS AFTER CORRECTION"
    )
    print("=" * 130)

    print(
        worst.head(20)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6e}",
        )
    )

    print()
    print("written:")
    print(checkpoint_path)
    print(summary_path)


if __name__ == "__main__":
    main()
