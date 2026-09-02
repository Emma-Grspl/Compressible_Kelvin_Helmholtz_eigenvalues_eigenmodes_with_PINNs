from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path.cwd()

PREDICTIONS = (
    REPO / 'assets/pinn_supersonic/csv/pinn_direct/validation/table_N76_validation_predictions_64_cf57c58769.csv'
)

OUTPUT_ROOT = (
    REPO
    / "assets/pinn_supersonic/"
      "atlas2d_v1_continuousM/N76/"
      "shooting_validation"
)

BASE_BENCHMARK = (
    REPO / 'code/scripts/benchmarks/benchmark_atlas2d_correctors_N76.py'
)


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


def finite(value: Any) -> bool:
    try:
        return bool(
            np.isfinite(float(value))
        )
    except Exception:
        return False


def write_progress(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    frame = pd.DataFrame(rows)

    if "benchmark_id" in frame.columns:
        frame = (
            frame
            .sort_values("benchmark_id")
            .reset_index(drop=True)
        )

    frame.to_csv(
        path,
        index=False,
    )


def build_summary(
    frame: pd.DataFrame,
) -> dict[str, Any]:

    pinn = pd.to_numeric(
        frame["pinn_spectral_error"],
        errors="coerce",
    )

    shoot = pd.to_numeric(
        frame["shoot_spectral_error"],
        errors="coerce",
    )

    valid = (
        shoot.notna()
        & np.isfinite(shoot)
    )

    e = shoot[valid]

    summary = {
        "n_total":
            int(len(frame)),

        "n_shoot_completed":
            int(
                frame["shoot_status"]
                .astype(str)
                .str.upper()
                .eq("COMPLETED")
                .sum()
            ),

        "n_spectral_success":
            int(
                frame[
                    "shoot_spectral_success"
                ]
                .astype(str)
                .str.lower()
                .eq("true")
                .sum()
            ),

        "n_mode_success":
            int(
                frame[
                    "shoot_mode_success"
                ]
                .astype(str)
                .str.lower()
                .eq("true")
                .sum()
            ),

        "spectral_error_mean":
            float(e.mean()),

        "spectral_error_median":
            float(e.median()),

        "spectral_error_p90":
            float(e.quantile(0.90)),

        "spectral_error_p95":
            float(e.quantile(0.95)),

        "spectral_error_p99":
            float(e.quantile(0.99)),

        "spectral_error_max":
            float(e.max()),

        "n_error_le_1e-6":
            int((e <= 1e-6).sum()),

        "n_error_le_1e-5":
            int((e <= 1e-5).sum()),

        "n_error_le_1e-4":
            int((e <= 1e-4).sum()),

        "n_error_le_1e-3":
            int((e <= 1e-3).sum()),

        "n_error_le_5e-3":
            int((e <= 5e-3).sum()),

        "n_error_le_1e-2":
            int((e <= 1e-2).sum()),

        "n_improved_vs_pinn":
            int(
                (
                    shoot[valid]
                    < pinn[valid]
                ).sum()
            ),

        "n_worsened_vs_pinn":
            int(
                (
                    shoot[valid]
                    > pinn[valid]
                ).sum()
            ),

        "mean_seconds":
            float(
                pd.to_numeric(
                    frame["shoot_seconds"],
                    errors="coerce",
                ).mean()
            ),

        "median_seconds":
            float(
                pd.to_numeric(
                    frame["shoot_seconds"],
                    errors="coerce",
                ).median()
            ),

        "total_seconds":
            float(
                pd.to_numeric(
                    frame["shoot_seconds"],
                    errors="coerce",
                ).sum()
            ),
    }

    return summary


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    if not BASE_BENCHMARK.is_file():
        raise FileNotFoundError(
            BASE_BENCHMARK
        )

    if not PREDICTIONS.is_file():
        raise FileNotFoundError(
            PREDICTIONS
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = (
        OUTPUT_ROOT
        / "shooting_validation_64.csv"
    )

    base = load_module(
        "atlas_corrector_base",
        BASE_BENCHMARK,
    )

    # IMPORTANT:
    # The historical benchmark loads the compare/GEP
    # module first because it performs the repository-
    # specific classical_solver path setup.
    #
    # We preserve that import side effect only.
    # NO GEP solver is evaluated.
    base.load_module(
        "atlas_gep_path_setup_only",
        base.COMPARE_GEP_SCRIPT,
    )

    shooting_module = base.load_module(
        "atlas_shooting_continuousM",
        base.SHOOTING_SCRIPT,
    )

    shooting_defaults = (
        base.parser_defaults(
            shooting_module.build_parser()
        )
    )

    settings = {
        "predictions":
            str(PREDICTIONS),

        "method":
            "PINN-seeded shooting only",

        "pinn_training":
            "atlas2d_v1 continuous-M N76",

        "reference_used_for_selection":
            False,

        "selection_policy":
            (
                "technical spectral+mode success; "
                "then minimum stage1+stage2 mismatch; "
                "PINN-seed displacement as tie-breaker"
            ),

        "shooting_defaults": {
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
    }

    (
        OUTPUT_ROOT
        / "shooting_settings.json"
    ).write_text(
        json.dumps(
            settings,
            indent=2,
            default=str,
        )
        + "\n"
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
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    if len(frame) < 1:
        raise RuntimeError(
            f"Expected at least one input point, "
            f"got {len(frame)}."
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

    frame["benchmark_id"] = (
        np.arange(len(frame))
    )

    n_target = (
        len(frame)
        if args.limit is None
        else min(
            len(frame),
            int(args.limit),
        )
    )

    existing: dict[
        int,
        dict[str, Any],
    ] = {}

    if (
        args.resume
        and output_csv.is_file()
    ):
        old = pd.read_csv(
            output_csv
        )

        for _, old_row in old.iterrows():
            existing[
                int(
                    old_row[
                        "benchmark_id"
                    ]
                )
            ] = old_row.to_dict()

        print(
            "Resume:",
            len(existing),
            "points already available.",
            flush=True,
        )

    rows: list[
        dict[str, Any]
    ] = []

    print(
        "=" * 110,
        flush=True,
    )

    print(
        "PINN-SEEDED SHOOTING — "
        "CONTINUOUS-M N76",
        flush=True,
    )

    print(
        "=" * 110,
        flush=True,
    )

    print(
        "points:",
        n_target,
        flush=True,
    )

    for idx in range(n_target):

        source = frame.iloc[idx]

        benchmark_id = int(
            source["benchmark_id"]
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
                "SKIP existing",
                flush=True,
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
            source["atlas_chart"]
        )

        pinn_error = spectral_error(
            cr_seed,
            ci_seed,
            cr_ref,
            ci_ref,
        )

        print()
        print(
            "=" * 110,
            flush=True,
        )

        print(
            f"[{idx+1:02d}/{n_target:02d}] "
            f"id={benchmark_id:02d} "
            f"{chart} "
            f"M={mach:.3f} "
            f"alpha={alpha:.3f}",
            flush=True,
        )

        print(
            f"PINN seed = "
            f"({cr_seed:.8f}, "
            f"{ci_seed:.8f})",
            flush=True,
        )

        print(
            f"PINN validation error = "
            f"{pinn_error:.6e}",
            flush=True,
        )

        row: dict[str, Any] = {
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

        shooting = (
            base.run_shooting(
                shooting_module=
                    shooting_module,

                shooting_defaults=
                    shooting_defaults,

                mach=mach,
                alpha=alpha,
                cr_seed=cr_seed,
                ci_seed=ci_seed,
            )
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
                "shooting: "
                f"success="
                f"{row['shoot_spectral_success']}"
                "/"
                f"{row['shoot_mode_success']} "
                f"error="
                f"{row['shoot_spectral_error']:.6e} "
                f"mismatch="
                f"{row['shoot_total_mismatch']:.3e} "
                f"time="
                f"{row['shoot_seconds']:.1f}s",
                flush=True,
            )
        else:
            print(
                "shooting FAILED:",
                row.get(
                    "shoot_error",
                    "",
                ),
                flush=True,
            )

        rows.append(
            row
        )

        write_progress(
            rows,
            output_csv,
        )

    result = pd.DataFrame(
        rows
    )

    result = (
        result
        .sort_values(
            "benchmark_id"
        )
        .reset_index(drop=True)
    )

    write_progress(
        rows,
        output_csv,
    )

    summary = build_summary(
        result
    )

    summary_path = (
        OUTPUT_ROOT
        / "shooting_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    # Worst points for rapid inspection.
    worst = (
        result
        .sort_values(
            "shoot_spectral_error",
            ascending=False,
        )
    )

    worst.to_csv(
        OUTPUT_ROOT
        / "shooting_worst_points.csv",
        index=False,
    )

    print()
    print(
        "=" * 110,
        flush=True,
    )

    print(
        "FINAL SUMMARY",
        flush=True,
    )

    print(
        "=" * 110,
        flush=True,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        ),
        flush=True,
    )

    print()
    print(
        "written:",
        output_csv,
        flush=True,
    )

    print(
        "written:",
        summary_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
