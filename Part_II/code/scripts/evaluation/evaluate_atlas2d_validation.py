from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

from utils.migrated_paths import resolve_migrated_path

import numpy as np
import pandas as pd
import torch


REPO = Path.cwd()

TRAINER_PATH = (
    REPO / 'code/scripts/training/train_global_supersonic_kappa_q_logamp.py'
)

VALIDATION_FILE = (
    REPO / 'assets/pinn_supersonic/csv/pinn_direct/atlas2d_v1/table_validation_reference_64.csv'
)

CHECKPOINT_SOURCE_PREFIX = "assets/pinn_supersonic/atlas2d_v1"

OUTPUT_ROOT = REPO / "experiments/pinn_direct/validation"

CHARTS = [
    "C00", "C01", "C02",
    "C10", "C11", "C12",
    "C20", "C21", "C22",
    "C30", "C31", "C32",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--budget",
        type=int,
        default=76,
    )

    p.add_argument(
        "--device",
        default="cpu",
    )

    return p.parse_args()


def load_trainer_module():
    spec = importlib.util.spec_from_file_location(
        "atlas_global_trainer",
        TRAINER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load trainer: {TRAINER_PATH}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def call_spectral_audit(
    fn,
    *,
    frame: pd.DataFrame,
    model,
    device: torch.device,
):
    """
    Call the trainer's own spectral_audit() without
    hard-coding its exact argument order.

    Required parameters are resolved semantically
    from their names.
    """

    sig = inspect.signature(fn)

    kwargs: dict[str, Any] = {}

    for name, param in sig.parameters.items():

        lname = name.lower()

        if (
            lname in {
                "frame",
                "df",
                "data",
                "dataset",
            }
            or "frame" in lname
        ):
            kwargs[name] = frame
            continue

        if lname == "model" or lname.endswith("_model"):
            kwargs[name] = model
            continue

        if lname == "device":
            kwargs[name] = device
            continue

        if lname in {
            "include_test",
            "audit_test",
            "expose_test",
            "allow_test",
        }:
            kwargs[name] = False
            continue

        if param.default is not inspect.Parameter.empty:
            continue

        raise RuntimeError(
            "Cannot automatically resolve required "
            f"spectral_audit parameter {name!r}. "
            f"Signature = {sig}"
        )

    print(
        "spectral_audit signature:",
        sig,
    )

    print(
        "resolved arguments:",
        sorted(kwargs),
    )

    return fn(
        **kwargs
    )


def metric_summary(
    df: pd.DataFrame,
) -> dict[str, Any]:

    err = df[
        "spectral_error"
    ].to_numpy(float)

    cr_err = df[
        "cr_abs_error"
    ].to_numpy(float)

    ci_err = df[
        "ci_abs_error"
    ].to_numpy(float)

    omega_err = df[
        "omega_i_abs_error"
    ].to_numpy(float)

    rel = df[
        "spectral_relative_error"
    ].to_numpy(float)

    return {
        "n": int(len(df)),

        "cr_mae":
            float(np.mean(cr_err)),

        "cr_median_abs":
            float(np.median(cr_err)),

        "cr_max_abs":
            float(np.max(cr_err)),

        "ci_mae":
            float(np.mean(ci_err)),

        "ci_median_abs":
            float(np.median(ci_err)),

        "ci_max_abs":
            float(np.max(ci_err)),

        "omega_i_mae":
            float(np.mean(omega_err)),

        "spectral_error_mean":
            float(np.mean(err)),

        "spectral_error_median":
            float(np.median(err)),

        "spectral_error_p90":
            float(np.quantile(err, 0.90)),

        "spectral_error_p95":
            float(np.quantile(err, 0.95)),

        "spectral_error_p99":
            float(np.quantile(err, 0.99)),

        "spectral_error_max":
            float(np.max(err)),

        "spectral_relative_mean":
            float(np.mean(rel)),

        "spectral_relative_p95":
            float(np.quantile(rel, 0.95)),

        "spectral_relative_max":
            float(np.max(rel)),

        "fraction_error_le_1e-4":
            float(np.mean(err <= 1e-4)),

        "fraction_error_le_5e-4":
            float(np.mean(err <= 5e-4)),

        "fraction_error_le_1e-3":
            float(np.mean(err <= 1e-3)),

        "fraction_error_le_2e-3":
            float(np.mean(err <= 2e-3)),

        "fraction_error_le_5e-3":
            float(np.mean(err <= 5e-3)),

        "fraction_error_le_1e-2":
            float(np.mean(err <= 1e-2)),
    }


def main() -> None:

    args = parse_args()

    budget = int(
        args.budget
    )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    out_root = (
        OUTPUT_ROOT
        / f"N{budget}"
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not VALIDATION_FILE.is_file():
        raise FileNotFoundError(
            VALIDATION_FILE
        )

    validation = pd.read_csv(
        VALIDATION_FILE
    )

    required = {
        "Mach",
        "alpha",
        "cr",
        "ci",
        "primary_chart",
        "point_role",
    }

    missing = (
        required
        - set(validation.columns)
    )

    if missing:
        raise RuntimeError(
            "Validation file missing columns: "
            f"{sorted(missing)}"
        )

    if len(validation) != 64:
        raise RuntimeError(
            "Expected 64 validation points, "
            f"got {len(validation)}."
        )

    if not validation[
        "point_role"
    ].eq(
        "validation"
    ).all():
        raise RuntimeError(
            "Non-validation rows found in "
            "validation_reference_64.csv."
        )

    observed_charts = set(
        validation[
            "primary_chart"
        ].astype(str)
    )

    if observed_charts != set(CHARTS):
        raise RuntimeError(
            "Primary-chart mismatch. "
            f"Observed={sorted(observed_charts)}"
        )

    trainer = load_trainer_module()

    if not hasattr(
        trainer,
        "build_model",
    ):
        raise RuntimeError(
            "Trainer has no build_model()."
        )

    if not hasattr(
        trainer,
        "spectral_audit",
    ):
        raise RuntimeError(
            "Trainer has no spectral_audit()."
        )

    all_predictions = []

    print()
    print("=" * 100)
    print(
        f"ATLAS-2D VALIDATION — N{budget}"
    )
    print("=" * 100)
    print(
        "validation points:",
        len(validation),
    )
    print(
        "device:",
        device,
    )

    for chart in CHARTS:

        sub = (
            validation[
                validation[
                    "primary_chart"
                ].astype(str).eq(chart)
            ]
            .copy()
            .reset_index(drop=True)
        )

        if sub.empty:
            raise RuntimeError(
                f"No validation point for {chart}."
            )

        checkpoint_path = resolve_migrated_path(
            REPO,
            f"{CHECKPOINT_SOURCE_PREFIX}/N{budget}/runs/"
            f"{chart}/best_joint_checkpoint.pt",
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                checkpoint_path
            )

        print()
        print("-" * 100)
        print(
            chart,
            "| validation points =",
            len(sub),
        )
        print(
            "checkpoint =",
            checkpoint_path,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        if (
            not isinstance(checkpoint, dict)
            or "model_state_dict"
               not in checkpoint
        ):
            raise RuntimeError(
                f"{chart}: invalid checkpoint."
            )

        config = checkpoint.get(
            "config"
        )

        if not isinstance(
            config,
            dict,
        ):
            raise RuntimeError(
                f"{chart}: checkpoint "
                "does not contain config."
            )

        model = (
            trainer.build_model(
                config
            )
            .to(device)
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        model.eval()

        # Ensure the trainer's historical split
        # machinery sees these rows as validation,
        # never as training or sealed test.
        local_frame = sub.copy()

        local_frame[
            "mach_split"
        ] = "validation"

        if (
            "usable_as_training_anchor"
            in local_frame.columns
        ):
            local_frame[
                "usable_as_training_anchor"
            ] = False

        with torch.no_grad():

            audit_output = (
                call_spectral_audit(
                    trainer.spectral_audit,
                    frame=local_frame,
                    model=model,
                    device=device,
                )
            )

        if not (
            isinstance(
                audit_output,
                tuple,
            )
            and len(audit_output) >= 1
        ):
            raise RuntimeError(
                f"{chart}: unexpected "
                "spectral_audit return type: "
                f"{type(audit_output)}"
            )

        pred = audit_output[0]

        if not isinstance(
            pred,
            pd.DataFrame,
        ):
            raise RuntimeError(
                f"{chart}: spectral_audit "
                "did not return DataFrame."
            )

        needed = {
            "Mach",
            "alpha",
            "cr",
            "ci",
            "cr_pred",
            "ci_pred",
        }

        missing_pred = (
            needed
            - set(pred.columns)
        )

        if missing_pred:
            raise RuntimeError(
                f"{chart}: missing prediction "
                f"columns {sorted(missing_pred)}"
            )

        if len(pred) != len(sub):
            raise RuntimeError(
                f"{chart}: expected {len(sub)} "
                f"predictions, got {len(pred)}."
            )

        pred = pred.copy()

        pred[
            "atlas_chart"
        ] = chart

        pred[
            "atlas_budget"
        ] = budget

        all_predictions.append(
            pred
        )

        print(
            f"{chart}: OK — "
            f"{len(pred)} predictions"
        )

        del model
        del checkpoint

    result = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # -------------------------------------------------------------
    # Safety
    # -------------------------------------------------------------

    if len(result) != 64:
        raise RuntimeError(
            f"Expected 64 final predictions, "
            f"got {len(result)}."
        )

    key_count = (
        result[
            ["Mach", "alpha"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    if key_count != 64:
        raise RuntimeError(
            "Duplicate validation coordinates "
            "after atlas inference."
        )

    # -------------------------------------------------------------
    # Recompute metrics ourselves.
    # Do not depend on trainer metric structure.
    # -------------------------------------------------------------

    result[
        "cr_abs_error"
    ] = np.abs(
        result["cr_pred"].to_numpy(float)
        - result["cr"].to_numpy(float)
    )

    result[
        "ci_abs_error"
    ] = np.abs(
        result["ci_pred"].to_numpy(float)
        - result["ci"].to_numpy(float)
    )

    result[
        "spectral_error"
    ] = np.hypot(
        result[
            "cr_pred"
        ].to_numpy(float)
        - result[
            "cr"
        ].to_numpy(float),

        result[
            "ci_pred"
        ].to_numpy(float)
        - result[
            "ci"
        ].to_numpy(float),
    )

    reference_norm = np.hypot(
        result[
            "cr"
        ].to_numpy(float),
        result[
            "ci"
        ].to_numpy(float),
    )

    result[
        "spectral_relative_error"
    ] = (
        result[
            "spectral_error"
        ].to_numpy(float)
        / np.maximum(
            reference_norm,
            1e-12,
        )
    )

    result[
        "omega_i_ref"
    ] = (
        result["alpha"].to_numpy(float)
        * result["ci"].to_numpy(float)
    )

    result[
        "omega_i_pred"
    ] = (
        result["alpha"].to_numpy(float)
        * result["ci_pred"].to_numpy(float)
    )

    result[
        "omega_i_abs_error"
    ] = np.abs(
        result[
            "omega_i_pred"
        ].to_numpy(float)
        - result[
            "omega_i_ref"
        ].to_numpy(float)
    )

    result = result.sort_values(
        [
            "atlas_chart",
            "Mach",
            "alpha",
        ]
    ).reset_index(drop=True)

    # -------------------------------------------------------------
    # Global metrics
    # -------------------------------------------------------------

    global_metrics = (
        metric_summary(
            result
        )
    )

    global_metrics[
        "budget"
    ] = budget

    global_metrics[
        "n_charts"
    ] = len(CHARTS)

    # -------------------------------------------------------------
    # Metrics per primary chart
    # -------------------------------------------------------------

    chart_rows = []

    for chart, group in result.groupby(
        "atlas_chart",
        sort=True,
    ):

        row = {
            "chart": chart,
            **metric_summary(group),
        }

        chart_rows.append(
            row
        )

    by_chart = pd.DataFrame(
        chart_rows
    )

    # -------------------------------------------------------------
    # Worst validation points
    # -------------------------------------------------------------

    worst = (
        result.sort_values(
            "spectral_error",
            ascending=False,
        )
        .head(20)
        [
            [
                "Mach",
                "alpha",
                "atlas_chart",
                "cr",
                "ci",
                "cr_pred",
                "ci_pred",
                "cr_abs_error",
                "ci_abs_error",
                "spectral_error",
                "spectral_relative_error",
                "omega_i_abs_error",
            ]
        ]
        .copy()
    )

    # -------------------------------------------------------------
    # Write
    # -------------------------------------------------------------

    pred_path = (
        out_root
        / f"N{budget}_validation_predictions_64.csv"
    )

    global_path = (
        out_root
        / f"N{budget}_validation_metrics_global.json"
    )

    chart_path = (
        out_root
        / f"N{budget}_validation_metrics_by_chart.csv"
    )

    worst_path = (
        out_root
        / f"N{budget}_validation_worst_points.csv"
    )

    result.to_csv(
        pred_path,
        index=False,
    )

    with global_path.open(
        "w"
    ) as f:
        json.dump(
            global_metrics,
            f,
            indent=2,
        )

    by_chart.to_csv(
        chart_path,
        index=False,
    )

    worst.to_csv(
        worst_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Human-readable output
    # -------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        f"ATLAS-N{budget} — "
        "64-POINT VALIDATION"
    )
    print("=" * 100)

    print(
        json.dumps(
            global_metrics,
            indent=2,
        )
    )

    print()
    print("=" * 130)
    print("BY PRIMARY CHART")
    print("=" * 130)

    show_cols = [
        "chart",
        "n",
        "cr_mae",
        "ci_mae",
        "spectral_error_mean",
        "spectral_error_p95",
        "spectral_error_max",
        "fraction_error_le_1e-3",
        "fraction_error_le_5e-3",
        "fraction_error_le_1e-2",
    ]

    print(
        by_chart[
            show_cols
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6e}",
        )
    )

    print()
    print("=" * 130)
    print("20 WORST VALIDATION POINTS")
    print("=" * 130)

    print(
        worst.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8e}",
        )
    )

    print()
    print("=" * 100)
    print("WRITTEN")
    print("=" * 100)

    for p in [
        pred_path,
        global_path,
        chart_path,
        worst_path,
    ]:
        print(p)


if __name__ == "__main__":
    main()
