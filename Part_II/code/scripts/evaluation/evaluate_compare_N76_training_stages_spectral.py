#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.migrated_paths import resolve_migrated_path


REPO = Path.cwd().resolve()

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TRAINER_PATH = (
    REPO / 'code/scripts/training/train_global_supersonic_kappa_q_logamp_continuousM.py'
)

CHECKPOINT_SOURCE_PREFIX = "assets/pinn_supersonic/atlas2d_v1_continuousM/N76/runs"

T401_PATH = (
    REPO / 'assets/classic_supersonic/csv/pinn_direct/shooting_T401/table_N76_T401_shooting_401.csv'
)

OUT_ROOT = REPO / "experiments/training_stage_comparison/N76"

CHARTS = [
    "C00", "C01", "C02",
    "C10", "C11", "C12",
    "C20", "C21", "C22",
    "C30", "C31", "C32",
]

STAGES = {
    "spectral_prefit":
        "best_spectral_prefit_checkpoint.pt",

    "modal_frozen_spectrum":
        "best_modal_frozen_spectrum_checkpoint.pt",

    "joint":
        "best_joint_checkpoint.pt",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module

    spec.loader.exec_module(
        module
    )

    return module


def stats(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    return {
        "n":
            int(len(x)),

        "mean":
            float(np.mean(x)),

        "median":
            float(np.median(x)),

        "p90":
            float(np.quantile(x, .90)),

        "p95":
            float(np.quantile(x, .95)),

        "max":
            float(np.max(x)),
    }


def predict_chart(
    trainer,
    checkpoint_path: Path,
    frame: pd.DataFrame,
    device: torch.device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        not isinstance(checkpoint, dict)
        or "model_state_dict"
        not in checkpoint
        or "config"
        not in checkpoint
    ):
        raise RuntimeError(
            f"Invalid checkpoint: {checkpoint_path}"
        )

    model = (
        trainer.build_model(
            checkpoint["config"]
        )
        .to(device)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    dtype = next(
        model.parameters()
    ).dtype

    alpha = torch.as_tensor(
        frame["alpha"]
        .to_numpy(float)
        .reshape(-1, 1),
        dtype=dtype,
        device=device,
    )

    mach = torch.as_tensor(
        frame["Mach"]
        .to_numpy(float)
        .reshape(-1, 1),
        dtype=dtype,
        device=device,
    )

    with torch.inference_mode():
        cr, ci = model.get_spectrum(
            alpha,
            mach,
        )

    cr = (
        cr.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    ci = (
        ci.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    del model
    del checkpoint

    return cr, ci


def metric_rows(
    frame: pd.DataFrame,
    *,
    stage: str,
    chart: str | None,
):
    rows = []

    for metric in [
        "cr_abs_error",
        "ci_abs_error",
        "spectral_error",
    ]:
        rows.append({
            "stage": stage,
            "chart":
                "GLOBAL"
                if chart is None
                else chart,
            "metric": metric,
            **stats(
                frame[metric]
            ),
        })

    return rows


def main():
    device = torch.device("cpu")

    print(
        "device =",
        device,
        flush=True,
    )

    trainer = load_module(
        TRAINER_PATH,
        "stage_comparison_trainer",
    )

    assert T401_PATH.is_file(), T401_PATH

    base = pd.read_csv(
        T401_PATH
    )

    assert len(base) == 401, len(base)

    needed = {
        "Mach",
        "alpha",
        "atlas_chart",
        "cr_reference",
        "ci_reference",
    }

    missing = needed - set(
        base.columns
    )

    if missing:
        raise RuntimeError(
            f"T401 missing columns: {sorted(missing)}"
        )

    predictions = []

    for stage, filename in STAGES.items():

        print()
        print("=" * 100)
        print("STAGE:", stage)
        print("=" * 100)

        for chart in CHARTS:

            sub = (
                base[
                    base[
                        "atlas_chart"
                    ]
                    .astype(str)
                    .eq(chart)
                ]
                .copy()
                .reset_index(drop=True)
            )

            if sub.empty:
                raise RuntimeError(
                    f"No T401 rows for {chart}"
                )

            checkpoint = resolve_migrated_path(
                REPO,
                f"{CHECKPOINT_SOURCE_PREFIX}/{chart}/{filename}",
            )

            if not checkpoint.is_file():
                raise FileNotFoundError(
                    checkpoint
                )

            cr, ci = predict_chart(
                trainer,
                checkpoint,
                sub,
                device,
            )

            sub["stage"] = stage
            sub["cr_stage"] = cr
            sub["ci_stage"] = ci

            sub["cr_abs_error"] = np.abs(
                sub["cr_stage"]
                - sub["cr_reference"]
            )

            sub["ci_abs_error"] = np.abs(
                sub["ci_stage"]
                - sub["ci_reference"]
            )

            sub["spectral_error"] = np.hypot(
                sub["cr_abs_error"],
                sub["ci_abs_error"],
            )

            predictions.append(
                sub
            )

            print(
                chart,
                "n=",
                len(sub),
                "median |dc|=",
                f"{sub['spectral_error'].median():.6e}",
                "p95=",
                f"{sub['spectral_error'].quantile(.95):.6e}",
                flush=True,
            )

    pred = pd.concat(
        predictions,
        ignore_index=True,
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_path = (
        OUT_ROOT
        / "Tab_supersonic_N76_training_stage_T401_predictions.csv"
    )

    pred.to_csv(
        pred_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Global + per-chart summaries
    # ------------------------------------------------------------

    summary_rows = []

    for stage in STAGES:
        s = pred[
            pred["stage"].eq(stage)
        ]

        summary_rows.extend(
            metric_rows(
                s,
                stage=stage,
                chart=None,
            )
        )

        for chart in CHARTS:
            g = s[
                s["atlas_chart"]
                .astype(str)
                .eq(chart)
            ]

            summary_rows.extend(
                metric_rows(
                    g,
                    stage=stage,
                    chart=chart,
                )
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUT_ROOT
        / "Tab_supersonic_N76_training_stage_T401_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Wide frame: one row per T401 point, one prediction per stage
    # ------------------------------------------------------------

    keys = [
        "Mach",
        "alpha",
        "atlas_chart",
        "cr_reference",
        "ci_reference",
    ]

    wide_parts = []

    for stage in STAGES:

        s = pred[
            pred["stage"].eq(stage)
        ][
            keys
            + [
                "cr_stage",
                "ci_stage",
                "cr_abs_error",
                "ci_abs_error",
                "spectral_error",
            ]
        ].copy()

        s = s.rename(
            columns={
                "cr_stage":
                    f"cr_{stage}",

                "ci_stage":
                    f"ci_{stage}",

                "cr_abs_error":
                    f"cr_error_{stage}",

                "ci_abs_error":
                    f"ci_error_{stage}",

                "spectral_error":
                    f"spectral_error_{stage}",
            }
        )

        wide_parts.append(s)

    wide = wide_parts[0]

    for part in wide_parts[1:]:
        wide = wide.merge(
            part,
            on=keys,
            how="inner",
            validate="one_to_one",
        )

    assert len(wide) == 401

    wide_path = (
        OUT_ROOT
        / "Tab_supersonic_N76_training_stage_T401_comparison.csv"
    )

    wide.to_csv(
        wide_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Stage-to-stage prediction movement
    # ------------------------------------------------------------

    comparisons = [
        (
            "spectral_prefit",
            "modal_frozen_spectrum",
        ),
        (
            "modal_frozen_spectrum",
            "joint",
        ),
        (
            "spectral_prefit",
            "joint",
        ),
    ]

    pair_rows = []

    for a, b in comparisons:

        dcr = np.abs(
            wide[f"cr_{b}"]
            - wide[f"cr_{a}"]
        )

        dci = np.abs(
            wide[f"ci_{b}"]
            - wide[f"ci_{a}"]
        )

        dc = np.hypot(
            dcr,
            dci,
        )

        for metric, values in [
            ("delta_cr", dcr),
            ("delta_ci", dci),
            ("delta_c", dc),
        ]:
            pair_rows.append({
                "from_stage": a,
                "to_stage": b,
                "metric": metric,
                **stats(values),
            })

    pairwise = pd.DataFrame(
        pair_rows
    )

    pairwise_path = (
        OUT_ROOT
        / "Tab_supersonic_N76_training_stage_prediction_changes.csv"
    )

    pairwise.to_csv(
        pairwise_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Prefit -> joint improvement diagnostics
    # ------------------------------------------------------------

    e0 = wide[
        "spectral_error_spectral_prefit"
    ].to_numpy(float)

    e1 = wide[
        "spectral_error_joint"
    ].to_numpy(float)

    improvement = e0 - e1

    better = (
        e1 < e0
    )

    worse = (
        e1 > e0
    )

    equal = np.isclose(
        e1,
        e0,
        rtol=0.0,
        atol=1e-12,
    )

    thresholds = [
        1e-3,
        5e-3,
        1e-2,
        2e-2,
        5e-2,
    ]

    threshold_rows = []

    for stage in STAGES:
        e = wide[
            f"spectral_error_{stage}"
        ].to_numpy(float)

        row = {
            "stage": stage,
        }

        for threshold in thresholds:
            row[
                f"n_le_{threshold:.0e}"
            ] = int(
                np.sum(
                    e <= threshold
                )
            )

        threshold_rows.append(
            row
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_path = (
        OUT_ROOT
        / "Tab_supersonic_N76_training_stage_threshold_counts.csv"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    print()
    print("=" * 100)
    print("GLOBAL T401 SPECTRAL METRICS")
    print("=" * 100)

    global_summary = summary[
        summary["chart"].eq(
            "GLOBAL"
        )
    ]

    print(
        global_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10e}",
        )
    )

    print()
    print("=" * 100)
    print("STAGE-TO-STAGE PREDICTION CHANGE")
    print("=" * 100)

    print(
        pairwise.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10e}",
        )
    )

    print()
    print("=" * 100)
    print("PREFIT -> JOINT POINTWISE EFFECT")
    print("=" * 100)

    print(
        "joint better:",
        int(np.sum(better)),
        "/ 401 =",
        f"{np.mean(better):.4%}",
    )

    print(
        "joint worse :",
        int(np.sum(worse)),
        "/ 401 =",
        f"{np.mean(worse):.4%}",
    )

    print(
        "numerically equal:",
        int(np.sum(equal)),
        "/ 401",
    )

    print(
        "median reduction in |dc| =",
        f"{np.median(improvement):.10e}",
    )

    print(
        "mean reduction in |dc|   =",
        f"{np.mean(improvement):.10e}",
    )

    print()
    print("=" * 100)
    print("SEED-PROXIMITY THRESHOLDS")
    print("=" * 100)

    print(
        threshold_df.to_string(
            index=False
        )
    )

    print()
    print("saved:")
    print(pred_path)
    print(summary_path)
    print(wide_path)
    print(pairwise_path)
    print(threshold_path)


if __name__ == "__main__":
    main()
