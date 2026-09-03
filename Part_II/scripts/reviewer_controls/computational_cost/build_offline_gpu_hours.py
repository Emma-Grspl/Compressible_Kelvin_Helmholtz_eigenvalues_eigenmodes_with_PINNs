#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(
    "assets/pinn_supersonic/"
    "atlas2d_v1_continuousM_fullc_v1/N76/"
    "article_compute_cost"
)

RAW = ROOT / "training_jobs_sacct_raw.psv"

CHARTS = [
    "C00", "C01", "C02",
    "C10", "C11", "C12",
    "C20", "C21", "C22",
    "C30", "C31", "C32",
]


def main():
    if not RAW.is_file():
        raise FileNotFoundError(RAW)

    df = pd.read_csv(
        RAW,
        sep="|",
        header=None,
        names=[
            "JobID",
            "JobIDRaw",
            "JobName",
            "State",
            "ElapsedRaw",
            "AllocTRES",
            "Start",
            "End",
        ],
    )

    df["JobID"] = df["JobID"].astype(str)

    # Keep array tasks only.
    tasks = df[
        df["JobID"].str.match(r"^\d+_\d+$")
    ].copy()

    if len(tasks) != 12:
        raise RuntimeError(
            f"Expected 12 array tasks, found {len(tasks)}"
        )

    tasks["array_task_id"] = (
        tasks["JobID"]
        .str.extract(r"_(\d+)$")[0]
        .astype(int)
    )

    tasks = (
        tasks
        .sort_values("array_task_id")
        .reset_index(drop=True)
    )

    if tasks["array_task_id"].tolist() != list(range(12)):
        raise RuntimeError(
            "Expected array task IDs 0..11"
        )

    if not tasks["State"].eq("COMPLETED").all():
        raise RuntimeError(
            "At least one training task is not COMPLETED"
        )

    tasks["chart"] = CHARTS

    # Production SLURM requested exactly one GPU per task.
    tasks["n_gpus"] = 1

    tasks["elapsed_seconds"] = pd.to_numeric(
        tasks["ElapsedRaw"],
        errors="raise",
    ).astype(int)

    tasks["gpu_hours"] = (
        tasks["elapsed_seconds"]
        * tasks["n_gpus"]
        / 3600.0
    )

    tasks["training_stage"] = (
        "prefit2000+modal4000+joint4000"
    )

    tasks["included_in_article_total"] = True

    clean_cols = [
        "chart",
        "array_task_id",
        "JobID",
        "JobIDRaw",
        "JobName",
        "State",
        "elapsed_seconds",
        "n_gpus",
        "gpu_hours",
        "Start",
        "End",
        "training_stage",
        "included_in_article_total",
    ]

    clean = tasks[clean_cols].copy()

    total_seconds = int(
        clean["elapsed_seconds"].sum()
    )

    total_gpu_hours = float(
        clean["gpu_hours"].sum()
    )

    summary = {
        "model": "N76 FULLC production atlas",
        "n_charts": 12,
        "slurm_array_job_id": "1188003",
        "gpu_type": "NVIDIA V100",
        "gpus_per_chart": 1,
        "prefit_steps": 2000,
        "modal_steps": 4000,
        "joint_steps": 4000,
        "total_training_steps_per_chart": 10000,
        "total_gpu_seconds": total_seconds,
        "total_gpu_hours": total_gpu_hours,
        "mean_seconds_per_chart": float(
            clean["elapsed_seconds"].mean()
        ),
        "median_seconds_per_chart": float(
            clean["elapsed_seconds"].median()
        ),
        "min_seconds_per_chart": int(
            clean["elapsed_seconds"].min()
        ),
        "max_seconds_per_chart": int(
            clean["elapsed_seconds"].max()
        ),
        "scope": (
            "Retained 12-chart N76 FULLC production training only; "
            "post-training classical shooting, validation, "
            "anchor-budget experiments and reviewer ablations excluded."
        ),
    }

    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_csv(
        ROOT / "training_jobs_clean.csv",
        index=False,
    )

    (
        ROOT / "gpu_hours_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print("=" * 80)
    print("PHASE 10 — OFFLINE GPU-HOURS")
    print("=" * 80)
    print(
        clean[
            [
                "chart",
                "elapsed_seconds",
                "gpu_hours",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        f"total GPU seconds = {total_seconds}"
    )
    print(
        f"total GPU hours   = {total_gpu_hours:.6f}"
    )
    print(
        f"mean/chart        = "
        f"{summary['mean_seconds_per_chart']:.2f} s"
    )
    print(
        f"median/chart      = "
        f"{summary['median_seconds_per_chart']:.2f} s"
    )
    print(
        f"range/chart       = "
        f"{summary['min_seconds_per_chart']}–"
        f"{summary['max_seconds_per_chart']} s"
    )

    if abs(total_gpu_hours - 2.1441666666666666) > 1e-10:
        raise RuntimeError(
            "Unexpected total GPU-hours"
        )

    print()
    print("PHASE 10: PASS")
    print(
        "WROTE:",
        ROOT / "training_jobs_clean.csv",
    )
    print(
        "WROTE:",
        ROOT / "gpu_hours_summary.json",
    )


if __name__ == "__main__":
    main()
