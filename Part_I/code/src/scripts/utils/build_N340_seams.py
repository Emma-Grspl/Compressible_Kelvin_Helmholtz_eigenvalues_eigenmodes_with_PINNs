#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

import src.scripts.evaluation.evaluate_joint_pinn_global_validation as V


ROOT = Path(__file__).resolve().parents[4]

BASE_PLAN = (
    ROOT
    / "archive/csv/assets/pinn_subsonic/"
    "joint_ci_mode_atlas_v2/training_plan.tsv"
)

RUN = (
    ROOT
    / "assets/pinn_subsonic/"
    "anchor_budget_runs/N340"
)

OUT = (
    ROOT
    / "assets/pinn_subsonic/"
    "article/N340"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def find_checkpoint(chart_id: str) -> Path:
    directory = (
        RUN
        / "joint"
        / str(chart_id)
    )

    candidates = [
        directory / "model_best.pt",
        directory / "model_state.pt",
    ]

    for p in candidates:
        if p.is_file():
            return p

    pts = sorted(directory.glob("*.pt"))

    if len(pts) == 1:
        return pts[0]

    raise FileNotFoundError(
        f"Cannot uniquely identify checkpoint "
        f"for {chart_id}: {pts}"
    )


plan = pd.read_csv(
    BASE_PLAN,
    sep="\t",
).copy()

plan["checkpoint"] = [
    str(find_checkpoint(chart_id))
    for chart_id in plan["chart_id"]
]

assert len(plan) == 49
assert all(
    Path(p).is_file()
    for p in plan["checkpoint"]
)

plan_csv = (
    OUT
    / "N340_training_plan_for_seams.csv"
)

plan.to_csv(
    plan_csv,
    index=False,
)

seams = V.build_seam_points(plan)

seam_csv = (
    OUT
    / "N340_seam_points.csv"
)

seams.to_csv(
    seam_csv,
    index=False,
)

print("charts:", len(plan))
print("seam points:", len(seams))
print("plan:", plan_csv)
print("seams:", seam_csv)

print("\nCheckpoint sample:")
print(
    plan[
        ["chart_id", "checkpoint"]
    ]
    .head()
    .to_string(index=False)
)
