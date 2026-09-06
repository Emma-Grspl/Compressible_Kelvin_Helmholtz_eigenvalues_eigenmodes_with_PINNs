#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

import src.scripts.evaluation.evaluate_joint_pinn_global_validation as V


ROOT = Path(__file__).resolve().parents[4]

ROUTING = ROOT / "configs/atlas/N340_chart_routing.csv"

MODEL_ROOT = (
    ROOT
    / "models_saved/production/atlas/N340"
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
    checkpoint = (
        MODEL_ROOT
        / str(chart_id)
        / "model_state.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing canonical N340 checkpoint for "
            f"{chart_id}: {checkpoint}"
        )
    return checkpoint


plan = pd.read_csv(ROUTING).copy()

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
