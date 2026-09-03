#!/usr/bin/env python3
"""Route one query to a production N340 neural-atlas chart without loading it.

Run from the Part_I directory with:
    python examples/02_query_neural_atlas.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PART_I_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_I_ROOT / "code"))

from src.scripts.gep.selection.solve_blumen_exact_joint_gep_v3 import (  # noqa: E402
    route_chart,
)


def main() -> None:
    mach = 0.5
    alpha = 0.5
    eta = alpha / (1.0 - mach * mach) ** 0.5
    routing = pd.read_csv(PART_I_ROOT / "configs/atlas/N340_chart_routing.csv")
    chart = route_chart(routing, mach=mach, eta=eta)
    checkpoint = (
        PART_I_ROOT
        / "models_saved/production/atlas/N340"
        / str(chart.chart_id)
        / "model_state.pt"
    )

    print(f"Mach={mach:.3f}, alpha={alpha:.3f}, eta={eta:.6f}")
    print(f"chart_id={chart.chart_id}, family={chart.family}")
    print(f"checkpoint={checkpoint.relative_to(PART_I_ROOT)}")
    print(f"checkpoint_exists={checkpoint.is_file()}")


if __name__ == "__main__":
    main()
