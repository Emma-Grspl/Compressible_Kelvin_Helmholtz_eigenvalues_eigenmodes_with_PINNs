#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from classical_solver.gep.dense_gep_notebook_style import (
    NotebookStyleDenseGEPSolver,
)


REFERENCE_PATH = Path('assets/classic_supersonic/csv/modal_reconstruction/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/table_classical_supersonic_final_reference.csv')

OUTPUT_PATH = Path(
    "assets/pinn_supersonic/hybrid_gep/"
    "local_M150_S4M0_v1/classic_seeded/"
    "nearest_candidate_resolution_audit.csv"
)

MACH = 1.50

ANCHORS = [
    0.070,
    0.140,
    0.205,
    0.250,
]

RESOLUTIONS = [
    201,
    301,
    401,
]


reference = pd.read_csv(
    REFERENCE_PATH
)

reference = reference[
    np.isclose(
        reference["Mach"],
        MACH,
        rtol=0.0,
        atol=1.0e-12,
    )
]

rows = []

for n_points in RESOLUTIONS:
    for alpha in ANCHORS:
        match = reference[
            np.isclose(
                reference["alpha"],
                alpha,
                rtol=0.0,
                atol=1.0e-12,
            )
        ]

        if len(match) != 1:
            raise RuntimeError(
                f"Reference mismatch for alpha={alpha}"
            )

        match = match.iloc[0]

        target = (
            float(match["cr"]),
            float(match["ci"]),
        )

        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=MACH,
            n_points=n_points,
            mapping_kind="pin",
            mapping_scale=3.0,
            xi_max=0.985,
        )

        modes = solver.finite_modes()

        ranked = sorted(
            modes,
            key=lambda mode: (
                solver.spectral_distance(
                    mode,
                    target,
                    ci_weight=2.0,
                )
            ),
        )

        print()
        print("=" * 90)
        print(
            f"N={n_points}, alpha={alpha:.6f}, "
            f"target={target}, modes={len(modes)}"
        )
        print("=" * 90)

        for rank, mode in enumerate(
            ranked[:5],
            start=1,
        ):
            ordinary_distance = float(
                np.hypot(
                    mode["cr"] - target[0],
                    mode["ci"] - target[1],
                )
            )

            weighted_distance = (
                solver.spectral_distance(
                    mode,
                    target,
                    ci_weight=2.0,
                )
            )

            print(
                f"{rank:2d} "
                f"c=({mode['cr']:.9f},"
                f"{mode['ci']:.9f}) "
                f"d={ordinary_distance:.6e} "
                f"dw={weighted_distance:.6e}"
            )

            rows.append(
                {
                    "n_points": n_points,
                    "Mach": MACH,
                    "alpha": alpha,
                    "rank": rank,
                    "cr_reference": target[0],
                    "ci_reference": target[1],
                    "cr_gep": float(
                        mode["cr"]
                    ),
                    "ci_gep": float(
                        mode["ci"]
                    ),
                    "ordinary_distance": (
                        ordinary_distance
                    ),
                    "weighted_distance": (
                        weighted_distance
                    ),
                    "n_finite_modes": len(
                        modes
                    ),
                }
            )

output = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output.to_csv(
    OUTPUT_PATH,
    index=False,
)

print()
print("written:", OUTPUT_PATH)
