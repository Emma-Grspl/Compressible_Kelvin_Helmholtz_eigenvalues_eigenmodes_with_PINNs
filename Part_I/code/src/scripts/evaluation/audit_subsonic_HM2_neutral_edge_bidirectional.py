from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from src.scripts.gep.selection.solve_dense_gep_notebook_style import (
    NotebookStyleDenseGEPSolver,
)
from src.scripts.evaluation.evaluate_kh_subsonic_fixed_mach_modal_candidates import (
    load_classic_full_mode,
)

MACH = 0.97
N = 401
MAPPING_SCALE = 5.0
XI_MAX = 0.98

ETAS_FORWARD = [0.9700, 0.9725, 0.9750, 0.9775, 0.9800]
ETAS_REVERSE = list(reversed(ETAS_FORWARD))

OUTDIR = Path(
    "assets/pinn_subsonic/local_atlas_v1/"
    "audit_HM2_neutral_edge_bidirectional_M097"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


def nearest_mode(modes, target_ci):
    return min(
        modes,
        key=lambda m: np.sqrt(
            float(m["cr"]) ** 2
            + (2.0 * (float(m["ci"]) - target_ci)) ** 2
        ),
    )


def follow_branch(etas, direction):
    rows = []

    previous_guess = None
    previous_signature = None

    for i, eta in enumerate(etas):
        alpha = float(eta * np.sqrt(1.0 - MACH**2))
        _, ci_classic = load_classic_full_mode(alpha, MACH)

        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=MACH,
            n_points=N,
            mapping_kind="pin",
            mapping_scale=MAPPING_SCALE,
            xi_max=XI_MAX,
        )

        modes = solver.finite_modes()
        if not modes:
            raise RuntimeError(f"No modes at eta={eta}")

        if i == 0:
            # Les deux extrémités eta=0.970 et eta=0.980 sont cohérentes
            # avec la branche classique.
            selected = nearest_mode(modes, float(ci_classic))
            source = "endpoint_classic_initialization"
        else:
            selected, source, _ = solver.get_branch_mode(
                target_guess=previous_guess,
                previous_guess=previous_guess,
                previous_signature=previous_signature,
                prefer_positive_cr=False,
                ci_weight=2.0,
                spectral_window_factor=3.0,
                spectral_window_floor=0.015,
                overlap_top_k=20,
                overlap_weight=0.75,
                jump_cr_weight=0.25,
                jump_ci_weight=0.75,
            )

            if selected is None:
                raise RuntimeError(
                    f"Branch continuation failed at eta={eta}"
                )

        previous_guess = (
            float(selected["cr"]),
            float(selected["ci"]),
        )
        previous_signature = np.asarray(
            selected["signature"],
            dtype=float,
        )

        mode_near_classic = nearest_mode(modes, float(ci_classic))

        rows.append(
            {
                "direction": direction,
                "Mach": MACH,
                "eta": eta,
                "alpha": alpha,
                "ci_classic": float(ci_classic),
                "ci_continuation": float(selected["ci"]),
                "cr_continuation": float(selected["cr"]),
                "continuation_source": source,
                "ci_nearest_classic": float(mode_near_classic["ci"]),
                "continuation_abs_err_vs_classic": abs(
                    float(selected["ci"]) - float(ci_classic)
                ),
                "n_finite_modes": len(modes),
            }
        )

    return rows


rows = []
rows += follow_branch(ETAS_FORWARD, "forward_0970_to_0980")
rows += follow_branch(ETAS_REVERSE, "reverse_0980_to_0970")

df = pd.DataFrame(rows)

out = OUTDIR / "summary_bidirectional.csv"
df.to_csv(out, index=False)

print(df.sort_values(["eta", "direction"]).to_string(index=False))

pivot = df.pivot(
    index="eta",
    columns="direction",
    values="ci_continuation",
).reset_index()

pivot["forward_reverse_abs_diff"] = abs(
    pivot["forward_0970_to_0980"]
    - pivot["reverse_0980_to_0970"]
)

pivot_out = OUTDIR / "forward_reverse_comparison.csv"
pivot.to_csv(pivot_out, index=False)

print("\n===== FORWARD / REVERSE =====")
print(pivot.to_string(index=False))

print("\nSummary:", out)
print("Comparison:", pivot_out)
