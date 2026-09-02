from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path.cwd()

ANCHORS = (
    REPO / 'assets/pinn_supersonic/csv/anchor_budget/N76/table_C00_anchors.csv'
)

VALIDATION = (
    REPO / 'assets/pinn_supersonic/csv/pinn_direct/validation/table_N76_validation_predictions_64_cf57c58769.csv'
)

BASE_SCRIPT = (
    REPO / 'code/scripts/benchmarks/benchmark_atlas2d_correctors_N76.py'
)

OUTPUT = (
    REPO / 'assets/pinn_supersonic/csv/atlas_12charts/corrector_benchmark/table_C00_anchor_continuation_M105_to_M110.csv'
)


ALPHA = 0.09

START_MACH = 1.05
TARGET_MACH = 1.10

MACH_STEP = 0.0025

# Fixed BEFORE looking at the validation solution.
# These are local continuation boxes around
# the PREVIOUS classical solution.
CR_HALF_WINDOW = 0.020
CI_HALF_WINDOW = 0.025


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
            f"Cannot load {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module

    spec.loader.exec_module(
        module
    )

    return module


def exact_row(
    frame: pd.DataFrame,
    mach: float,
    alpha: float,
) -> pd.Series:

    mask = (
        np.isclose(
            frame["Mach"],
            mach,
            rtol=0.0,
            atol=1e-12,
        )
        &
        np.isclose(
            frame["alpha"],
            alpha,
            rtol=0.0,
            atol=1e-12,
        )
    )

    selected = frame.loc[mask]

    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one row at "
            f"M={mach}, alpha={alpha}; "
            f"got {len(selected)}"
        )

    return selected.iloc[0]


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


# ----------------------------------------------------------------------
# Load repository solver exactly as in the validated benchmark.
# ----------------------------------------------------------------------

base = load_module(
    "anchor_cont_base",
    BASE_SCRIPT,
)

# Existing benchmark requires this import first
# for repository-specific classical_solver setup.
base.load_module(
    "anchor_cont_path_setup",
    base.COMPARE_GEP_SCRIPT,
)

shooting_module = base.load_module(
    "anchor_cont_shooting",
    base.SHOOTING_SCRIPT,
)

defaults = base.parser_defaults(
    shooting_module.build_parser()
)

Solver = (
    shooting_module
    .Mstab17SupersonicSolver
)


# ----------------------------------------------------------------------
# Start ONLY from a classical TRAINING anchor.
# ----------------------------------------------------------------------

anchors = pd.read_csv(
    ANCHORS
)

anchor = exact_row(
    anchors,
    START_MACH,
    ALPHA,
)

if not bool(
    anchor["is_anchor_N76"]
):
    raise RuntimeError(
        "Selected start point is not "
        "an N76 training anchor."
    )

if str(
    anchor["direction"]
).lower() != "high":
    raise RuntimeError(
        "Selected anchor is not labelled "
        "as high branch."
    )

seed_cr = float(
    anchor["cr"]
)

seed_ci = float(
    anchor["ci"]
)

print("=" * 100)
print("ANCHOR-GUIDED MACH CONTINUATION")
print("=" * 100)

print(
    "START TRAINING ANCHOR:"
)

print(
    f"M={START_MACH:.4f} "
    f"alpha={ALPHA:.4f} "
    f"c=({seed_cr:.10f}, "
    f"{seed_ci:.10f}) "
    f"direction={anchor['direction']}"
)


# ----------------------------------------------------------------------
# Deterministic Mach path.
# ----------------------------------------------------------------------

n_steps = int(
    round(
        (
            TARGET_MACH
            - START_MACH
        )
        / MACH_STEP
    )
)

mach_values = (
    START_MACH
    + MACH_STEP
    * np.arange(
        1,
        n_steps + 1,
    )
)

mach_values[-1] = (
    TARGET_MACH
)

rows = [
    {
        "step": 0,
        "Mach": START_MACH,
        "alpha": ALPHA,
        "cr": seed_cr,
        "ci": seed_ci,
        "stage1_mismatch": 0.0,
        "stage2_mismatch": 0.0,
        "spectral_success": True,
        "mode_success": bool(
            anchor.get(
                "mode_available",
                True,
            )
        ),
        "source": "N76_training_anchor",
    }
]


for step, mach in enumerate(
    mach_values,
    start=1,
):

    solver = Solver(
        alpha=ALPHA,
        Mach=float(mach),
        use_mapping=True,
        mapping_scale=5.0,
    )

    cr_min = max(
        -0.10,
        seed_cr - CR_HALF_WINDOW,
    )

    cr_max = min(
        0.65,
        seed_cr + CR_HALF_WINDOW,
    )

    ci_min = max(
        1.0e-6,
        seed_ci - CI_HALF_WINDOW,
    )

    ci_max = min(
        0.25,
        seed_ci + CI_HALF_WINDOW,
    )

    print()
    print(
        f"[{step:02d}/{n_steps:02d}] "
        f"M={mach:.4f}"
    )

    print(
        f"  incoming seed = "
        f"({seed_cr:.10f}, "
        f"{seed_ci:.10f})"
    )

    print(
        f"  box cr=[{cr_min:.6f},"
        f"{cr_max:.6f}] "
        f"ci=[{ci_min:.6f},"
        f"{ci_max:.6f}]"
    )

    result = solver.solve(
        cr_min=cr_min,
        cr_max=cr_max,
        ci_min=ci_min,
        ci_max=ci_max,
        max_iter=int(
            defaults[
                "max_iter"
            ]
        ),
        grid_size=int(
            defaults[
                "grid_size"
            ]
        ),
    )

    if not bool(
        result.spectral_success
    ):
        raise RuntimeError(
            f"Spectral continuation failed "
            f"at M={mach:.6f}"
        )

    new_cr = float(
        result.cr
    )

    new_ci = float(
        result.ci
    )

    jump = float(
        np.hypot(
            new_cr - seed_cr,
            new_ci - seed_ci,
        )
    )

    print(
        f"  result = "
        f"({new_cr:.10f}, "
        f"{new_ci:.10f})"
    )

    print(
        f"  jump={jump:.6e} "
        f"mismatch1="
        f"{float(result.stage1_mismatch):.3e} "
        f"mismatch2="
        f"{float(result.stage2_mismatch):.3e}"
    )

    rows.append(
        {
            "step": step,
            "Mach": float(mach),
            "alpha": ALPHA,
            "cr": new_cr,
            "ci": new_ci,
            "jump_from_previous": jump,
            "stage1_mismatch":
                float(
                    result.stage1_mismatch
                ),
            "stage2_mismatch":
                float(
                    result.stage2_mismatch
                ),
            "spectral_success":
                bool(
                    result.spectral_success
                ),
            "mode_success":
                bool(
                    result.mode_success
                ),
            "source":
                "classical_continuation",
        }
    )

    # Continuation uses ONLY previous
    # classical result.
    seed_cr = new_cr
    seed_ci = new_ci


result_frame = pd.DataFrame(
    rows
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result_frame.to_csv(
    OUTPUT,
    index=False,
)


# ----------------------------------------------------------------------
# Evaluation only AFTER the complete trajectory.
# Reference is not involved anywhere above.
# ----------------------------------------------------------------------

validation = pd.read_csv(
    VALIDATION
)

target = exact_row(
    validation,
    TARGET_MACH,
    ALPHA,
)

cr_ref = float(
    target["cr"]
)

ci_ref = float(
    target["ci"]
)

final_error = spectral_error(
    seed_cr,
    seed_ci,
    cr_ref,
    ci_ref,
)


print()
print("=" * 100)
print(
    "FINAL RESULT — REFERENCE-FREE "
    "CONTINUATION COMPLETE"
)
print("=" * 100)

print(
    f"candidate c = "
    f"({seed_cr:.10f}, "
    f"{seed_ci:.10f})"
)

print()
print(
    "EVALUATION ONLY:"
)

print(
    f"reference c = "
    f"({cr_ref:.10f}, "
    f"{ci_ref:.10f})"
)

print(
    f"spectral error = "
    f"{final_error:.9e}"
)

print()
print(
    "WRITTEN:",
    OUTPUT,
)
