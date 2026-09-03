from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path.cwd()

VALIDATION = (
    REPO / 'assets/pinn_supersonic/csv/pinn_direct/validation/table_N76_validation_predictions_64_cf57c58769.csv'
)

SHOOTING_RESULTS = (
    REPO / 'assets/classic_supersonic/csv/pinn_direct/shooting_validation/table_shooting_validation_64.csv'
)

MODE_BANK = (
    REPO / 'experiments/modal_reconstruction/support/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/classical_supersonic_final_modes_long.csv.gz'
)

CHECKPOINT = (
    REPO
    / "models_saved/atlas/N76/C00/"
      "best_joint_checkpoint_33647c0c65.pt"
)

BASE_SCRIPT = (
    REPO / 'code/scripts/benchmarks/benchmark_atlas2d_correctors_N76.py'
)

VALIDATOR_SCRIPT = (
    REPO / 'code/scripts/evaluation/evaluate_validate_atlas2d_modal_continuousM_N76.py'
)

COMPARE_SCRIPT = (
    REPO / 'code/scripts/evaluation/evaluate_compare_supersonic_gep_candidates_vs_shooting.py'
)

OUTPUT = (
    REPO / 'assets/pinn_supersonic/csv/chart_overlap/modal_validation/table_C00_branch_fingerprint_diagnostic.csv'
)


MACH = 1.10
ALPHA_TARGET = 0.09


def load_module(name, path):
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

    spec.loader.exec_module(module)

    return module


def exact_row(frame, mach, alpha):
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
            f"Expected one row for "
            f"M={mach}, alpha={alpha}; "
            f"got {len(selected)}"
        )

    return selected.iloc[0]


def interp_complex(
    y_new,
    y_old,
    z_old,
):
    order = np.argsort(y_old)

    y_old = np.asarray(
        y_old,
        dtype=float,
    )[order]

    z_old = np.asarray(
        z_old,
        dtype=np.complex128,
    )[order]

    return (
        np.interp(
            y_new,
            y_old,
            np.real(z_old),
        )
        + 1j
        * np.interp(
            y_new,
            y_old,
            np.imag(z_old),
        )
    )


def overlap_on_core(
    y_pinn,
    p_pinn,
    y_candidate,
    p_candidate,
    ymax,
):
    lo = max(
        -float(ymax),
        float(np.min(y_pinn)),
        float(np.min(y_candidate)),
    )

    hi = min(
        float(ymax),
        float(np.max(y_pinn)),
        float(np.max(y_candidate)),
    )

    if hi <= lo:
        return float("nan")

    y = np.linspace(
        lo,
        hi,
        1601,
    )

    p1 = interp_complex(
        y,
        y_pinn,
        p_pinn,
    )

    p2 = interp_complex(
        y,
        y_candidate,
        p_candidate,
    )

    denom = (
        np.linalg.norm(p1)
        * np.linalg.norm(p2)
    )

    if denom <= 1e-14:
        return float("nan")

    return float(
        abs(
            np.vdot(
                p2,
                p1,
            )
        )
        / denom
    )


def spectral_error(
    cr,
    ci,
    cr_ref,
    ci_ref,
):
    return float(
        np.hypot(
            cr - cr_ref,
            ci - ci_ref,
        )
    )


# ----------------------------------------------------------------------
# Load existing modules
# ----------------------------------------------------------------------

base = load_module(
    "branch_base",
    BASE_SCRIPT,
)

# Required repository-specific classical path setup.
base.load_module(
    "branch_gep_path_setup",
    base.COMPARE_GEP_SCRIPT,
)

shoot_module = base.load_module(
    "branch_shoot_module",
    base.SHOOTING_SCRIPT,
)

shoot_defaults = (
    base.parser_defaults(
        shoot_module.build_parser()
    )
)

validator = load_module(
    "branch_modal_validator",
    VALIDATOR_SCRIPT,
)

trainer = validator.load_module(
    "branch_continuousM_trainer",
    validator.TRAINER,
)

audit = validator.load_module(
    "branch_modal_audit",
    validator.AUDIT,
)

compare = load_module(
    "branch_mode_extract",
    COMPARE_SCRIPT,
)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

validation = pd.read_csv(
    VALIDATION
)

shooting = pd.read_csv(
    SHOOTING_RESULTS
)

target_v = exact_row(
    validation,
    MACH,
    ALPHA_TARGET,
)

target_s = exact_row(
    shooting,
    MACH,
    ALPHA_TARGET,
)

start_v = exact_row(
    validation,
    MACH,
    0.105,
)

cr_ref = float(
    target_v["cr"]
)

ci_ref = float(
    target_v["ci"]
)

direct_cr = float(
    target_s["shoot_cr"]
)

direct_ci = float(
    target_s["shoot_ci"]
)


print("=" * 100)
print("DIRECT CANDIDATE")
print("=" * 100)

print(
    f"c_direct = "
    f"{direct_cr:.10f} "
    f"+ i {direct_ci:.10f}"
)


# ----------------------------------------------------------------------
# Reference-free local continuation
#
# Start only from the PINN prediction at alpha=.105.
# Then classical continuation toward .09.
# ----------------------------------------------------------------------

seed_cr = float(
    start_v["cr_pred"]
)

seed_ci = float(
    start_v["ci_pred"]
)

continuation_rows = []

for alpha in [
    0.105,
    0.100,
    0.095,
    0.090,
]:

    print()
    print(
        f"Continuation alpha={alpha:.3f}"
    )

    print(
        f"  incoming seed = "
        f"({seed_cr:.9f}, "
        f"{seed_ci:.9f})"
    )

    result = base.run_shooting(
        shooting_module=
            shoot_module,

        shooting_defaults=
            shoot_defaults,

        mach=MACH,
        alpha=float(alpha),

        cr_seed=float(seed_cr),
        ci_seed=float(seed_ci),
    )

    if (
        result["shoot_status"]
        != "COMPLETED"
    ):
        raise RuntimeError(
            f"Shooting failed at "
            f"alpha={alpha}: "
            f"{result.get('shoot_error', '')}"
        )

    seed_cr = float(
        result["shoot_cr"]
    )

    seed_ci = float(
        result["shoot_ci"]
    )

    continuation_rows.append(
        {
            "alpha":
                alpha,

            "cr":
                seed_cr,

            "ci":
                seed_ci,

            "mismatch":
                float(
                    result[
                        "shoot_total_mismatch"
                    ]
                ),

            "spectral_success":
                bool(
                    result[
                        "shoot_spectral_success"
                    ]
                ),

            "mode_success":
                bool(
                    result[
                        "shoot_mode_success"
                    ]
                ),
        }
    )

    print(
        f"  corrected = "
        f"({seed_cr:.9f}, "
        f"{seed_ci:.9f}) "
        f"mismatch="
        f"{result['shoot_total_mismatch']:.3e}"
    )


continuation_cr = seed_cr
continuation_ci = seed_ci

print()
print("=" * 100)
print("FINAL CONTINUATION CANDIDATE")
print("=" * 100)

print(
    f"c_cont = "
    f"{continuation_cr:.10f} "
    f"+ i {continuation_ci:.10f}"
)


# ----------------------------------------------------------------------
# Reconstruct PINN pressure profile at target
# ----------------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

config = checkpoint.get(
    "config"
)

if not isinstance(
    config,
    dict,
):
    raise RuntimeError(
        "Checkpoint has no config."
    )

model = trainer.build_model(
    config
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

if hasattr(
    model,
    "set_mach_context",
):
    model.set_mach_context(
        MACH
    )


bank = pd.read_csv(
    MODE_BANK,
    usecols=[
        "Mach",
        "alpha",
        "coordinate_index",
        "y",
        "p_real",
        "p_imag",
    ],
)

mask = (
    np.isclose(
        bank["Mach"],
        MACH,
        rtol=0.0,
        atol=1e-12,
    )
    &
    np.isclose(
        bank["alpha"],
        ALPHA_TARGET,
        rtol=0.0,
        atol=1e-12,
    )
)

reference_mode = (
    bank.loc[mask]
    .sort_values(
        "coordinate_index"
    )
    .reset_index(drop=True)
)

if len(reference_mode) < 100:
    raise RuntimeError(
        "Reference y-grid not found."
    )

y_pinn = (
    reference_mode[
        "y"
    ]
    .to_numpy(
        dtype=float
    )
)

order = np.argsort(
    y_pinn
)

y_pinn = y_pinn[order]

dtype = next(
    model.parameters()
).dtype

y_tensor = torch.tensor(
    y_pinn[:, None],
    dtype=dtype,
)

alpha_tensor = torch.full_like(
    y_tensor,
    ALPHA_TARGET,
)

mach_tensor = torch.full_like(
    y_tensor,
    MACH,
)

scale = validator.mapping_scale(
    model,
    config,
    mach_tensor,
)

xi_tensor = audit.y_to_xi(
    y_tensor,
    scale,
)

with torch.no_grad():

    modal = validator.model_forward(
        model,
        xi_tensor,
        alpha_tensor,
        mach_tensor,
    )

modal = (
    modal.detach()
    .cpu()
    .numpy()
)

q_pinn = modal[:, 1]
log_amp_pinn = modal[:, 2]

center_index = int(
    np.argmin(
        np.abs(y_pinn)
    )
)

phase_pinn = (
    audit.integrate_phase(
        y_pinn,
        q_pinn,
        center_index,
    )
)

p_pinn = (
    np.exp(
        np.clip(
            log_amp_pinn,
            -50.0,
            20.0,
        )
    )
    * np.exp(
        1j * phase_pinn
    )
)


# ----------------------------------------------------------------------
# Reconstruct BOTH shooting candidate modes.
#
# Tiny boxes ensure we reconstruct the requested root,
# rather than performing a new broad branch search.
# ----------------------------------------------------------------------

def extract_candidate(
    cr,
    ci,
):
    return (
        compare.extract_shooting_mode(
            alpha=ALPHA_TARGET,
            mach=MACH,

            target_cr=float(cr),
            target_ci=float(ci),

            cr_window=0.003,
            ci_window=0.003,

            max_iter=int(
                shoot_defaults[
                    "max_iter"
                ]
            ),

            grid_size=int(
                shoot_defaults[
                    "grid_size"
                ]
            ),
        )
    )


print()
print(
    "Reconstructing direct mode..."
)

direct_mode = extract_candidate(
    direct_cr,
    direct_ci,
)

print(
    "Reconstructing continuation mode..."
)

continuation_mode = (
    extract_candidate(
        continuation_cr,
        continuation_ci,
    )
)


# ----------------------------------------------------------------------
# PINN-only fingerprint
# ----------------------------------------------------------------------

rows = []

for name, candidate in [
    (
        "direct",
        direct_mode,
    ),
    (
        "continuation",
        continuation_mode,
    ),
]:

    row = {
        "candidate":
            name,

        "cr":
            float(
                candidate["cr"]
            ),

        "ci":
            float(
                candidate["ci"]
            ),
    }

    for ymax in [
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
    ]:

        row[
            f"pinn_overlap_core{int(ymax)}"
        ] = overlap_on_core(
            y_pinn,
            p_pinn,

            np.asarray(
                candidate["y"],
                dtype=float,
            ),

            np.asarray(
                candidate["p"],
                dtype=np.complex128,
            ),

            ymax,
        )

    # Evaluation only.
    # These values are NEVER used for selection.
    row[
        "spectral_error_vs_reference_EVAL_ONLY"
    ] = spectral_error(
        row["cr"],
        row["ci"],
        cr_ref,
        ci_ref,
    )

    rows.append(row)


result = pd.DataFrame(
    rows
)

# Primary selector:
# PINN overlap in the core |y| <= 20.
#
# NO classical reference appears in this selection.
winner_index = (
    result[
        "pinn_overlap_core20"
    ]
    .idxmax()
)

winner = result.loc[
    winner_index
]

result[
    "selected_by_pinn_core20"
] = False

result.loc[
    winner_index,
    "selected_by_pinn_core20",
] = True

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT,
    index=False,
)


print()
print("=" * 120)
print("PINN MODAL FINGERPRINT")
print("=" * 120)

print(
    result.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.9e}",
    )
)

print()
print("=" * 120)
print("REFERENCE-FREE SELECTION")
print("=" * 120)

print(
    "selected candidate:",
    winner["candidate"],
)

print(
    "selected c =",
    f"({winner['cr']:.10f}, "
    f"{winner['ci']:.10f})",
)

margin = abs(
    float(
        result.loc[
            result[
                "candidate"
            ].eq(
                "continuation"
            ),
            "pinn_overlap_core20",
        ].iloc[0]
    )
    -
    float(
        result.loc[
            result[
                "candidate"
            ].eq(
                "direct"
            ),
            "pinn_overlap_core20",
        ].iloc[0]
    )
)

print(
    "core20 overlap margin =",
    f"{margin:.6f}",
)

print()
print("=" * 120)
print("EVALUATION AGAINST VALIDATION REFERENCE — NOT USED FOR SELECTION")
print("=" * 120)

print(
    "reference c =",
    f"({cr_ref:.10f}, "
    f"{ci_ref:.10f})",
)

print(
    "selected spectral error =",
    f"{winner['spectral_error_vs_reference_EVAL_ONLY']:.9e}",
)

print()
print("WRITTEN:", OUTPUT)
