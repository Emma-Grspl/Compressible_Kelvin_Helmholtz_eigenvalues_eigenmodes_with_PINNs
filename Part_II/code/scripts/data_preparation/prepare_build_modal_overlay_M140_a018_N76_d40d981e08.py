#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import torch

from classic_supersonic_reference.solver.mstab17_supersonic_solver import (
    Mstab17SupersonicSolver,
)


# =============================================================================
# Plot quality
# =============================================================================

# Critical for oscillatory eigenmodes: do not simplify long paths.
mpl.rcParams["path.simplify"] = False
mpl.rcParams["agg.path.chunksize"] = 0
mpl.rcParams["savefig.dpi"] = 500


# =============================================================================
# Configuration
# =============================================================================

REPO = Path.cwd().resolve()

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MACH = 1.40
ALPHA = 0.18
CHART = "C11"

PLOT_YMAX = 250.0
CORE_YMAX = 40.0

# 500-wide domain -> dy = 0.0125
N_Y = 40001

DEVICE = torch.device("cpu")


TRAINER_PATH = (
    REPO / 'code/scripts/training/train_global_supersonic_kappa_q_logamp_continuousM.py'
)

CHECKPOINT = (
    REPO
    / "models_saved/atlas/N76/"
      f"{CHART}/best_joint_checkpoint_33647c0c65.pt"
)

T401 = (
    REPO / 'assets/classic_supersonic/csv/pinn_direct/shooting_T401/table_N76_T401_shooting_401.csv'
)

OUT = (
    REPO
    / "assets/p3-supersonic-results"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Helpers
# =============================================================================

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
    spec.loader.exec_module(module)

    return module


def interp_complex_cubic(
    y_src: np.ndarray,
    f_src: np.ndarray,
    y_dst: np.ndarray,
):
    y_src = np.asarray(
        y_src,
        float,
    )

    f_src = np.asarray(
        f_src,
        complex,
    )

    order = np.argsort(
        y_src
    )

    y_src = y_src[
        order
    ]

    f_src = f_src[
        order
    ]

    # Remove any duplicated coordinate.
    y_src, unique_idx = np.unique(
        y_src,
        return_index=True,
    )

    f_src = f_src[
        unique_idx
    ]

    re = CubicSpline(
        y_src,
        f_src.real,
        extrapolate=False,
    )(
        y_dst
    )

    im = CubicSpline(
        y_src,
        f_src.imag,
        extrapolate=False,
    )(
        y_dst
    )

    return (
        re
        + 1j * im
    )


def reconstruct_primitives(
    *,
    y: np.ndarray,
    p: np.ndarray,
    gamma: np.ndarray,
    c: complex,
):
    U = np.tanh(
        y
    )

    Up = (
        1.0
        - U**2
    )

    py = (
        gamma
        * p
    )

    denom = (
        1j
        * ALPHA
        * (U - c)
    )

    v = (
        -py
        / denom
    )

    u = (
        -(
            Up * v
            + 1j * ALPHA * p
        )
        / denom
    )

    rho = (
        MACH**2
        * p
    )

    return {
        "p": p,
        "rho": rho,
        "u": u,
        "v": v,
    }


def complex_alignment(
    pred,
    ref,
    mask,
):
    a = pred[
        mask
    ]

    b = ref[
        mask
    ]

    den = np.vdot(
        a,
        a,
    )

    if abs(den) < 1.0e-30:
        raise RuntimeError(
            "Degenerate modal alignment."
        )

    return (
        np.vdot(
            a,
            b,
        )
        / den
    )


def apply_factor(
    fields,
    factor,
):
    return {
        key:
            factor * value
        for key, value
        in fields.items()
    }


def relative_l2(
    pred,
    ref,
    mask,
):
    return float(
        np.linalg.norm(
            pred[mask]
            - ref[mask]
        )
        / max(
            np.linalg.norm(
                ref[mask]
            ),
            1.0e-30,
        )
    )


def overlap(
    pred,
    ref,
    mask,
):
    a = pred[
        mask
    ]

    b = ref[
        mask
    ]

    den = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if den <= 1.0e-30:
        return np.nan

    return float(
        abs(
            np.vdot(
                a,
                b,
            )
        )
        / den
    )


def integrate_phase(
    y,
    q,
):
    y = np.asarray(
        y,
        float,
    )

    q = np.asarray(
        q,
        float,
    )

    phi = np.zeros_like(
        y
    )

    center = int(
        np.argmin(
            np.abs(y)
        )
    )

    for i in range(
        center + 1,
        len(y),
    ):
        dy = (
            y[i]
            - y[i - 1]
        )

        phi[i] = (
            phi[i - 1]
            + 0.5
            * (
                q[i]
                + q[i - 1]
            )
            * dy
        )

    for i in range(
        center - 1,
        -1,
        -1,
    ):
        dy = (
            y[i + 1]
            - y[i]
        )

        phi[i] = (
            phi[i + 1]
            - 0.5
            * (
                q[i]
                + q[i + 1]
            )
            * dy
        )

    return phi


def y_to_xi(
    y,
    scale,
):
    y = np.asarray(
        y,
        float,
    )

    return (
        2.0 * y
        / (
            scale
            + np.sqrt(
                scale**2
                + 4.0 * y**2
            )
        )
    )


# =============================================================================
# Frozen T401 row
# =============================================================================

def load_T401_row():

    frame = pd.read_csv(
        T401
    )

    sub = frame[
        np.isclose(
            frame["Mach"],
            MACH,
            atol=1e-12,
        )
        & np.isclose(
            frame["alpha"],
            ALPHA,
            atol=1e-12,
        )
    ]

    if len(sub) != 1:
        raise RuntimeError(
            f"Expected one row, found {len(sub)}."
        )

    row = sub.iloc[0]

    if str(
        row["atlas_chart"]
    ) != CHART:
        raise RuntimeError(
            f"Unexpected primary chart: "
            f"{row['atlas_chart']}"
        )

    return row


# =============================================================================
# Exact classical reconstruction for any frozen c
# =============================================================================

def reconstruct_shooting_at_c(
    y_common,
    *,
    cr,
    ci,
):
    solver = Mstab17SupersonicSolver(
        alpha=ALPHA,
        Mach=MACH,
        match_y=1.0,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.5,
        min_y_limit=PLOT_YMAX,
        max_y_limit=PLOT_YMAX,
        y_limit_factor=4.0,
        use_mapping=False,
    )

    ln_right = (
        solver.exact_right_log_amplitude(
            cr,
            ci,
        )
    )

    (
        sol_left,
        _,
        sol_right,
        y_limit,
    ) = solver.get_trajectories(
        cr,
        ci,
        ln_p_start_right=
            ln_right,
    )

    if not (
        sol_left.success
        and sol_right.success
    ):
        raise RuntimeError(
            "Classical trajectory reconstruction failed."
        )

    y_left = np.asarray(
        sol_left.t,
        float,
    )

    y_right = np.asarray(
        sol_right.t,
        float,
    )

    (
        k_left,
        q_left,
        ln_left,
        phi_left,
    ) = sol_left.y

    (
        k_right,
        q_right,
        ln_right_array,
        phi_right,
    ) = sol_right.y

    phi_left_0 = (
        solver._interp_component(
            0.0,
            sol_left,
            3,
        )
    )

    phi_right_0 = (
        solver._interp_component(
            0.0,
            sol_right,
            3,
        )
    )

    phase_shift = (
        phi_left_0
        - phi_right_0
    )

    p_left = (
        np.exp(ln_left)
        * np.exp(
            1j * phi_left
        )
    )

    p_right = (
        np.exp(
            ln_right_array
        )
        * np.exp(
            1j
            * (
                phi_right
                + phase_shift
            )
        )
    )

    gamma_left = (
        k_left
        + 1j * q_left
    )

    gamma_right = (
        k_right
        + 1j * q_right
    )

    left_mask = (
        y_left < 0.0
    )

    y_raw = np.concatenate(
        [
            y_left[
                left_mask
            ],
            y_right[::-1],
        ]
    )

    p_raw = np.concatenate(
        [
            p_left[
                left_mask
            ],
            p_right[::-1],
        ]
    )

    gamma_raw = np.concatenate(
        [
            gamma_left[
                left_mask
            ],
            gamma_right[::-1],
        ]
    )

    p = interp_complex_cubic(
        y_raw,
        p_raw,
        y_common,
    )

    gamma = interp_complex_cubic(
        y_raw,
        gamma_raw,
        y_common,
    )

    fields = reconstruct_primitives(
        y=y_common,
        p=p,
        gamma=gamma,
        c=complex(
            cr,
            ci,
        ),
    )

    return (
        fields,
        complex(
            cr,
            ci,
        ),
        float(y_limit),
    )


# =============================================================================
# Direct PINN + analytic far-field continuation
# =============================================================================

def load_direct_PINN(
    y_common,
):
    trainer = load_module(
        TRAINER_PATH,
        "modal_overlay_trainer",
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model = (
        trainer.build_model(
            checkpoint[
                "config"
            ]
        )
        .to(DEVICE)
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

    config = checkpoint[
        "config"
    ]

    scale = float(
        config[
            "model"
        ][
            "mapping_scale"
        ]
    )

    xi_max = float(
        config[
            "model"
        ][
            "xi_max"
        ]
    )

    mapped_ymax = float(
        scale
        * xi_max
        / (
            1.0
            - xi_max**2
        )
    )

    # Stay slightly inside the formal mapped boundary.
    neural_ymax = (
        0.995
        * mapped_ymax
    )

    inside = (
        np.abs(
            y_common
        )
        <= neural_ymax
    )

    y_neural = y_common[
        inside
    ]

    xi = y_to_xi(
        y_neural,
        scale,
    )

    xi_t = torch.as_tensor(
        xi.reshape(-1, 1),
        dtype=dtype,
        device=DEVICE,
    )

    alpha_t = torch.full_like(
        xi_t,
        ALPHA,
    )

    mach_t = torch.full_like(
        xi_t,
        MACH,
    )

    alpha_one = torch.tensor(
        [[ALPHA]],
        dtype=dtype,
        device=DEVICE,
    )

    mach_one = torch.tensor(
        [[MACH]],
        dtype=dtype,
        device=DEVICE,
    )

    with torch.inference_mode():

        cr_t, ci_t = (
            model.get_spectrum(
                alpha_one,
                mach_one,
            )
        )

        try:
            modal = model(
                xi_t,
                alpha_t,
                mach_t,
            )

        except TypeError:
            model.set_mach_context(
                MACH
            )

            modal = model(
                xi_t,
                alpha_t,
            )

    cr = float(
        cr_t.reshape(-1)[0]
        .detach()
        .cpu()
    )

    ci = float(
        ci_t.reshape(-1)[0]
        .detach()
        .cpu()
    )

    c = complex(
        cr,
        ci,
    )

    modal = (
        modal.detach()
        .cpu()
        .numpy()
    )

    kappa = modal[
        :,
        0,
    ]

    q = modal[
        :,
        1,
    ]

    log_amp = modal[
        :,
        2,
    ]

    phase = integrate_phase(
        y_neural,
        q,
    )

    p_neural = np.exp(
        log_amp
        + 1j * phase
    )

    gamma_neural = (
        kappa
        + 1j * q
    )

    # Full-domain arrays.
    p = np.empty(
        len(y_common),
        dtype=complex,
    )

    gamma = np.empty(
        len(y_common),
        dtype=complex,
    )

    p[
        inside
    ] = p_neural

    gamma[
        inside
    ] = gamma_neural

    # Physical asymptotic continuation outside neural mapping.
    tail_solver = Mstab17SupersonicSolver(
        alpha=ALPHA,
        Mach=MACH,
    )

    (
        gamma_left_inf,
        gamma_right_inf,
    ) = tail_solver.asymptotic_gammas(
        cr,
        ci,
    )

    neural_indices = np.flatnonzero(
        inside
    )

    i_left = int(
        neural_indices[0]
    )

    i_right = int(
        neural_indices[-1]
    )

    y_left = float(
        y_common[
            i_left
        ]
    )

    y_right = float(
        y_common[
            i_right
        ]
    )

    p_left = p[
        i_left
    ]

    p_right = p[
        i_right
    ]

    left_tail = (
        y_common
        < y_left
    )

    right_tail = (
        y_common
        > y_right
    )

    p[
        left_tail
    ] = (
        p_left
        * np.exp(
            gamma_left_inf
            * (
                y_common[
                    left_tail
                ]
                - y_left
            )
        )
    )

    gamma[
        left_tail
    ] = gamma_left_inf

    p[
        right_tail
    ] = (
        p_right
        * np.exp(
            gamma_right_inf
            * (
                y_common[
                    right_tail
                ]
                - y_right
            )
        )
    )

    gamma[
        right_tail
    ] = gamma_right_inf

    fields = reconstruct_primitives(
        y=y_common,
        p=p,
        gamma=gamma,
        c=c,
    )

    return (
        fields,
        c,
        {
            "mapping_scale":
                scale,

            "xi_max":
                xi_max,

            "mapped_ymax":
                mapped_ymax,

            "neural_ymax":
                neural_ymax,
        },
    )


# =============================================================================
# Plot
# =============================================================================

def make_figure(
    *,
    y,
    reference,
    shooting,
    pinn,
    component,
    suffix,
    c_ref,
    c_pinn,
    c_shoot,
    neural_ymax,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            15.5,
            10.2,
        ),
        sharex=True,
    )

    axes = axes.ravel()

    panels = [
        (
            r"(a) $\hat p$",
            "p",
        ),
        (
            r"(b) $\hat\rho$",
            "rho",
        ),
        (
            r"(c) $\hat u$",
            "u",
        ),
        (
            r"(d) $\hat v$",
            "v",
        ),
    ]

    getter = (
        np.real
        if component == "real"
        else np.imag
    )

    for ax, (
        title,
        field,
    ) in zip(
        axes,
        panels,
    ):
        # Blue first: because it almost overlaps black.
        ax.plot(
            y,
            getter(
                shooting[
                    field
                ]
            ),
            color="#1673b1",
            lw=1.8,
            linestyle=(0, (5, 2.3)),
            label="PINN-seeded shooting",
            antialiased=True,
        )

        # Direct PINN.
        ax.plot(
            y,
            getter(
                pinn[
                    field
                ]
            ),
            color="#d62728",
            lw=1.7,
            linestyle="-",
            label="Direct PINN",
            antialiased=True,
        )

        # Classical reference on top.
        ax.plot(
            y,
            getter(
                reference[
                    field
                ]
            ),
            color="black",
            lw=2.0,
            linestyle="-",
            label="Classical reference",
            antialiased=True,
        )

        # Boundary between neural prediction and analytic PINN tail.
        ax.axvline(
            -neural_ymax,
            color="0.70",
            lw=0.7,
            linestyle=":",
            zorder=0,
        )

        ax.axvline(
            neural_ymax,
            color="0.70",
            lw=0.7,
            linestyle=":",
            zorder=0,
        )

        ax.axvline(
            0.0,
            color="0.55",
            lw=0.7,
            linestyle=":",
            zorder=0,
        )

        ax.set_xlim(
            -PLOT_YMAX,
            PLOT_YMAX,
        )

        ax.set_title(
            title,
            loc="left",
            fontsize=15,
        )

        ax.set_ylabel(
            "Amplitude",
            fontsize=13,
        )

        ax.grid(
            True,
            alpha=0.18,
        )

    axes[2].set_xlabel(
        r"Transverse coordinate $y$",
        fontsize=13,
    )

    axes[3].set_xlabel(
        r"Transverse coordinate $y$",
        fontsize=13,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.2,
            label="Classical reference",
        ),
        Line2D(
            [0],
            [0],
            color="#1673b1",
            lw=2.0,
            linestyle=(0, (5, 2.3)),
            label="PINN-seeded shooting",
        ),
        Line2D(
            [0],
            [0],
            color="#d62728",
            lw=1.8,
            label="Direct PINN",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(
            0.5,
            0.925,
        ),
        frameon=False,
        fontsize=12,
    )

    component_label = (
        "Real parts"
        if component == "real"
        else "Imaginary parts"
    )

    fig.suptitle(
        (
            rf"{component_label} of the representative supersonic mode: "
            rf"$M={MACH:.1f}$, $\alpha={ALPHA:.2f}$"
            "\n"
            rf"$c^\star={c_ref.real:.6f}+{c_ref.imag:.6f}i$, "
            rf"$c_{{\rm PINN}}={c_pinn.real:.6f}+{c_pinn.imag:.6f}i$, "
            rf"$c_{{\rm shoot}}={c_shoot.real:.6f}+{c_shoot.imag:.6f}i$"
        ),
        fontsize=16,
        y=0.995,
    )

    fig.text(
        0.5,
        0.008,
        (
            "Light dotted vertical lines mark the direct-PINN mapped-domain "
            "boundary; outside it, only the analytical far-field continuation is shown."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.35",
    )

    fig.tight_layout(
        rect=[
            0.02,
            0.035,
            0.98,
            0.88,
        ]
    )

    stem = (
        OUT
        / f"Fig_supersonic_modal_overlay_M140_a018_{suffix}"
    )

    fig.savefig(
        stem.with_suffix(
            ".png"
        ),
        dpi=500,
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(
            ".svg"
        ),
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return stem


# =============================================================================
# Main
# =============================================================================

def main():

    print(
        "=" * 100
    )

    print(
        "HIGH-RESOLUTION MODAL OVERLAYS"
    )

    print(
        "=" * 100
    )

    row = load_T401_row()

    c_ref_target = complex(
        float(
            row[
                "cr_reference"
            ]
        ),
        float(
            row[
                "ci_reference"
            ]
        ),
    )

    c_shoot_target = complex(
        float(
            row[
                "shoot_cr"
            ]
        ),
        float(
            row[
                "shoot_ci"
            ]
        ),
    )

    y = np.linspace(
        -PLOT_YMAX,
        PLOT_YMAX,
        N_Y,
    )

    core40 = (
        np.abs(y)
        <= CORE_YMAX
    )

    print()
    print(
        "dense grid:",
        len(y),
        "points"
    )

    print(
        "dy =",
        y[1] - y[0],
    )

    # -------------------------------------------------------------------------
    # Classical reference reconstructed at the frozen reference c
    # -------------------------------------------------------------------------

    reference_raw, c_ref, ylimit_ref = (
        reconstruct_shooting_at_c(
            y,
            cr=c_ref_target.real,
            ci=c_ref_target.imag,
        )
    )

    # Normalize reference with max |p| = 1.
    pmax = float(
        np.max(
            np.abs(
                reference_raw[
                    "p"
                ]
            )
        )
    )

    reference = {
        field:
            value / pmax
        for field, value
        in reference_raw.items()
    }

    # -------------------------------------------------------------------------
    # Direct PINN
    # -------------------------------------------------------------------------

    pinn_raw, c_pinn, pinn_info = (
        load_direct_PINN(
            y
        )
    )

    frozen_pinn = complex(
        float(
            row[
                "cr_pinn"
            ]
        ),
        float(
            row[
                "ci_pinn"
            ]
        ),
    )

    checkpoint_error = abs(
        c_pinn
        - frozen_pinn
    )

    print()
    print(
        "PINN native mapped ymax =",
        pinn_info[
            "mapped_ymax"
        ],
    )

    print(
        "PINN neural plotting ymax =",
        pinn_info[
            "neural_ymax"
        ],
    )

    print(
        "checkpoint vs frozen T401 |dc| =",
        checkpoint_error,
    )

    if checkpoint_error > 1.0e-6:
        raise RuntimeError(
            "Wrong checkpoint: does not reproduce "
            "frozen T401 prediction."
        )

    # -------------------------------------------------------------------------
    # Corrected shooting
    # -------------------------------------------------------------------------

    shooting_raw, c_shoot, ylimit_shoot = (
        reconstruct_shooting_at_c(
            y,
            cr=c_shoot_target.real,
            ci=c_shoot_target.imag,
        )
    )

    # -------------------------------------------------------------------------
    # One alignment factor per method, determined from pressure over |y|<=40.
    # -------------------------------------------------------------------------

    A_pinn = complex_alignment(
        pinn_raw[
            "p"
        ],
        reference[
            "p"
        ],
        core40,
    )

    A_shoot = complex_alignment(
        shooting_raw[
            "p"
        ],
        reference[
            "p"
        ],
        core40,
    )

    pinn = apply_factor(
        pinn_raw,
        A_pinn,
    )

    shooting = apply_factor(
        shooting_raw,
        A_shoot,
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    rows = []

    for method, fields in [
        (
            "Direct PINN",
            pinn,
        ),
        (
            "PINN-seeded shooting",
            shooting,
        ),
    ]:
        for field in [
            "p",
            "rho",
            "u",
            "v",
        ]:
            rows.append(
                {
                    "method":
                        method,

                    "field":
                        field,

                    "rel_l2_core40":
                        relative_l2(
                            fields[
                                field
                            ],
                            reference[
                                field
                            ],
                            core40,
                        ),

                    "overlap_core40":
                        overlap(
                            fields[
                                field
                            ],
                            reference[
                                field
                            ],
                            core40,
                        ),
                }
            )

    metrics = pd.DataFrame(
        rows
    )

    print()
    print(
        "=" * 100
    )

    print(
        "MODAL METRICS — |y| <= 40"
    )

    print(
        "=" * 100
    )

    print(
        metrics.to_string(
            index=False,
            float_format=
                lambda x:
                    f"{x:.8e}",
        )
    )

    # -------------------------------------------------------------------------
    # Dense data
    # -------------------------------------------------------------------------

    data = pd.DataFrame(
        {
            "y":
                y,
        }
    )

    for field in [
        "p",
        "rho",
        "u",
        "v",
    ]:
        for prefix, fields in [
            (
                "classical",
                reference,
            ),
            (
                "pinn_seeded_shooting",
                shooting,
            ),
            (
                "direct_pinn",
                pinn,
            ),
        ]:
            data[
                f"{prefix}_{field}_real"
            ] = fields[
                field
            ].real

            data[
                f"{prefix}_{field}_imag"
            ] = fields[
                field
            ].imag

    data_path = (
        OUT
        / "Fig_supersonic_modal_overlay_M140_a018_dense_data.csv"
    )

    metrics_path = (
        OUT
        / "Tab_supersonic_modal_overlay_M140_a018_metrics.csv"
    )

    data.to_csv(
        data_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Two clean assets
    # -------------------------------------------------------------------------

    real_stem = make_figure(
        y=y,
        reference=reference,
        shooting=shooting,
        pinn=pinn,
        component="real",
        suffix="real",
        c_ref=c_ref,
        c_pinn=c_pinn,
        c_shoot=c_shoot,
        neural_ymax=
            pinn_info[
                "neural_ymax"
            ],
    )

    imag_stem = make_figure(
        y=y,
        reference=reference,
        shooting=shooting,
        pinn=pinn,
        component="imag",
        suffix="imag",
        c_ref=c_ref,
        c_pinn=c_pinn,
        c_shoot=c_shoot,
        neural_ymax=
            pinn_info[
                "neural_ymax"
            ],
    )

    meta = {
        "Mach":
            MACH,

        "alpha":
            ALPHA,

        "primary_chart":
            CHART,

        "plot_y_range": [
            -PLOT_YMAX,
            PLOT_YMAX,
        ],

        "n_plot_points":
            N_Y,

        "direct_pinn_native_mapped_ymax":
            pinn_info[
                "mapped_ymax"
            ],

        "direct_pinn_neural_ymax":
            pinn_info[
                "neural_ymax"
            ],

        "direct_pinn_tail":
            (
                "analytic far-field continuation "
                "outside neural mapped domain"
            ),

        "reference_c": {
            "cr":
                c_ref.real,
            "ci":
                c_ref.imag,
        },

        "direct_pinn_c": {
            "cr":
                c_pinn.real,
            "ci":
                c_pinn.imag,
        },

        "shooting_c": {
            "cr":
                c_shoot.real,
            "ci":
                c_shoot.imag,
        },

        "reference_y_limit":
            ylimit_ref,

        "shooting_y_limit":
            ylimit_shoot,

        "outputs": {
            "real":
                str(real_stem),

            "imag":
                str(imag_stem),

            "data":
                str(data_path),

            "metrics":
                str(metrics_path),
        },
    }

    meta_path = (
        OUT
        / "Fig_supersonic_modal_overlay_M140_a018_meta.json"
    )

    meta_path.write_text(
        json.dumps(
            meta,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "=" * 100
    )

    print(
        "ASSETS COMPLETE"
    )

    print(
        "=" * 100
    )

    for stem in [
        real_stem,
        imag_stem,
    ]:
        print(
            stem.with_suffix(
                ".png"
            )
        )

        print(
            stem.with_suffix(
                ".pdf"
            )
        )

        print(
            stem.with_suffix(
                ".svg"
            )
        )


if __name__ == "__main__":
    main()
