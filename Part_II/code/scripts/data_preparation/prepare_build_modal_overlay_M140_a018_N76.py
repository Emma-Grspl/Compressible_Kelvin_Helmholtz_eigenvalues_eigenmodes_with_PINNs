#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

from classic_supersonic_reference.solver.mstab17_supersonic_solver import (
    Mstab17SupersonicSolver,
)


# =============================================================================
# Configuration
# =============================================================================

REPO = Path.cwd().resolve()

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MACH = 1.40
ALPHA = 0.18

CHART = "C11"
MODE_INDEX = 26

PLOT_YMAX = 80.0
CORE_YMAX = 40.0
N_Y = 3201

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

REFERENCE_NPZ = (
    REPO / 'assets/classic_supersonic/data/modal_reconstruction/modes/modes_compact_with_analytic_tails_bf31f70f37.npz'
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
# Generic helpers
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

    spec.loader.exec_module(
        module
    )

    return module


def interp_complex(
    y_src: np.ndarray,
    f_src: np.ndarray,
    y_dst: np.ndarray,
) -> np.ndarray:

    order = np.argsort(
        y_src
    )

    ys = np.asarray(
        y_src,
        float,
    )[order]

    fs = np.asarray(
        f_src,
        complex,
    )[order]

    return (
        np.interp(
            y_dst,
            ys,
            fs.real,
        )
        + 1j
        * np.interp(
            y_dst,
            ys,
            fs.imag,
        )
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
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
) -> complex:

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
    fields: dict[str, np.ndarray],
    factor: complex,
):
    return {
        name:
            factor * value
        for name, value
        in fields.items()
    }


def relative_l2(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
):
    a = (
        pred[mask]
        - ref[mask]
    )

    b = ref[
        mask
    ]

    return float(
        np.linalg.norm(a)
        / max(
            np.linalg.norm(b),
            1.0e-30,
        )
    )


def overlap(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
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
    y: np.ndarray,
    q: np.ndarray,
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

    # y >= 0
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

    # y <= 0
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
    y: np.ndarray,
    scale: float,
):
    """
    Stable inverse of

        y = scale * xi / (1 - xi^2).
    """

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
# T401 spectral values
# =============================================================================

def load_T401_row():

    df = pd.read_csv(
        T401
    )

    sub = df[
        np.isclose(
            df["Mach"],
            MACH,
            atol=1e-12,
        )
        & np.isclose(
            df["alpha"],
            ALPHA,
            atol=1e-12,
        )
    ]

    if len(sub) != 1:
        raise RuntimeError(
            f"Expected one T401 row, got {len(sub)}."
        )

    row = sub.iloc[0]

    if str(
        row["atlas_chart"]
    ) != CHART:
        raise RuntimeError(
            "Unexpected primary chart: "
            f"{row['atlas_chart']}"
        )

    return row


# =============================================================================
# Classical frozen reference
# =============================================================================

def load_reference(
    y_common: np.ndarray,
):

    z = np.load(
        REFERENCE_NPZ,
        allow_pickle=True,
    )

    i = MODE_INDEX

    assert np.isclose(
        z["Mach"][i],
        MACH,
    )

    assert np.isclose(
        z["alpha"][i],
        ALPHA,
    )

    y = np.asarray(
        z["y"][i],
        float,
    )

    kappa = np.asarray(
        z["kappa"][i],
        float,
    )

    q = np.asarray(
        z["q"][i],
        float,
    )

    p = (
        np.asarray(
            z["p_real"][i],
            float,
        )
        + 1j
        * np.asarray(
            z["p_imag"][i],
            float,
        )
    )

    c = complex(
        float(
            z["cr"][i]
        ),
        float(
            z["ci"][i]
        ),
    )

    gamma = (
        kappa
        + 1j * q
    )

    fields = reconstruct_primitives(
        y=y,
        p=p,
        gamma=gamma,
        c=c,
    )

    fields_common = {
        name:
            interp_complex(
                y,
                value,
                y_common,
            )
        for name, value
        in fields.items()
    }

    # One common normalization exactly as in the classical figure:
    # max |p| = 1.
    pmax = float(
        np.max(
            np.abs(
                fields_common["p"]
            )
        )
    )

    fields_common = {
        name:
            value / pmax
        for name, value
        in fields_common.items()
    }

    return (
        fields_common,
        c,
    )


# =============================================================================
# PINN direct modal field
# =============================================================================

def load_direct_PINN(
    y_common: np.ndarray,
):

    trainer = load_module(
        TRAINER_PATH,
        "modal_overlay_trainer",
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model = trainer.build_model(
        checkpoint[
            "config"
        ]
    ).to(
        DEVICE
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

    supported_y = (
        scale
        * xi_max
        / (
            1.0
            - xi_max**2
        )
    )

    print(
        "PINN mapping:",
        f"scale={scale}",
        f"xi_max={xi_max}",
        f"|y|max={supported_y:.6f}",
    )

    if PLOT_YMAX >= supported_y:
        raise RuntimeError(
            "Requested plotting domain exceeds "
            "the PINN mapped domain."
        )

    xi = y_to_xi(
        y_common,
        scale,
    )

    if np.max(
        np.abs(xi)
    ) >= xi_max:
        raise RuntimeError(
            "Mapped xi outside model support."
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

    modal = (
        modal.detach()
        .cpu()
        .numpy()
    )

    if (
        modal.ndim != 2
        or modal.shape[1] != 3
    ):
        raise RuntimeError(
            f"Unexpected modal output shape: "
            f"{modal.shape}"
        )

    kappa = modal[:, 0]
    q = modal[:, 1]
    log_amp = modal[:, 2]

    phi = integrate_phase(
        y_common,
        q,
    )

    p = np.exp(
        log_amp
        + 1j * phi
    )

    gamma = (
        kappa
        + 1j * q
    )

    c = complex(
        cr,
        ci,
    )

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
            "kappa": kappa,
            "q": q,
            "log_amp": log_amp,
            "xi": xi,
            "scale": scale,
            "xi_max": xi_max,
        },
    )


# =============================================================================
# PINN-seeded shooting mode at corrected c
# =============================================================================

def reconstruct_corrected_shooting(
    y_common: np.ndarray,
    *,
    cr: float,
    ci: float,
):

    # We do NOT rerun root finding.
    #
    # We reconstruct the actual Riccati mode at the corrected
    # eigenvalue already stored by T401.

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
            "Shooting modal reconstruction failed."
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
        ln_right_arr,
        phi_right,
    ) = sol_right.y

    # Match the arbitrary phase at y=0.
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
        np.exp(ln_right_arr)
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

    # Left branch supplies y<0.
    # Right branch is integrated from +L to 0
    # and therefore needs reversing.
    left_mask = (
        y_left < 0.0
    )

    y = np.concatenate(
        [
            y_left[
                left_mask
            ],
            y_right[::-1],
        ]
    )

    p = np.concatenate(
        [
            p_left[
                left_mask
            ],
            p_right[::-1],
        ]
    )

    gamma = np.concatenate(
        [
            gamma_left[
                left_mask
            ],
            gamma_right[::-1],
        ]
    )

    c = complex(
        cr,
        ci,
    )

    fields = reconstruct_primitives(
        y=y,
        p=p,
        gamma=gamma,
        c=c,
    )

    fields_common = {
        name:
            interp_complex(
                y,
                value,
                y_common,
            )
        for name, value
        in fields.items()
    }

    return (
        fields_common,
        c,
        float(y_limit),
        float(ln_right),
    )


# =============================================================================
# Main
# =============================================================================

def main():

    print(
        "=" * 100
    )

    print(
        "FINAL MODAL OVERLAY:"
        " M=1.4 alpha=0.18"
    )

    print(
        "=" * 100
    )

    row = load_T401_row()

    print()
    print(
        "primary chart =",
        row["atlas_chart"],
    )

    print(
        "reference c =",
        f"{row['cr_reference']:.12f}",
        "+",
        f"{row['ci_reference']:.12f} i",
    )

    print(
        "PINN c      =",
        f"{row['cr_pinn']:.12f}",
        "+",
        f"{row['ci_pinn']:.12f} i",
    )

    print(
        "shoot c     =",
        f"{row['shoot_cr']:.12f}",
        "+",
        f"{row['shoot_ci']:.12f} i",
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

    # -------------------------------------------------------------------------
    # Reference
    # -------------------------------------------------------------------------

    reference, c_ref = (
        load_reference(
            y
        )
    )

    # -------------------------------------------------------------------------
    # Direct PINN
    # -------------------------------------------------------------------------

    pinn_raw, c_pinn, pinn_aux = (
        load_direct_PINN(
            y
        )
    )

    # Sanity check against the already frozen T401 spectral output.
    d_checkpoint = abs(
        c_pinn
        - complex(
            float(
                row["cr_pinn"]
            ),
            float(
                row["ci_pinn"]
            ),
        )
    )

    print()
    print(
        "checkpoint vs T401 |dc| =",
        f"{d_checkpoint:.12e}",
    )

    if d_checkpoint > 1.0e-6:
        raise RuntimeError(
            "Loaded checkpoint does not reproduce "
            "the frozen T401 PINN prediction."
        )

    # -------------------------------------------------------------------------
    # Corrected shooting
    # -------------------------------------------------------------------------

    shooting_raw, c_shoot, y_limit, ln_right = (
        reconstruct_corrected_shooting(
            y,
            cr=float(
                row["shoot_cr"]
            ),
            ci=float(
                row["shoot_ci"]
            ),
        )
    )

    print(
        "shooting reconstruction y_limit =",
        y_limit,
    )

    print(
        "shooting ln_p_start_right =",
        ln_right,
    )

    # -------------------------------------------------------------------------
    # ONE complex alignment factor per predicted method,
    # calculated from p on |y| <= 40,
    # then shared across p,rho,u,v.
    # -------------------------------------------------------------------------

    A_pinn = complex_alignment(
        pinn_raw["p"],
        reference["p"],
        core40,
    )

    A_shoot = complex_alignment(
        shooting_raw["p"],
        reference["p"],
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

    print()
    print(
        "PINN alignment factor  =",
        A_pinn,
    )

    print(
        "shoot alignment factor =",
        A_shoot,
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metric_rows = []

    for method_name, fields in [
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

            metric_rows.append(
                {
                    "method":
                        method_name,

                    "field":
                        field,

                    "rel_l2_core40":
                        relative_l2(
                            fields[field],
                            reference[field],
                            core40,
                        ),

                    "overlap_core40":
                        overlap(
                            fields[field],
                            reference[field],
                            core40,
                        ),
                }
            )

    metrics = pd.DataFrame(
        metric_rows
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
    # Save data
    # -------------------------------------------------------------------------

    data = pd.DataFrame(
        {
            "y": y,
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
        / "Fig_supersonic_modal_overlay_M140_a018_data.csv"
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
    # Plot
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            15.5,
            10.5,
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

    methods = [
        (
            "Classical reference",
            reference,
            "black",
            2.1,
            1.0,
        ),
        (
            "PINN-seeded shooting",
            shooting,
            "tab:blue",
            1.8,
            0.90,
        ),
        (
            "Direct PINN",
            pinn,
            "tab:red",
            1.7,
            0.90,
        ),
    ]

    for ax, (
        title,
        field,
    ) in zip(
        axes,
        panels,
    ):

        # Draw shooting first so the nearly coincident
        # classical reference remains visible.
        drawing_order = [
            methods[1],
            methods[0],
            methods[2],
        ]

        for (
            _,
            fields,
            color,
            linewidth,
            alpha_plot,
        ) in drawing_order:

            value = fields[
                field
            ]

            ax.plot(
                y,
                value.real,
                color=color,
                linewidth=linewidth,
                linestyle="-",
                alpha=alpha_plot,
            )

            ax.plot(
                y,
                value.imag,
                color=color,
                linewidth=linewidth,
                linestyle="--",
                alpha=alpha_plot,
            )

        ax.axvline(
            0.0,
            linestyle=":",
            linewidth=0.9,
            color="0.5",
        )

        ax.set_title(
            title,
            fontsize=15,
            loc="left",
        )

        ax.set_ylabel(
            "Amplitude",
            fontsize=13,
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.set_xlim(
            -PLOT_YMAX,
            PLOT_YMAX,
        )

    axes[2].set_xlabel(
        r"Transverse coordinate $y$",
        fontsize=13,
    )

    axes[3].set_xlabel(
        r"Transverse coordinate $y$",
        fontsize=13,
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.1,
            label="Classical reference",
        ),
        Line2D(
            [0],
            [0],
            color="tab:blue",
            lw=2.0,
            label="PINN-seeded shooting",
        ),
        Line2D(
            [0],
            [0],
            color="tab:red",
            lw=2.0,
            label="Direct PINN",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            lw=1.8,
            linestyle="-",
            label=r"$\Re$",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            lw=1.8,
            linestyle="--",
            label=r"$\Im$",
        ),
    ]

    fig.legend(
        handles=method_handles,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(
            0.5,
            0.935,
        ),
        frameon=False,
        fontsize=11,
    )

    fig.suptitle(
        (
            r"Representative supersonic mode comparison: "
            rf"$M={MACH:.1f}$, "
            rf"$\alpha={ALPHA:.2f}$"
            "\n"
            rf"$c^\star={c_ref.real:.6f}"
            rf"+{c_ref.imag:.6f}i$, "
            rf"$c_{{\rm PINN}}={c_pinn.real:.6f}"
            rf"+{c_pinn.imag:.6f}i$, "
            rf"$c_{{\rm shoot}}={c_shoot.real:.6f}"
            rf"+{c_shoot.imag:.6f}i$"
        ),
        fontsize=16,
        y=0.995,
    )

    fig.tight_layout(
        rect=[
            0.02,
            0.02,
            0.98,
            0.89,
        ]
    )

    png = (
        OUT
        / "Fig_supersonic_modal_overlay_M140_a018.png"
    )

    pdf = (
        OUT
        / "Fig_supersonic_modal_overlay_M140_a018.pdf"
    )

    fig.savefig(
        png,
        dpi=250,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata = {
        "Mach":
            MACH,

        "alpha":
            ALPHA,

        "primary_chart":
            CHART,

        "reference_mode_index":
            MODE_INDEX,

        "plot_ymax":
            PLOT_YMAX,

        "alignment_core_ymax":
            CORE_YMAX,

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

        "pinn_seeded_shooting_c": {
            "cr":
                c_shoot.real,
            "ci":
                c_shoot.imag,
        },

        "alignment_factor_direct_pinn": {
            "real":
                A_pinn.real,
            "imag":
                A_pinn.imag,
        },

        "alignment_factor_shooting": {
            "real":
                A_shoot.real,
            "imag":
                A_shoot.imag,
        },

        "shooting_reconstruction_y_limit":
            y_limit,

        "outputs": {
            "png":
                str(png),
            "pdf":
                str(pdf),
            "data":
                str(data_path),
            "metrics":
                str(metrics_path),
        },
    }

    metadata_path = (
        OUT
        / "Fig_supersonic_modal_overlay_M140_a018_meta.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "=" * 100
    )

    print(
        "ASSET COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        png
    )

    print(
        pdf
    )

    print(
        data_path
    )

    print(
        metrics_path
    )

    print(
        metadata_path
    )


if __name__ == "__main__":
    main()
