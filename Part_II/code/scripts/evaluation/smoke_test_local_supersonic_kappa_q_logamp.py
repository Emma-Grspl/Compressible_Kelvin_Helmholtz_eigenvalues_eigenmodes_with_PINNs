#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.kh_supersonic_kappa_q_logamp import (
    KHSupersonicLocalPINN,
)
from src.physics.kh_supersonic_riccati_residual import (
    asymptotic_riccati_gammas,
    riccati_boundary_losses,
    riccati_regularized_residuals,
    y_to_xi,
)


torch.set_default_dtype(torch.float64)

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    REPO_ROOT / 'code/configs/legacy/S4M4.json'
)

MODAL_BANK_PATH = (
    REPO_ROOT
    / "assets/pinn_supersonic/pilots/"
    "local_M100_sparse_v1/data/"
    "modal_bank_S4M4_M100.npz"
)


def finite_gradient_audit(
    model: torch.nn.Module,
) -> tuple[int, int]:
    total = 0
    finite = 0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        total += int(parameter.grad.numel())

        finite += int(
            torch.isfinite(
                parameter.grad
            ).sum().item()
        )

    return finite, total


def main() -> None:
    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    device = torch.device("cpu")

    model = KHSupersonicLocalPINN(
        mach=float(config["Mach"]),
        alpha_min=float(
            config["alpha_min"]
        ),
        alpha_max=float(
            config["alpha_max"]
        ),
        xi_max=0.985,
        mapping_scale=3.0,
        spectral_width=96,
        spectral_depth=3,
        modal_width=128,
        modal_depth=4,
        n_frequencies=8,
        mode_experts=2,
        alpha_split=0.20,
        alpha_gate_width=0.02,
        cr_min=-0.10,
        cr_max=0.30,
        ci_floor=1.0e-6,
        ci_max=0.20,
    ).to(device)

    anchors = pd.read_csv(
        REPO_ROOT
        / config["spectral_anchor_file"]
    )

    alpha_anchor = torch.tensor(
        anchors["alpha"]
        .to_numpy(float),
        device=device,
    ).view(-1, 1)

    cr_reference = torch.tensor(
        anchors["cr"]
        .to_numpy(float),
        device=device,
    ).view(-1, 1)

    ci_reference = torch.tensor(
        anchors["ci"]
        .to_numpy(float),
        device=device,
    ).view(-1, 1)

    cr_prediction, ci_prediction = (
        model.get_spectrum(alpha_anchor)
    )

    loss_spectral = (
        (
            (cr_prediction - cr_reference)
            / 0.20
        )
        .square()
        .mean()
        +
        (
            (ci_prediction - ci_reference)
            / 0.06
        )
        .square()
        .mean()
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(12345)

    n_interior = 512

    xi = (
        -0.985
        + 2.0
        * 0.985
        * torch.rand(
            n_interior,
            1,
            generator=generator,
            device=device,
        )
    )

    alpha = (
        float(config["alpha_min"])
        + (
            float(config["alpha_max"])
            - float(config["alpha_min"])
        )
        * torch.rand(
            n_interior,
            1,
            generator=generator,
            device=device,
        )
    )

    xi.requires_grad_(True)

    physics = (
        riccati_regularized_residuals(
            model,
            xi,
            alpha,
            mach=float(config["Mach"]),
        )
    )

    alpha_boundary = torch.linspace(
        float(config["alpha_min"]),
        float(config["alpha_max"]),
        64,
        device=device,
    ).view(-1, 1)

    boundary = riccati_boundary_losses(
        model,
        alpha_boundary,
        mach=float(config["Mach"]),
        xi_boundary=0.985,
    )

    with np.load(
        MODAL_BANK_PATH,
        allow_pickle=False,
    ) as bank:
        mode_ptr = bank[
            "mode_ptr"
        ].astype(np.int64)

        # Use the fourth modal anchor for the smoke test.
        mode_index = 3

        start = int(
            mode_ptr[mode_index]
        )

        stop = int(
            mode_ptr[mode_index + 1]
        )

        y_numpy = bank["y"][
            start:stop
        ].astype(float)

        kappa_numpy = bank["kappa"][
            start:stop
        ].astype(float)

        q_numpy = bank["q"][
            start:stop
        ].astype(float)

        log_amp_numpy = bank[
            "logabs_p_center_gauge"
        ][start:stop].astype(float)

        alpha_modal_value = float(
            bank["alpha"][mode_index]
        )

    valid = (
        np.isfinite(y_numpy)
        & np.isfinite(kappa_numpy)
        & np.isfinite(q_numpy)
        & np.isfinite(log_amp_numpy)
        & (np.abs(y_numpy) <= 80.0)
    )

    valid_indices = np.flatnonzero(
        valid
    )

    if len(valid_indices) < 257:
        raise RuntimeError(
            "Not enough valid modal points"
        )

    selection = np.linspace(
        0,
        len(valid_indices) - 1,
        257,
    ).round().astype(int)

    selected_indices = valid_indices[
        selection
    ]

    y_modal = torch.tensor(
        y_numpy[selected_indices],
        device=device,
    ).view(-1, 1)

    alpha_modal = torch.full_like(
        y_modal,
        alpha_modal_value,
    )

    xi_modal = y_to_xi(
        y_modal,
        model.get_mapping_scale(),
    )

    modal_prediction = model(
        xi_modal,
        alpha_modal,
    )

    kappa_reference = torch.tensor(
        kappa_numpy[selected_indices],
        device=device,
    ).view(-1, 1)

    q_reference = torch.tensor(
        q_numpy[selected_indices],
        device=device,
    ).view(-1, 1)

    log_amp_reference = torch.tensor(
        log_amp_numpy[selected_indices],
        device=device,
    ).view(-1, 1)

    loss_modal_kappa = (
        (
            modal_prediction[:, 0:1]
            - kappa_reference
        )
        / 0.05
    ).square().mean()

    loss_modal_q = (
        (
            modal_prediction[:, 1:2]
            - q_reference
        )
        / 0.05
    ).square().mean()

    # Do not supervise the clipped log-amplitude tail.
    amplitude_mask = (
        log_amp_reference > -12.0
    )

    if not amplitude_mask.any():
        raise RuntimeError(
            "Empty log-amplitude mask"
        )

    loss_modal_log_amp = (
        (
            modal_prediction[
                amplitude_mask[:, 0],
                2:3,
            ]
            - log_amp_reference[
                amplitude_mask
            ].view(-1, 1)
        )
        / 2.0
    ).square().mean()

    total_loss = (
        physics["loss_kappa"]
        + physics[
            "loss_phase_gradient"
        ]
        + 10.0
        * physics["loss_log_amp"]
        + 20.0
        * boundary["loss_bc_kappa"]
        + 20.0
        * boundary[
            "loss_bc_phase_gradient"
        ]
        + 100.0
        * loss_spectral
        + loss_modal_kappa
        + loss_modal_q
        + loss_modal_log_amp
    )

    model.zero_grad(
        set_to_none=True
    )

    total_loss.backward()

    finite_gradients, total_gradients = (
        finite_gradient_audit(model)
    )

    if total_gradients == 0:
        raise RuntimeError(
            "No model gradient was produced"
        )

    if finite_gradients != total_gradients:
        raise RuntimeError(
            "Non-finite model gradients: "
            f"{finite_gradients}/"
            f"{total_gradients}"
        )

    gamma_left, gamma_right = (
        asymptotic_riccati_gammas(
            alpha_anchor,
            float(config["Mach"]),
            cr_prediction,
            ci_prediction,
        )
    )

    if not torch.all(
        gamma_left.real > 0.0
    ):
        raise RuntimeError(
            "Invalid left asymptotic branch"
        )

    if not torch.all(
        gamma_right.real < 0.0
    ):
        raise RuntimeError(
            "Invalid right asymptotic branch"
        )

    print("CONFIG")
    print("  Mach             :", config["Mach"])
    print(
        "  alpha interval   :",
        config["alpha_min"],
        config["alpha_max"],
    )

    print()
    print("OUTPUTS")
    print(
        "  spectral shape   :",
        tuple(cr_prediction.shape),
        tuple(ci_prediction.shape),
    )
    print(
        "  modal shape      :",
        tuple(modal_prediction.shape),
    )
    print(
        "  modal variables  :",
        "kappa, phase_gradient, log_amp",
    )
    print(
        "  predicted phase  :",
        False,
    )

    print()
    print("LOSSES")
    print(
        "  physics kappa    :",
        float(
            physics[
                "loss_kappa"
            ].detach()
        ),
    )
    print(
        "  physics q        :",
        float(
            physics[
                "loss_phase_gradient"
            ].detach()
        ),
    )
    print(
        "  amplitude        :",
        float(
            physics[
                "loss_log_amp"
            ].detach()
        ),
    )
    print(
        "  BC kappa         :",
        float(
            boundary[
                "loss_bc_kappa"
            ].detach()
        ),
    )
    print(
        "  BC q             :",
        float(
            boundary[
                "loss_bc_phase_gradient"
            ].detach()
        ),
    )
    print(
        "  spectral         :",
        float(
            loss_spectral.detach()
        ),
    )
    print(
        "  modal kappa      :",
        float(
            loss_modal_kappa.detach()
        ),
    )
    print(
        "  modal q          :",
        float(
            loss_modal_q.detach()
        ),
    )
    print(
        "  modal log_amp    :",
        float(
            loss_modal_log_amp.detach()
        ),
    )
    print(
        "  total            :",
        float(total_loss.detach()),
    )

    print()
    print(
        "LOG-AMPLITUDE MASK"
    )
    print(
        "  retained         :",
        int(amplitude_mask.sum()),
        "/",
        int(amplitude_mask.numel()),
    )

    print()
    print("GRADIENTS")
    print(
        "  finite           :",
        finite_gradients,
        "/",
        total_gradients,
    )

    print()
    print(
        "SUPERSONIC KAPPA-Q-LOGAMP "
        "SMOKE TEST: OK"
    )


if __name__ == "__main__":
    main()
