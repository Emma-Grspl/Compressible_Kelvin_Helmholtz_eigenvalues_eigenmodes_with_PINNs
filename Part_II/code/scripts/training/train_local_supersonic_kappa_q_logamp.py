#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.kh_supersonic_kappa_q_logamp import (
    KHSupersonicLocalPINN,
)
from src.physics.kh_supersonic_riccati_residual import (
    riccati_boundary_losses,
    riccati_regularized_residuals,
    y_to_xi,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    config: dict[str, Any],
    device: torch.device,
) -> KHSupersonicLocalPINN:
    model_config = config["model"]

    return KHSupersonicLocalPINN(
        mach=float(config["Mach"]),
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        xi_max=float(model_config["xi_max"]),
        mapping_scale=float(
            model_config["mapping_scale"]
        ),
        spectral_width=int(
            model_config["spectral_width"]
        ),
        spectral_depth=int(
            model_config["spectral_depth"]
        ),
        modal_width=int(
            model_config["modal_width"]
        ),
        modal_depth=int(
            model_config["modal_depth"]
        ),
        n_frequencies=int(
            model_config["n_frequencies"]
        ),
        mode_experts=int(
            model_config["mode_experts"]
        ),
        alpha_split=float(
            model_config["alpha_split"]
        ),
        alpha_gate_width=float(
            model_config["alpha_gate_width"]
        ),
        cr_min=float(model_config["cr_min"]),
        cr_max=float(model_config["cr_max"]),
        ci_floor=float(
            model_config["ci_floor"]
        ),
        ci_max=float(model_config["ci_max"]),
    ).to(device)


def load_spectral_targets(
    config: dict[str, Any],
    model: KHSupersonicLocalPINN,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    dtype = next(model.parameters()).dtype
    dataframe = pd.read_csv(
        resolve_path(
            config["spectral_anchor_file"]
        )
    ).sort_values("anchor_rank")

    alpha = torch.tensor(
        dataframe["alpha"].to_numpy(float),
        device=device,
        dtype=dtype,
    ).view(-1, 1)

    cr = torch.tensor(
        dataframe["cr"].to_numpy(float),
        device=device,
        dtype=dtype,
    ).view(-1, 1)

    ci = torch.tensor(
        dataframe["ci"].to_numpy(float),
        device=device,
        dtype=dtype,
    ).view(-1, 1)

    cr_scale = max(
        float(
            dataframe["cr"].max()
            - dataframe["cr"].min()
        ),
        0.05,
    )

    ci_scale = max(
        float(
            dataframe["ci"].max()
            - dataframe["ci"].min()
        ),
        0.02,
    )

    return {
        "alpha": alpha,
        "cr": cr,
        "ci": ci,
        "cr_scale": torch.tensor(
            cr_scale,
            device=device,
            dtype=dtype,
        ),
        "ci_scale": torch.tensor(
            ci_scale,
            device=device,
            dtype=dtype,
        ),
    }


def spectral_loss(
    model: KHSupersonicLocalPINN,
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    cr_prediction, ci_prediction = (
        model.get_spectrum(targets["alpha"])
    )

    loss_cr = (
        (
            cr_prediction - targets["cr"]
        )
        / targets["cr_scale"]
    ).square().mean()

    loss_ci = (
        (
            ci_prediction - targets["ci"]
        )
        / targets["ci_scale"]
    ).square().mean()

    epsilon = 1.0e-6

    loss_log_ci = (
        torch.log(ci_prediction + epsilon)
        - torch.log(targets["ci"] + epsilon)
    ).square().mean()

    return (
        loss_cr
        + loss_ci
        + 0.25 * loss_log_ci
    )


def load_modal_targets(
    path: Path,
    *,
    model: KHSupersonicLocalPINN,
    device: torch.device,
    points_per_mode: int,
) -> dict[str, torch.Tensor]:
    dtype = next(model.parameters()).dtype

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        mode_ptr = data[
            "mode_ptr"
        ].astype(np.int64)

        alpha_mode = data[
            "alpha"
        ].astype(float)

        y_all = data["y"].astype(float)
        kappa_all = data["kappa"].astype(float)
        q_all = data["q"].astype(float)

        log_amp_all = data[
            "logabs_p_center_gauge"
        ].astype(float)

    y_parts = []
    alpha_parts = []
    kappa_parts = []
    q_parts = []
    log_amp_parts = []

    for mode_index, alpha_value in enumerate(
        alpha_mode
    ):
        start = int(mode_ptr[mode_index])
        stop = int(mode_ptr[mode_index + 1])

        valid = (
            np.isfinite(y_all[start:stop])
            & np.isfinite(kappa_all[start:stop])
            & np.isfinite(q_all[start:stop])
            & np.isfinite(log_amp_all[start:stop])
            & (np.abs(y_all[start:stop]) <= 80.0)
        )

        valid_indices = np.flatnonzero(valid)

        n_selected = min(
            int(points_per_mode),
            len(valid_indices),
        )

        if n_selected < 32:
            raise RuntimeError(
                "Insufficient modal reference points "
                f"for alpha={alpha_value}"
            )

        positions = np.linspace(
            0,
            len(valid_indices) - 1,
            n_selected,
        ).round().astype(int)

        selected = start + valid_indices[positions]

        y_parts.append(y_all[selected])

        alpha_parts.append(
            np.full(
                n_selected,
                alpha_value,
                dtype=float,
            )
        )

        kappa_parts.append(kappa_all[selected])
        q_parts.append(q_all[selected])
        log_amp_parts.append(
            log_amp_all[selected]
        )

    y = torch.tensor(
        np.concatenate(y_parts),
        device=device,
        dtype=dtype,
    ).view(-1, 1)

    alpha = torch.tensor(
        np.concatenate(alpha_parts),
        device=device,
        dtype=dtype,
    ).view(-1, 1)

    kappa = torch.tensor(
        np.concatenate(kappa_parts),
        device=device,
        dtype=dtype,
    ).view(-1, 1)

    q = torch.tensor(
        np.concatenate(q_parts),
        device=device,
        dtype=dtype,
    ).view(-1, 1)

    log_amp = torch.tensor(
        np.concatenate(log_amp_parts),
        device=device,
        dtype=dtype,
    ).view(-1, 1)

    xi = y_to_xi(
        y,
        model.get_mapping_scale(),
    )

    return {
        "xi": xi,
        "alpha": alpha,
        "kappa": kappa,
        "q": q,
        "log_amp": log_amp,
        "log_amp_mask": log_amp > -12.0,
        "kappa_scale": torch.std(kappa).clamp_min(
            0.02
        ),
        "q_scale": torch.std(q).clamp_min(
            0.02
        ),
        "log_amp_scale": torch.std(
            log_amp[log_amp > -12.0]
        ).clamp_min(0.5),
    }


def modal_losses(
    model: KHSupersonicLocalPINN,
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    prediction = model(
        targets["xi"],
        targets["alpha"],
    )

    loss_kappa = (
        (
            prediction[:, 0:1]
            - targets["kappa"]
        )
        / targets["kappa_scale"]
    ).square().mean()

    loss_q = (
        (
            prediction[:, 1:2]
            - targets["q"]
        )
        / targets["q_scale"]
    ).square().mean()

    mask = targets["log_amp_mask"][:, 0]

    loss_log_amp = (
        (
            prediction[mask, 2:3]
            - targets["log_amp"][mask]
        )
        / targets["log_amp_scale"]
    ).square().mean()

    return {
        "modal_kappa": loss_kappa,
        "modal_q": loss_q,
        "modal_log_amp": loss_log_amp,
    }


def sample_interior(
    model: KHSupersonicLocalPINN,
    *,
    n_points: int,
    alpha_min: float,
    alpha_max: float,
    xi_max: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_global = int(round(0.40 * n_points))
    n_center = int(round(0.25 * n_points))
    n_critical = int(round(0.25 * n_points))

    n_tail = (
        n_points
        - n_global
        - n_center
        - n_critical
    )

    def random_alpha(count: int) -> torch.Tensor:
        return (
            alpha_min
            + (alpha_max - alpha_min)
            * torch.rand(
                count,
                1,
                generator=generator,
                device=device,
            )
        )

    alpha_global = random_alpha(n_global)

    xi_global = (
        -xi_max
        + 2.0 * xi_max
        * torch.rand(
            n_global,
            1,
            generator=generator,
            device=device,
        )
    )

    alpha_center = random_alpha(n_center)

    y_center = (
        1.5
        * torch.randn(
            n_center,
            1,
            generator=generator,
            device=device,
        )
    )

    xi_center = y_to_xi(
        y_center,
        model.get_mapping_scale(),
    )

    alpha_critical = random_alpha(
        n_critical
    )

    with torch.no_grad():
        cr_critical = model.get_cr(
            alpha_critical
        )

        y_critical_center = torch.atanh(
            cr_critical.clamp(
                min=-0.98,
                max=0.98,
            )
        )

    y_critical = (
        y_critical_center
        + 0.75
        * torch.randn(
            n_critical,
            1,
            generator=generator,
            device=device,
        )
    )

    xi_critical = y_to_xi(
        y_critical,
        model.get_mapping_scale(),
    )

    alpha_tail = random_alpha(n_tail)

    tail_sign = torch.where(
        torch.rand(
            n_tail,
            1,
            generator=generator,
            device=device,
        ) < 0.5,
        -torch.ones(
            n_tail,
            1,
            device=device,
        ),
        torch.ones(
            n_tail,
            1,
            device=device,
        ),
    )

    xi_tail = (
        tail_sign
        * (
            0.80 * xi_max
            + 0.20 * xi_max
            * torch.rand(
                n_tail,
                1,
                generator=generator,
                device=device,
            )
        )
    )

    xi = torch.cat(
        [
            xi_global,
            xi_center,
            xi_critical,
            xi_tail,
        ],
        dim=0,
    ).clamp(
        min=-xi_max,
        max=xi_max,
    )

    alpha = torch.cat(
        [
            alpha_global,
            alpha_center,
            alpha_critical,
            alpha_tail,
        ],
        dim=0,
    )

    xi.requires_grad_(True)

    return xi, alpha


def compute_losses(
    model: KHSupersonicLocalPINN,
    config: dict[str, Any],
    *,
    spectral_targets: dict[str, torch.Tensor] | None,
    modal_targets: dict[str, torch.Tensor] | None,
    generator: torch.Generator,
    n_interior: int,
    n_boundary: int,
    include_spectral: bool,
    include_modal: bool,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    model_config = config["model"]
    weights = config["loss_weights"]

    xi, alpha = sample_interior(
        model,
        n_points=n_interior,
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(config["alpha_max"]),
        xi_max=float(model_config["xi_max"]),
        generator=generator,
        device=device,
    )

    physics = riccati_regularized_residuals(
        model,
        xi,
        alpha,
        mach=float(config["Mach"]),
    )

    alpha_boundary = torch.linspace(
        float(config["alpha_min"]),
        float(config["alpha_max"]),
        n_boundary,
        device=device,
    ).view(-1, 1)

    boundary = riccati_boundary_losses(
        model,
        alpha_boundary,
        mach=float(config["Mach"]),
        xi_boundary=float(
            model_config["xi_max"]
        ),
    )

    zero = torch.zeros(
        (),
        device=device,
    )

    loss_spectral = zero

    if (
        include_spectral
        and spectral_targets is not None
    ):
        loss_spectral = spectral_loss(
            model,
            spectral_targets,
        )

    modal = {
        "modal_kappa": zero,
        "modal_q": zero,
        "modal_log_amp": zero,
    }

    if (
        include_modal
        and modal_targets is not None
    ):
        modal = modal_losses(
            model,
            modal_targets,
        )

    total = (
        float(weights["riccati_kappa"])
        * physics["loss_kappa"]
        + float(weights["riccati_q"])
        * physics["loss_phase_gradient"]
        + float(
            weights["log_amp_compatibility"]
        )
        * physics["loss_log_amp"]
        + float(weights["boundary_kappa"])
        * boundary["loss_bc_kappa"]
        + float(weights["boundary_q"])
        * boundary[
            "loss_bc_phase_gradient"
        ]
        + float(weights["spectral"])
        * loss_spectral
        + float(weights["modal_kappa"])
        * modal["modal_kappa"]
        + float(weights["modal_q"])
        * modal["modal_q"]
        + float(weights["modal_log_amp"])
        * modal["modal_log_amp"]
    )

    values = {
        "loss": float(total.detach()),
        "riccati_kappa": float(
            physics["loss_kappa"].detach()
        ),
        "riccati_q": float(
            physics[
                "loss_phase_gradient"
            ].detach()
        ),
        "log_amp_compatibility": float(
            physics["loss_log_amp"].detach()
        ),
        "boundary_kappa": float(
            boundary["loss_bc_kappa"].detach()
        ),
        "boundary_q": float(
            boundary[
                "loss_bc_phase_gradient"
            ].detach()
        ),
        "spectral": float(
            loss_spectral.detach()
        ),
        "modal_kappa": float(
            modal["modal_kappa"].detach()
        ),
        "modal_q": float(
            modal["modal_q"].detach()
        ),
        "modal_log_amp": float(
            modal["modal_log_amp"].detach()
        ),
    }

    return total, values


def evaluate_spectrum(
    model: KHSupersonicLocalPINN,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, float]:
    reference = pd.read_csv(
        resolve_path(
            config["spectral_reference"]
        )
    ).sort_values("alpha")

    dtype = next(model.parameters()).dtype

    alpha = torch.tensor(
        reference["alpha"].to_numpy(float),
        device=device,
        dtype=dtype,
    ).view(-1, 1)

    with torch.no_grad():
        cr_prediction, ci_prediction = (
            model.get_spectrum(alpha)
        )

    cr_prediction_np = (
        cr_prediction.cpu().numpy()[:, 0]
    )

    ci_prediction_np = (
        ci_prediction.cpu().numpy()[:, 0]
    )

    reference = reference.copy()

    reference["cr_pinn"] = cr_prediction_np
    reference["ci_pinn"] = ci_prediction_np

    reference["omega_i_pinn"] = (
        reference["alpha"]
        * reference["ci_pinn"]
    )

    reference["cr_error"] = (
        reference["cr_pinn"]
        - reference["cr"]
    )

    reference["ci_error"] = (
        reference["ci_pinn"]
        - reference["ci"]
    )

    reference.to_csv(
        output_dir
        / "spectral_audit_predictions.csv",
        index=False,
    )

    metrics = {
        "cr_mae": float(
            np.mean(
                np.abs(reference["cr_error"])
            )
        ),
        "cr_max_abs": float(
            np.max(
                np.abs(reference["cr_error"])
            )
        ),
        "ci_mae": float(
            np.mean(
                np.abs(reference["ci_error"])
            )
        ),
        "ci_max_abs": float(
            np.max(
                np.abs(reference["ci_error"])
            )
        ),
        "omega_i_mae": float(
            np.mean(
                np.abs(
                    reference["omega_i_pinn"]
                    - reference["omega_i"]
                )
            )
        ),
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--prefit-steps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--modal-steps",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--joint-steps",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--spectral-lr",
        type=float,
        default=2.0e-4,
    )

    parser.add_argument(
        "--modal-lr",
        type=float,
        default=1.0e-4,
    )

    parser.add_argument(
        "--joint-spectral-lr",
        type=float,
        default=5.0e-5,
    )

    parser.add_argument(
        "--joint-modal-lr",
        type=float,
        default=5.0e-5,
    )

    parser.add_argument(
        "--n-interior",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--n-boundary",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--modal-points-per-mode",
        type=int,
        default=257,
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    config_path = resolve_path(args.config)

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else resolve_path(config["output_dir"])
    )

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists"
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    device = torch.device(args.device)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    seed = int(config["seed"])

    set_seed(seed)

    model = build_model(
        config,
        device,
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "config": config,
        },
        output_dir / "initial_checkpoint.pt",
    )

    spectral_targets = None

    if config[
        "use_spectral_supervision"
    ]:
        spectral_targets = (
            load_spectral_targets(
                config,
                model,
                device,
            )
        )

    modal_targets = None

    if config["use_modal_supervision"]:
        modal_targets = load_modal_targets(
            resolve_path(
                config["modal_anchor_file"]
            ),
            model=model,
            device=device,
            points_per_mode=(
                args.modal_points_per_mode
            ),
        )

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(seed)

    history: list[dict[str, Any]] = []

    best_stage_losses: dict[str, float] = {}
    best_checkpoint_paths: dict[str, Path] = {}

    global_step = 0
    start_time = time.time()

    def run_stage(
        name: str,
        steps: int,
        optimizer: torch.optim.Optimizer,
        *,
        include_spectral: bool,
        include_modal: bool,
    ) -> None:
        nonlocal global_step

        for stage_step in range(
            1,
            steps + 1,
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            loss, values = compute_losses(
                model,
                config,
                spectral_targets=(
                    spectral_targets
                ),
                modal_targets=modal_targets,
                generator=generator,
                n_interior=args.n_interior,
                n_boundary=args.n_boundary,
                include_spectral=(
                    include_spectral
                ),
                include_modal=include_modal,
                device=device,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss in {name}"
                )

            loss.backward()

            trainable = [
                parameter
                for parameter in model.parameters()
                if (
                    parameter.requires_grad
                    and parameter.grad is not None
                )
            ]

            nn.utils.clip_grad_norm_(
                trainable,
                max_norm=5.0,
            )

            optimizer.step()

            global_step += 1

            row = {
                "stage": name,
                "stage_step": stage_step,
                "global_step": global_step,
                "elapsed_seconds": (
                    time.time() - start_time
                ),
                **values,
            }

            history.append(row)

            current_loss = float(
                values["loss"]
            )

            previous_best = (
                best_stage_losses.get(
                    name,
                    float("inf"),
                )
            )

            if current_loss < previous_best:
                best_stage_losses[name] = (
                    current_loss
                )

                best_path = (
                    output_dir
                    / f"best_{name}_checkpoint.pt"
                )

                torch.save(
                    {
                        "model_state_dict": (
                            model.state_dict()
                        ),
                        "config": config,
                        "stage": name,
                        "stage_step": stage_step,
                        "global_step": global_step,
                        "loss": current_loss,
                    },
                    best_path,
                )

                best_checkpoint_paths[name] = (
                    best_path
                )

            if (
                stage_step == 1
                or stage_step % args.print_every == 0
                or stage_step == steps
            ):
                print(
                    f"[{name}] "
                    f"{stage_step}/{steps} "
                    f"loss={values['loss']:.6e} "
                    f"ric=({values['riccati_kappa']:.3e},"
                    f"{values['riccati_q']:.3e}) "
                    f"spec={values['spectral']:.3e} "
                    f"modal=({values['modal_kappa']:.3e},"
                    f"{values['modal_q']:.3e},"
                    f"{values['modal_log_amp']:.3e})",
                    flush=True,
                )

    if (
        spectral_targets is not None
        and args.prefit_steps > 0
    ):
        for parameter in model.modal_parameters():
            parameter.requires_grad_(False)

        for parameter in model.spectral_parameters():
            parameter.requires_grad_(True)

        optimizer = torch.optim.Adam(
            model.spectral_parameters(),
            lr=args.spectral_lr,
        )

        for step in range(
            1,
            args.prefit_steps + 1,
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            loss = spectral_loss(
                model,
                spectral_targets,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                list(
                    model.spectral_parameters()
                ),
                max_norm=5.0,
            )

            optimizer.step()

            global_step += 1

            history.append(
                {
                    "stage": "spectral_prefit",
                    "stage_step": step,
                    "global_step": global_step,
                    "elapsed_seconds": (
                        time.time() - start_time
                    ),
                    "loss": float(
                        loss.detach()
                    ),
                    "spectral": float(
                        loss.detach()
                    ),
                }
            )

            if (
                step == 1
                or step % args.print_every == 0
                or step == args.prefit_steps
            ):
                print(
                    "[spectral_prefit] "
                    f"{step}/{args.prefit_steps} "
                    f"loss={float(loss.detach()):.6e}",
                    flush=True,
                )

    for parameter in model.spectral_parameters():
        parameter.requires_grad_(False)

    for parameter in model.modal_parameters():
        parameter.requires_grad_(True)

    if args.modal_steps > 0:
        modal_optimizer = torch.optim.Adam(
            model.modal_parameters(),
            lr=args.modal_lr,
        )

        run_stage(
            "modal_frozen_spectrum",
            args.modal_steps,
            modal_optimizer,
            include_spectral=False,
            include_modal=(
                modal_targets is not None
            ),
        )

    for parameter in model.parameters():
        parameter.requires_grad_(True)

    if args.joint_steps > 0:
        joint_optimizer = torch.optim.Adam(
            [
                {
                    "params": (
                        model.spectral_parameters()
                    ),
                    "lr": args.joint_spectral_lr,
                },
                {
                    "params": (
                        model.modal_parameters()
                    ),
                    "lr": args.joint_modal_lr,
                },
            ]
        )

        run_stage(
            "joint",
            args.joint_steps,
            joint_optimizer,
            include_spectral=(
                spectral_targets is not None
            ),
            include_modal=(
                modal_targets is not None
            ),
        )

    last_checkpoint_path = (
        output_dir / "last_checkpoint.pt"
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "config": config,
            "global_step": global_step,
        },
        last_checkpoint_path,
    )

    selected_stage = None
    selected_checkpoint_path = None

    for candidate_stage in [
        "joint",
        "modal_frozen_spectrum",
    ]:
        if candidate_stage in best_checkpoint_paths:
            selected_stage = candidate_stage
            selected_checkpoint_path = (
                best_checkpoint_paths[
                    candidate_stage
                ]
            )
            break

    if selected_checkpoint_path is not None:
        selected_checkpoint = torch.load(
            selected_checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(
            selected_checkpoint[
                "model_state_dict"
            ]
        )

    selection_report = {
        "selected_stage": selected_stage,
        "selected_checkpoint": (
            str(selected_checkpoint_path)
            if selected_checkpoint_path
            is not None
            else str(last_checkpoint_path)
        ),
        "best_stage_losses": (
            best_stage_losses
        ),
        "global_step": global_step,
    }

    (
        output_dir
        / "checkpoint_selection.json"
    ).write_text(
        json.dumps(
            selection_report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    history_dataframe = pd.DataFrame(
        history
    )

    history_dataframe.to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    metrics = evaluate_spectrum(
        model,
        config,
        output_dir,
        device,
    )

    (
        output_dir / "spectral_metrics.json"
    ).write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "config": config,
            "training_arguments": vars(args),
            "spectral_metrics": metrics,
        },
        output_dir / "final_checkpoint.pt",
    )

    (
        output_dir / "resolved_config.json"
    ).write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("SPECTRAL METRICS")
    print(json.dumps(metrics, indent=2))
    print()
    print(
        "LOCAL SUPERSONIC TRAINING: COMPLETED"
    )


if __name__ == "__main__":
    main()
