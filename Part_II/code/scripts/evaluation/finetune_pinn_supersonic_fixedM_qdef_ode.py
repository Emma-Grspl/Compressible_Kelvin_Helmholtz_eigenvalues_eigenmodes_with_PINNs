#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, activation):
        super().__init__()

        activations = {
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }

        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")

        layers = [
            nn.Linear(in_dim, width),
            activations[activation](),
        ]

        for _ in range(depth - 1):
            layers += [
                nn.Linear(width, width),
                activations[activation](),
            ]

        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def alpha_feature(alpha, alpha_min, alpha_max):
    return (
        2.0 * (alpha - alpha_min)
        / (alpha_max - alpha_min)
        - 1.0
    )


def y_feature(y, transform, scale, ymax):
    if transform == "tanh":
        return torch.tanh(y / scale)

    if transform == "asinh":
        return (
            torch.asinh(y / scale)
            / math.asinh(ymax / scale)
        )

    if transform == "compact":
        return y / (scale + torch.abs(y))

    raise ValueError(f"Unknown y transform: {transform}")


def modal_forward(
    model,
    y,
    alpha,
    *,
    alpha_min,
    alpha_max,
    y_transform,
    y_scale,
    ymax,
    x_mean,
    x_std,
    modal_mean,
    modal_std,
):
    features = torch.stack(
        [
            y_feature(
                y,
                y_transform,
                y_scale,
                ymax,
            ),
            alpha_feature(
                alpha,
                alpha_min,
                alpha_max,
            ),
        ],
        dim=1,
    )

    normalized_features = (
        features - x_mean
    ) / x_std

    normalized_output = model(
        normalized_features
    )

    physical_output = (
        normalized_output * modal_std
        + modal_mean
    )

    return normalized_output, physical_output


def spectral_forward(
    model,
    alpha,
    *,
    alpha_min,
    alpha_max,
    x_mean,
    x_std,
    spec_mean,
    spec_std,
):
    alpha_x = alpha_feature(
        alpha,
        alpha_min,
        alpha_max,
    ).unsqueeze(1)

    alpha_x_normalized = (
        alpha_x - x_mean[1:2]
    ) / x_std[1:2]

    return (
        model(alpha_x_normalized) * spec_std
        + spec_mean
    )


def physics_losses(
    modal_net,
    spectral_net,
    y,
    alpha,
    *,
    Mach,
    alpha_min,
    alpha_max,
    y_transform,
    y_scale,
    ymax,
    x_mean,
    x_std,
    modal_mean,
    modal_std,
    spec_mean,
    spec_std,
    create_graph,
):
    y = y.requires_grad_(True)

    _, modal = modal_forward(
        modal_net,
        y,
        alpha,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        y_transform=y_transform,
        y_scale=y_scale,
        ymax=ymax,
        x_mean=x_mean,
        x_std=x_std,
        modal_mean=modal_mean,
        modal_std=modal_std,
    )

    spectral = spectral_forward(
        spectral_net,
        alpha,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        x_mean=x_mean,
        x_std=x_std,
        spec_mean=spec_mean,
        spec_std=spec_std,
    )

    p_real = modal[:, 0]
    p_imag = modal[:, 1]
    q_real = modal[:, 2]
    q_imag = modal[:, 3]

    c_real = spectral[:, 0]
    c_imag = spectral[:, 1]

    def derivative(value):
        return torch.autograd.grad(
            value.sum(),
            y,
            create_graph=create_graph,
            retain_graph=True,
        )[0]

    dp_real_dy = derivative(p_real)
    dp_imag_dy = derivative(p_imag)
    dq_real_dy = derivative(q_real)
    dq_imag_dy = derivative(q_imag)

    q_real_scale = torch.clamp(
        modal_std[2],
        min=1.0e-12,
    )
    q_imag_scale = torch.clamp(
        modal_std[3],
        min=1.0e-12,
    )

    # R1 = p_y - q
    r1_real = (
        dp_real_dy - q_real
    ) / q_real_scale

    r1_imag = (
        dp_imag_dy - q_imag
    ) / q_imag_scale

    qdef_mse = torch.mean(
        r1_real.square()
        + r1_imag.square()
    )

    # U = tanh(y)
    U = torch.tanh(y)
    U_y = 1.0 - U.square()

    # D = U-c = (U-cr) - i ci
    D_real = U - c_real
    D_imag = -c_imag

    denominator = (
        D_real.square()
        + D_imag.square()
        + 1.0e-12
    )

    inv_D_real = D_real / denominator
    inv_D_imag = -D_imag / denominator

    # q / D
    q_over_D_real = (
        q_real * inv_D_real
        - q_imag * inv_D_imag
    )

    q_over_D_imag = (
        q_real * inv_D_imag
        + q_imag * inv_D_real
    )

    shear_real = (
        2.0 * U_y * q_over_D_real
    )

    shear_imag = (
        2.0 * U_y * q_over_D_imag
    )

    # D^2
    D2_real = (
        D_real.square()
        - D_imag.square()
    )

    D2_imag = (
        2.0 * D_real * D_imag
    )

    # B = 1 - M^2 D^2
    B_real = (
        1.0
        - Mach**2 * D2_real
    )

    B_imag = (
        -Mach**2 * D2_imag
    )

    # B p
    Bp_real = (
        B_real * p_real
        - B_imag * p_imag
    )

    Bp_imag = (
        B_real * p_imag
        + B_imag * p_real
    )

    alpha_squared = alpha.square()

    # R2 = q_y - 2 Uy/D q - alpha^2 B p
    r2_real = (
        dq_real_dy
        - shear_real
        - alpha_squared * Bp_real
    ) / q_real_scale

    r2_imag = (
        dq_imag_dy
        - shear_imag
        - alpha_squared * Bp_imag
    ) / q_imag_scale

    ode_mse = torch.mean(
        r2_real.square()
        + r2_imag.square()
    )

    diagnostics = {
        "qdef_rms": torch.sqrt(
            qdef_mse.detach()
        ),
        "ode_rms": torch.sqrt(
            ode_mse.detach()
        ),
        "min_abs_U_minus_c": torch.sqrt(
            denominator.detach()
        ).min(),
        "min_ci": c_imag.detach().min(),
    }

    return qdef_mse, ode_mse, diagnostics


def sample_alpha(
    n,
    alpha_min,
    alpha_max,
    edge_fraction,
    edge_width,
    device,
):
    n_edge = min(
        max(round(edge_fraction * n), 0),
        n,
    )
    n_bulk = n - n_edge

    values = []

    if n_bulk:
        values.append(
            alpha_min
            + (alpha_max - alpha_min)
            * torch.rand(
                n_bulk,
                device=device,
            )
        )

    if n_edge:
        n_left = n_edge // 2
        n_right = n_edge - n_left

        width = min(
            edge_width,
            0.5 * (
                alpha_max - alpha_min
            ),
        )

        if n_left:
            values.append(
                alpha_min
                + width
                * torch.rand(
                    n_left,
                    device=device,
                )
            )

        if n_right:
            values.append(
                alpha_max
                - width
                * torch.rand(
                    n_right,
                    device=device,
                )
            )

    alpha = torch.cat(values)

    return alpha[
        torch.randperm(
            alpha.numel(),
            device=device,
        )
    ]


def sample_y(
    n,
    full_ymax,
    core_fraction,
    core_ymax,
    device,
):
    n_core = min(
        max(round(core_fraction * n), 0),
        n,
    )
    n_full = n - n_core

    values = []

    if n_core:
        values.append(
            2.0
            * core_ymax
            * torch.rand(
                n_core,
                device=device,
            )
            - core_ymax
        )

    if n_full:
        values.append(
            2.0
            * full_ymax
            * torch.rand(
                n_full,
                device=device,
            )
            - full_ymax
        )

    y = torch.cat(values)

    return y[
        torch.randperm(
            y.numel(),
            device=device,
        )
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=2500,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--collocation-batch-size",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5.0e-6,
    )

    parser.add_argument(
        "--qdef-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--qdef-warmup-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--ode-weight",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--ode-warmup-steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--collocation-ymax",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--core-fraction",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--core-ymax",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--edge-fraction",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--edge-width",
        type=float,
        default=0.0125,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--eval-data-size",
        type=int,
        default=32768,
    )
    parser.add_argument(
        "--eval-collocation-size",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=6543,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    seed_all(args.seed)
    args.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if (
            args.device == "cuda"
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    print("[setup] device:", device)

    source = torch.load(
        args.source_checkpoint,
        map_location=device,
    )

    config = dict(source["config"])
    dataset_path = Path(config["dataset"])

    if not dataset_path.exists():
        raise FileNotFoundError(
            dataset_path
        )

    data = np.load(
        dataset_path,
        allow_pickle=True,
    )

    width = int(config["width"])
    depth = int(config["depth"])
    activation = str(
        config["activation"]
    )

    modal_net = MLP(
        2,
        4,
        width,
        depth,
        activation,
    ).to(device)

    spectral_net = MLP(
        1,
        2,
        max(64, width // 2),
        max(3, depth - 1),
        activation,
    ).to(device)

    modal_net.load_state_dict(
        source["modal_net"]
    )

    spectral_net.load_state_dict(
        source["spectral_net"]
    )

    # Première passe conservative :
    # c(alpha) est fixé pendant le fine-tuning ODE.
    spectral_net.eval()

    for parameter in spectral_net.parameters():
        parameter.requires_grad_(False)

    normalization = config["normalization"]

    def as_tensor(value):
        return torch.tensor(
            value,
            dtype=torch.float32,
            device=device,
        )

    x_mean = as_tensor(
        normalization["x_mean"]
    )
    x_std = as_tensor(
        normalization["x_std"]
    )

    modal_mean = as_tensor(
        normalization["modal_mean"]
    )
    modal_std = as_tensor(
        normalization["modal_std"]
    )

    spec_mean = as_tensor(
        normalization["spec_mean"]
    )
    spec_std = as_tensor(
        normalization["spec_std"]
    )

    alpha_min = float(
        config["alpha_min"]
    )
    alpha_max = float(
        config["alpha_max"]
    )

    Mach = float(
        config.get(
            "Mach_fixed",
            np.asarray(
                data["Mach_fixed"]
            ).item(),
        )
    )

    y_scale = float(
        config.get(
            "y_scale",
            10.0,
        )
    )

    y_transform = str(
        config.get(
            "y_transform",
            "tanh",
        )
    )

    alpha_np = np.asarray(
        data["row_alpha"],
        dtype=np.float32,
    )

    y_np = np.asarray(
        data["y"],
        dtype=np.float32,
    )

    target_np = np.column_stack(
        [
            data["p_real"],
            data["p_imag"],
            data["q_real"],
            data["q_imag"],
        ]
    ).astype(np.float32)

    ymax = max(
        float(np.max(np.abs(y_np))),
        1.0,
    )

    alpha_data = torch.from_numpy(
        alpha_np
    ).to(device)

    y_data = torch.from_numpy(
        y_np
    ).to(device)

    modal_target = torch.from_numpy(
        target_np
    ).to(device)

    modal_target_normalized = (
        modal_target - modal_mean
    ) / modal_std

    n_rows = alpha_data.numel()

    eval_indices = torch.randperm(
        n_rows,
        device=device,
    )[
        :min(
            args.eval_data_size,
            n_rows,
        )
    ]

    eval_alpha = sample_alpha(
        args.eval_collocation_size,
        alpha_min,
        alpha_max,
        args.edge_fraction,
        args.edge_width,
        device,
    )

    eval_y = sample_y(
        args.eval_collocation_size,
        args.collocation_ymax,
        args.core_fraction,
        args.core_ymax,
        device,
    )

    config.update(
        {
            "run_dir": str(
                args.run_dir
            ),
            "source_checkpoint": str(
                args.source_checkpoint
            ),
            "finetune_kind": (
                "qdef_plus_compressible_pressure_ode"
            ),
            "pressure_ode": (
                "q_y - 2*U_y/(U-c)*q "
                "- alpha^2*(1-M^2*(U-c)^2)*p = 0"
            ),
            "base_flow": "U=tanh(y)",
            "spectral_network_frozen": True,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "collocation_batch_size": (
                args.collocation_batch_size
            ),
            "lr": args.lr,
            "qdef_weight": (
                args.qdef_weight
            ),
            "ode_weight": (
                args.ode_weight
            ),
            "collocation_ymax": (
                args.collocation_ymax
            ),
            "core_fraction": (
                args.core_fraction
            ),
            "core_ymax": args.core_ymax,
            "edge_fraction": (
                args.edge_fraction
            ),
            "edge_width": args.edge_width,
            "seed": args.seed,
        }
    )

    (args.run_dir / "config.json").write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    optimizer = torch.optim.AdamW(
        modal_net.parameters(),
        lr=args.lr,
        weight_decay=1.0e-8,
    )

    def evaluate():
        modal_net.eval()

        with torch.no_grad():
            predicted_normalized, _ = modal_forward(
                modal_net,
                y_data[eval_indices],
                alpha_data[eval_indices],
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                y_transform=y_transform,
                y_scale=y_scale,
                ymax=ymax,
                x_mean=x_mean,
                x_std=x_std,
                modal_mean=modal_mean,
                modal_std=modal_std,
            )

            modal_mse = torch.mean(
                (
                    predicted_normalized
                    - modal_target_normalized[
                        eval_indices
                    ]
                ).square()
            )

        qdef_mse, ode_mse, diagnostics = (
            physics_losses(
                modal_net,
                spectral_net,
                eval_y.detach().clone(),
                eval_alpha,
                Mach=Mach,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                y_transform=y_transform,
                y_scale=y_scale,
                ymax=ymax,
                x_mean=x_mean,
                x_std=x_std,
                modal_mean=modal_mean,
                modal_std=modal_std,
                spec_mean=spec_mean,
                spec_std=spec_std,
                create_graph=False,
            )
        )

        result = {
            "modal_mse": float(
                modal_mse.cpu()
            ),
            "qdef_mse": float(
                qdef_mse.detach().cpu()
            ),
            "ode_mse": float(
                ode_mse.detach().cpu()
            ),
        }

        result.update(
            {
                name: float(value.cpu())
                for name, value
                in diagnostics.items()
            }
        )

        modal_net.train()
        return result

    def objective(metrics):
        return (
            metrics["modal_mse"]
            + args.qdef_weight
            * metrics["qdef_mse"]
            + args.ode_weight
            * metrics["ode_mse"]
        )

    def save_checkpoint(value):
        torch.save(
            {
                "modal_net": (
                    modal_net.state_dict()
                ),
                "spectral_net": (
                    spectral_net.state_dict()
                ),
                "config": config,
                "best_val": value,
            },
            args.run_dir
            / "best_model.pt",
        )

    initial = evaluate()
    best_objective = objective(initial)

    save_checkpoint(best_objective)

    history = [
        {
            "step": 0,
            "lambda_qdef": 0.0,
            "lambda_ode": 0.0,
            "eval_objective": (
                best_objective
            ),
            **{
                f"eval_{key}": value
                for key, value
                in initial.items()
            },
        }
    ]

    print(
        "[step 00000] "
        f"modal={initial['modal_mse']:.6e} "
        f"qdef={initial['qdef_mse']:.6e} "
        f"ode={initial['ode_mse']:.6e} "
        f"min|U-c|="
        f"{initial['min_abs_U_minus_c']:.6e}"
    )

    for step in range(
        1,
        args.steps + 1,
    ):
        data_indices = torch.randint(
            0,
            n_rows,
            (args.batch_size,),
            device=device,
        )

        predicted_normalized, _ = modal_forward(
            modal_net,
            y_data[data_indices],
            alpha_data[data_indices],
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            y_transform=y_transform,
            y_scale=y_scale,
            ymax=ymax,
            x_mean=x_mean,
            x_std=x_std,
            modal_mean=modal_mean,
            modal_std=modal_std,
        )

        modal_loss = torch.mean(
            (
                predicted_normalized
                - modal_target_normalized[
                    data_indices
                ]
            ).square()
        )

        collocation_alpha = sample_alpha(
            args.collocation_batch_size,
            alpha_min,
            alpha_max,
            args.edge_fraction,
            args.edge_width,
            device,
        )

        collocation_y = sample_y(
            args.collocation_batch_size,
            args.collocation_ymax,
            args.core_fraction,
            args.core_ymax,
            device,
        )

        (
            qdef_loss,
            ode_loss,
            train_diagnostics,
        ) = physics_losses(
            modal_net,
            spectral_net,
            collocation_y,
            collocation_alpha,
            Mach=Mach,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            y_transform=y_transform,
            y_scale=y_scale,
            ymax=ymax,
            x_mean=x_mean,
            x_std=x_std,
            modal_mean=modal_mean,
            modal_std=modal_std,
            spec_mean=spec_mean,
            spec_std=spec_std,
            create_graph=True,
        )

        lambda_qdef = (
            args.qdef_weight
            * min(
                1.0,
                step
                / max(
                    1,
                    args.qdef_warmup_steps,
                ),
            )
        )

        lambda_ode = (
            args.ode_weight
            * min(
                1.0,
                step
                / max(
                    1,
                    args.ode_warmup_steps,
                ),
            )
        )

        total_loss = (
            modal_loss
            + lambda_qdef * qdef_loss
            + lambda_ode * ode_loss
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            modal_net.parameters(),
            max_norm=10.0,
        )

        optimizer.step()

        if (
            step == 1
            or step % args.eval_every == 0
            or step == args.steps
        ):
            evaluated = evaluate()
            eval_objective = objective(
                evaluated
            )

            row = {
                "step": step,
                "lambda_qdef": (
                    lambda_qdef
                ),
                "lambda_ode": (
                    lambda_ode
                ),
                "train_modal_mse": float(
                    modal_loss.detach().cpu()
                ),
                "train_qdef_mse": float(
                    qdef_loss.detach().cpu()
                ),
                "train_ode_mse": float(
                    ode_loss.detach().cpu()
                ),
                "train_total": float(
                    total_loss.detach().cpu()
                ),
                "train_min_abs_U_minus_c": (
                    float(
                        train_diagnostics[
                            "min_abs_U_minus_c"
                        ].cpu()
                    )
                ),
                "eval_objective": (
                    eval_objective
                ),
                **{
                    f"eval_{key}": value
                    for key, value
                    in evaluated.items()
                },
            }

            history.append(row)

            pd.DataFrame(
                history
            ).to_csv(
                args.run_dir / "history.csv",
                index=False,
            )

            print(
                f"[step {step:05d}] "
                f"modal="
                f"{evaluated['modal_mse']:.6e} "
                f"qdef="
                f"{evaluated['qdef_mse']:.6e} "
                f"ode="
                f"{evaluated['ode_mse']:.6e} "
                f"lambda_q="
                f"{lambda_qdef:.3e} "
                f"lambda_ode="
                f"{lambda_ode:.3e}"
            )

            if (
                eval_objective
                < best_objective
            ):
                best_objective = (
                    eval_objective
                )
                save_checkpoint(
                    best_objective
                )

    best_checkpoint = torch.load(
        args.run_dir / "best_model.pt",
        map_location=device,
    )

    modal_net.load_state_dict(
        best_checkpoint["modal_net"]
    )

    final = evaluate()

    physics_metrics = {
        "source_checkpoint": str(
            args.source_checkpoint
        ),
        "best_checkpoint": str(
            args.run_dir
            / "best_model.pt"
        ),
        "Mach_fixed": Mach,
        "spectral_network_frozen": True,
        "qdef_weight": (
            args.qdef_weight
        ),
        "ode_weight": (
            args.ode_weight
        ),
        "initial": initial,
        "final": final,
        "relative_change": {
            key: (
                final[key] - initial[key]
            )
            / max(
                abs(initial[key]),
                1.0e-30,
            )
            for key in (
                "modal_mse",
                "qdef_mse",
                "ode_mse",
            )
        },
    }

    (
        args.run_dir
        / "physics_metrics.json"
    ).write_text(
        json.dumps(
            physics_metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "[done] best objective:",
        best_objective,
    )
    print(
        "[done] checkpoint:",
        args.run_dir / "best_model.pt",
    )
    print(
        "[done] physics metrics:",
        args.run_dir
        / "physics_metrics.json",
    )


if __name__ == "__main__":
    main()
