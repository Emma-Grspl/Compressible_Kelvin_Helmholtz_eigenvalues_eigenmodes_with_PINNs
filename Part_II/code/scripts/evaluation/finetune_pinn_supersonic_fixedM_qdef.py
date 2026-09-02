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

        act = activations[activation]

        layers = [nn.Linear(in_dim, width), act()]

        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]

        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def alpha_feature(alpha, alpha_min, alpha_max):
    return (
        2.0 * (alpha - alpha_min) / (alpha_max - alpha_min)
        - 1.0
    )


def y_feature(y, transform, scale, ymax):
    if transform == "tanh":
        return torch.tanh(y / scale)

    if transform == "asinh":
        denominator = math.asinh(ymax / scale)
        return torch.asinh(y / scale) / denominator

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
            y_feature(y, y_transform, y_scale, ymax),
            alpha_feature(alpha, alpha_min, alpha_max),
        ],
        dim=1,
    )

    normalized_features = (features - x_mean) / x_std
    normalized_output = model(normalized_features)
    physical_output = normalized_output * modal_std + modal_mean

    return normalized_output, physical_output


def qdef_loss(
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
    create_graph,
):
    if not y.requires_grad:
        y = y.detach().clone().requires_grad_(True)

    _, output = modal_forward(
        model,
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

    p_real = output[:, 0]
    p_imag = output[:, 1]
    q_real = output[:, 2]
    q_imag = output[:, 3]

    dp_real_dy = torch.autograd.grad(
        p_real.sum(),
        y,
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    dp_imag_dy = torch.autograd.grad(
        p_imag.sum(),
        y,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]

    q_real_scale = torch.clamp(modal_std[2], min=1.0e-12)
    q_imag_scale = torch.clamp(modal_std[3], min=1.0e-12)

    residual_real = (dp_real_dy - q_real) / q_real_scale
    residual_imag = (dp_imag_dy - q_imag) / q_imag_scale

    return torch.mean(
        residual_real**2 + residual_imag**2
    )


def sample_alpha(
    n,
    alpha_min,
    alpha_max,
    edge_fraction,
    edge_width,
    device,
):
    """
    Most points are uniform on the full alpha interval.
    A fraction is oversampled near alpha_min and alpha_max because
    the audit found the largest qdef errors near the two edges.
    """
    n_edge = int(round(edge_fraction * n))
    n_edge = min(max(n_edge, 0), n)
    n_bulk = n - n_edge

    values = []

    if n_bulk > 0:
        values.append(
            alpha_min
            + (alpha_max - alpha_min)
            * torch.rand(n_bulk, device=device)
        )

    if n_edge > 0:
        n_left = n_edge // 2
        n_right = n_edge - n_left

        width = min(
            edge_width,
            0.5 * (alpha_max - alpha_min),
        )

        if n_left > 0:
            values.append(
                alpha_min
                + width * torch.rand(n_left, device=device)
            )

        if n_right > 0:
            values.append(
                alpha_max
                - width * torch.rand(n_right, device=device)
            )

    alpha = torch.cat(values)

    permutation = torch.randperm(
        alpha.numel(),
        device=device,
    )

    return alpha[permutation]


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

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--collocation-batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument(
        "--qdef-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--qdef-warmup-steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--collocation-ymax",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--edge-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--edge-width",
        type=float,
        default=0.01,
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
        default=8192,
    )

    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    set_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    print("[setup] device:", device)

    checkpoint = torch.load(
        args.source_checkpoint,
        map_location=device,
    )

    config = dict(checkpoint["config"])
    dataset_path = Path(config["dataset"])

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    data = np.load(dataset_path, allow_pickle=True)

    width = int(config["width"])
    depth = int(config["depth"])
    activation = str(config["activation"])

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

    modal_net.load_state_dict(checkpoint["modal_net"])
    spectral_net.load_state_dict(checkpoint["spectral_net"])

    # Le réseau spectral est gelé : cr et ci ne doivent pas dériver
    # pendant ce fine-tuning uniquement consacré à p et q.
    spectral_net.eval()

    for parameter in spectral_net.parameters():
        parameter.requires_grad_(False)

    normalization = config["normalization"]

    x_mean = torch.tensor(
        normalization["x_mean"],
        dtype=torch.float32,
        device=device,
    )
    x_std = torch.tensor(
        normalization["x_std"],
        dtype=torch.float32,
        device=device,
    )
    modal_mean = torch.tensor(
        normalization["modal_mean"],
        dtype=torch.float32,
        device=device,
    )
    modal_std = torch.tensor(
        normalization["modal_std"],
        dtype=torch.float32,
        device=device,
    )

    alpha_min = float(config["alpha_min"])
    alpha_max = float(config["alpha_max"])
    y_scale = float(config.get("y_scale", 10.0))
    y_transform = str(config.get("y_transform", "tanh"))

    row_alpha_np = np.asarray(
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

    row_alpha = torch.from_numpy(row_alpha_np).to(device)
    row_y = torch.from_numpy(y_np).to(device)
    modal_target = torch.from_numpy(target_np).to(device)

    modal_target_normalized = (
        modal_target - modal_mean
    ) / modal_std

    n_rows = len(row_alpha_np)

    eval_indices = torch.randperm(
        n_rows,
        device=device,
    )[: min(args.eval_data_size, n_rows)]

    # Collocation fixe pour comparer les checkpoints.
    eval_alpha = sample_alpha(
        args.eval_collocation_size,
        alpha_min,
        alpha_max,
        args.edge_fraction,
        args.edge_width,
        device,
    )

    eval_y = (
        2.0
        * args.collocation_ymax
        * torch.rand(
            args.eval_collocation_size,
            device=device,
        )
        - args.collocation_ymax
    )

    optimizer = torch.optim.AdamW(
        modal_net.parameters(),
        lr=args.lr,
        weight_decay=1.0e-8,
    )

    config.update(
        {
            "run_dir": str(args.run_dir),
            "source_checkpoint": str(args.source_checkpoint),
            "finetune_kind": (
                "qdef_on_unreferenced_collocation"
            ),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "collocation_batch_size": (
                args.collocation_batch_size
            ),
            "lr": args.lr,
            "qdef_weight": args.qdef_weight,
            "qdef_warmup_steps": (
                args.qdef_warmup_steps
            ),
            "collocation_alpha_min": alpha_min,
            "collocation_alpha_max": alpha_max,
            "collocation_y_min": (
                -args.collocation_ymax
            ),
            "collocation_y_max": (
                args.collocation_ymax
            ),
            "edge_fraction": args.edge_fraction,
            "edge_width": args.edge_width,
            "spectral_network_frozen": True,
            "seed": args.seed,
        }
    )

    (args.run_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    def evaluate():
        modal_net.eval()

        with torch.no_grad():
            predicted_normalized, _ = modal_forward(
                modal_net,
                row_y[eval_indices],
                row_alpha[eval_indices],
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
                    - modal_target_normalized[eval_indices]
                )
                ** 2
            )

        derivative_mse = qdef_loss(
            modal_net,
            eval_y.detach().clone().requires_grad_(True),
            eval_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            y_transform=y_transform,
            y_scale=y_scale,
            ymax=ymax,
            x_mean=x_mean,
            x_std=x_std,
            modal_mean=modal_mean,
            modal_std=modal_std,
            create_graph=False,
        )

        modal_net.train()

        return (
            float(modal_mse.detach().cpu()),
            float(derivative_mse.detach().cpu()),
        )

    def save_checkpoint(best_value):
        torch.save(
            {
                "modal_net": modal_net.state_dict(),
                "spectral_net": spectral_net.state_dict(),
                "config": config,
                "best_val": best_value,
            },
            args.run_dir / "best_model.pt",
        )

    history = []

    initial_modal_mse, initial_qdef_mse = evaluate()

    best_objective = (
        initial_modal_mse
        + args.qdef_weight * initial_qdef_mse
    )

    save_checkpoint(best_objective)

    history.append(
        {
            "step": 0,
            "lambda_qdef": 0.0,
            "eval_modal_mse": initial_modal_mse,
            "eval_qdef_mse": initial_qdef_mse,
            "eval_objective": best_objective,
        }
    )

    print(
        f"[step 00000] "
        f"eval_modal={initial_modal_mse:.6e} "
        f"eval_qdef={initial_qdef_mse:.6e}"
    )

    modal_net.train()

    for step in range(1, args.steps + 1):
        # Partie supervisée : uniquement aux lignes classiques existantes.
        data_indices = torch.randint(
            low=0,
            high=n_rows,
            size=(args.batch_size,),
            device=device,
        )

        predicted_normalized, _ = modal_forward(
            modal_net,
            row_y[data_indices],
            row_alpha[data_indices],
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
                - modal_target_normalized[data_indices]
            )
            ** 2
        )

        # Partie non supervisée :
        # alpha et y sont tirés indépendamment des ancres classiques.
        collocation_alpha = sample_alpha(
            args.collocation_batch_size,
            alpha_min,
            alpha_max,
            args.edge_fraction,
            args.edge_width,
            device,
        )

        collocation_y = (
            2.0
            * args.collocation_ymax
            * torch.rand(
                args.collocation_batch_size,
                device=device,
            )
            - args.collocation_ymax
        ).requires_grad_(True)

        derivative_loss = qdef_loss(
            modal_net,
            collocation_y,
            collocation_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            y_transform=y_transform,
            y_scale=y_scale,
            ymax=ymax,
            x_mean=x_mean,
            x_std=x_std,
            modal_mean=modal_mean,
            modal_std=modal_std,
            create_graph=True,
        )

        warmup = min(
            1.0,
            step / max(1, args.qdef_warmup_steps),
        )

        lambda_qdef = args.qdef_weight * warmup

        total_loss = (
            modal_loss
            + lambda_qdef * derivative_loss
        )

        optimizer.zero_grad(set_to_none=True)
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
            eval_modal_mse, eval_qdef_mse = evaluate()

            # Toujours utiliser le poids final pour comparer les checkpoints.
            eval_objective = (
                eval_modal_mse
                + args.qdef_weight * eval_qdef_mse
            )

            row = {
                "step": step,
                "lambda_qdef": lambda_qdef,
                "train_modal_mse": float(
                    modal_loss.detach().cpu()
                ),
                "train_qdef_mse": float(
                    derivative_loss.detach().cpu()
                ),
                "train_total": float(
                    total_loss.detach().cpu()
                ),
                "eval_modal_mse": eval_modal_mse,
                "eval_qdef_mse": eval_qdef_mse,
                "eval_objective": eval_objective,
            }

            history.append(row)

            pd.DataFrame(history).to_csv(
                args.run_dir / "history.csv",
                index=False,
            )

            print(
                f"[step {step:05d}] "
                f"train_modal={row['train_modal_mse']:.6e} "
                f"train_qdef={row['train_qdef_mse']:.6e} "
                f"eval_modal={eval_modal_mse:.6e} "
                f"eval_qdef={eval_qdef_mse:.6e} "
                f"lambda={lambda_qdef:.3e}"
            )

            if eval_objective < best_objective:
                best_objective = eval_objective
                save_checkpoint(best_objective)

            modal_net.train()

    print("[done] best objective:", best_objective)
    print(
        "[done] checkpoint:",
        args.run_dir / "best_model.pt",
    )


if __name__ == "__main__":
    main()
