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
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int, activation: str = "silu"):
        super().__init__()

        if activation == "silu":
            act = nn.SiLU
        elif activation == "tanh":
            act = nn.Tanh
        elif activation == "gelu":
            act = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(act())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(act())

        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FixedMPQRows(Dataset):
    def __init__(self, x: np.ndarray, target: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.target = torch.as_tensor(target, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.target[i]


def normalize(arr, mean=None, std=None, eps=1e-12):
    arr = np.asarray(arr, dtype=np.float64)
    if mean is None:
        mean = arr.mean(axis=0)
    if std is None:
        std = arr.std(axis=0)
    std = np.maximum(std, eps)
    return (arr - mean) / std, mean, std


def rel_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.linalg.norm(y_pred - y_true)
    den = np.linalg.norm(y_true) + 1e-30
    return float(num / den)


@torch.no_grad()
def evaluate_modal(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = torch.mean((pred - yb) ** 2)

        bs = xb.shape[0]
        total += float(loss.item()) * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def evaluate_spectral(model: nn.Module, alpha_x: torch.Tensor, target: torch.Tensor, device: torch.device):
    model.eval()
    alpha_x = alpha_x.to(device)
    target = target.to(device)

    pred = model(alpha_x)
    loss = torch.mean((pred - target) ** 2)
    return float(loss.item())


def main():
    ap = argparse.ArgumentParser(
        description="Train first supervised fixed-M supersonic PINN chart with spectral and p/q modal anchors."
    )

    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)

    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-8)

    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--activation", choices=["silu", "tanh", "gelu"], default="silu")

    ap.add_argument("--spectral-weight", type=float, default=1.0)
    ap.add_argument("--modal-weight", type=float, default=1.0)

    ap.add_argument("--y-scale", type=float, default=10.0)
    ap.add_argument(
        "--y-transform",
        choices=["tanh", "asinh", "compact"],
        default="asinh",
        help=(
            "Coordinate feature for y. tanh saturates strongly on very large domains; "
            "asinh is recommended for y in [-2000,2000]."
        ),
    )
    ap.add_argument("--val-every-alpha", type=int, default=0,
                    help="If >0, hold out every k-th alpha for validation.")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    args.run_dir.mkdir(parents=True, exist_ok=True)

    print("[info] dataset:", args.dataset)
    print("[info] run dir:", args.run_dir)
    print("[info] device:", device)

    d = np.load(args.dataset, allow_pickle=True)

    Mach_fixed = float(d["Mach_fixed"])
    alpha_anchors = d["alpha_anchors"].astype(np.float64)
    cr_ref = d["cr_ref"].astype(np.float64)
    ci_ref = d["ci_ref"].astype(np.float64)

    row_alpha = d["row_alpha"].astype(np.float64)
    y = d["y"].astype(np.float64)

    p_real = d["p_real"].astype(np.float64)
    p_imag = d["p_imag"].astype(np.float64)
    q_real = d["q_real"].astype(np.float64)
    q_imag = d["q_imag"].astype(np.float64)

    alpha_min = float(alpha_anchors.min())
    alpha_max = float(alpha_anchors.max())

    def alpha_to_x(a):
        return 2.0 * (a - alpha_min) / (alpha_max - alpha_min) - 1.0

    # Bounded y feature. For the supersonic modal exports, y may reach O(10^3).
    # tanh(y/10) saturates too early and collapses the far tails.
    if args.y_transform == "tanh":
        y_feat = np.tanh(y / args.y_scale)
    elif args.y_transform == "asinh":
        ymax = max(float(np.max(np.abs(y))), 1.0)
        y_feat = np.arcsinh(y / args.y_scale) / np.arcsinh(ymax / args.y_scale)
    elif args.y_transform == "compact":
        y_feat = y / (args.y_scale + np.abs(y))
    else:
        raise RuntimeError(f"Unknown y_transform: {args.y_transform}")

    alpha_feat = alpha_to_x(row_alpha)

    x_modal = np.stack([y_feat, alpha_feat], axis=1)
    target_modal = np.stack([p_real, p_imag, q_real, q_imag], axis=1)

    alpha_spec_x = alpha_to_x(alpha_anchors)[:, None]
    target_spec = np.stack([cr_ref, ci_ref], axis=1)

    x_modal_n, x_mean, x_std = normalize(x_modal)
    target_modal_n, modal_mean, modal_std = normalize(target_modal)
    alpha_spec_x_n, alpha_mean, alpha_std = normalize(alpha_spec_x, mean=x_mean[1:2], std=x_std[1:2])
    target_spec_n, spec_mean, spec_std = normalize(target_spec)

    # Train/val split by alpha, optional.
    if args.val_every_alpha and args.val_every_alpha > 0:
        alpha_sorted = np.array(sorted(np.unique(np.round(alpha_anchors, 12))))
        val_alphas = set(alpha_sorted[::args.val_every_alpha])
        val_mask = np.array([round(float(a), 12) in val_alphas for a in row_alpha])
        train_mask = ~val_mask
        print(f"[info] validation alphas: {len(val_alphas)} / {len(alpha_sorted)}")
    else:
        train_mask = np.ones(len(row_alpha), dtype=bool)
        val_mask = np.zeros(len(row_alpha), dtype=bool)

    train_ds = FixedMPQRows(x_modal_n[train_mask], target_modal_n[train_mask])

    if val_mask.any():
        val_ds = FixedMPQRows(x_modal_n[val_mask], target_modal_n[val_mask])
    else:
        val_ds = FixedMPQRows(x_modal_n[train_mask], target_modal_n[train_mask])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    modal_net = MLP(
        in_dim=2,
        out_dim=4,
        width=args.width,
        depth=args.depth,
        activation=args.activation,
    ).to(device)

    spectral_net = MLP(
        in_dim=1,
        out_dim=2,
        width=max(64, args.width // 2),
        depth=max(3, args.depth - 1),
        activation=args.activation,
    ).to(device)

    opt = torch.optim.AdamW(
        list(modal_net.parameters()) + list(spectral_net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    alpha_spec_t = torch.as_tensor(alpha_spec_x_n, dtype=torch.float32)
    target_spec_t = torch.as_tensor(target_spec_n, dtype=torch.float32)

    history = []

    best_val = math.inf
    best_path = args.run_dir / "best_model.pt"

    config = {
        "dataset": str(args.dataset),
        "run_dir": str(args.run_dir),
        "Mach_fixed": Mach_fixed,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "width": args.width,
        "depth": args.depth,
        "activation": args.activation,
        "spectral_weight": args.spectral_weight,
        "modal_weight": args.modal_weight,
        "y_scale": args.y_scale,
        "y_transform": args.y_transform,
        "val_every_alpha": args.val_every_alpha,
        "seed": args.seed,
        "normalization": {
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "modal_mean": modal_mean.tolist(),
            "modal_std": modal_std.tolist(),
            "spec_mean": spec_mean.tolist(),
            "spec_std": spec_std.tolist(),
        },
    }

    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("[info] n train rows:", len(train_ds))
    print("[info] n val rows:", len(val_ds))
    print("[info] n spectral anchors:", len(alpha_anchors))

    for epoch in range(1, args.epochs + 1):
        modal_net.train()
        spectral_net.train()

        epoch_modal = 0.0
        epoch_spec = 0.0
        epoch_total = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)

            modal_pred = modal_net(xb)
            modal_loss = torch.mean((modal_pred - yb) ** 2)

            spec_pred = spectral_net(alpha_spec_t.to(device))
            spec_loss = torch.mean((spec_pred - target_spec_t.to(device)) ** 2)

            loss = args.modal_weight * modal_loss + args.spectral_weight * spec_loss
            loss.backward()
            opt.step()

            bs = xb.shape[0]
            epoch_modal += float(modal_loss.item()) * bs
            epoch_spec += float(spec_loss.item()) * bs
            epoch_total += float(loss.item()) * bs
            n_seen += bs

        epoch_modal /= max(n_seen, 1)
        epoch_spec /= max(n_seen, 1)
        epoch_total /= max(n_seen, 1)

        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            val_modal = evaluate_modal(modal_net, val_loader, device)
            val_spec = evaluate_spectral(spectral_net, alpha_spec_t, target_spec_t, device)
            val_total = args.modal_weight * val_modal + args.spectral_weight * val_spec

            row = {
                "epoch": epoch,
                "train_total": epoch_total,
                "train_modal": epoch_modal,
                "train_spec": epoch_spec,
                "val_total": val_total,
                "val_modal": val_modal,
                "val_spec": val_spec,
                "lr": opt.param_groups[0]["lr"],
            }
            history.append(row)

            print(
                f"[epoch {epoch:06d}] "
                f"train={epoch_total:.4e} "
                f"modal={epoch_modal:.4e} "
                f"spec={epoch_spec:.4e} "
                f"val={val_total:.4e}"
            )

            pd.DataFrame(history).to_csv(args.run_dir / "history.csv", index=False)

            if val_total < best_val:
                best_val = val_total
                torch.save(
                    {
                        "modal_net": modal_net.state_dict(),
                        "spectral_net": spectral_net.state_dict(),
                        "config": config,
                        "best_val": best_val,
                    },
                    best_path,
                )

    # Final evaluation in physical units on all rows and all spectral anchors.
    modal_net.eval()
    spectral_net.eval()

    with torch.no_grad():
        x_all_t = torch.as_tensor(x_modal_n, dtype=torch.float32, device=device)
        modal_pred_n = modal_net(x_all_t).cpu().numpy()
        modal_pred = modal_pred_n * modal_std + modal_mean

        alpha_t = torch.as_tensor(alpha_spec_x_n, dtype=torch.float32, device=device)
        spec_pred_n = spectral_net(alpha_t).cpu().numpy()
        spec_pred = spec_pred_n * spec_std + spec_mean

    p_pred = modal_pred[:, 0] + 1j * modal_pred[:, 1]
    q_pred = modal_pred[:, 2] + 1j * modal_pred[:, 3]

    p_ref = p_real + 1j * p_imag
    q_ref = q_real + 1j * q_imag

    metrics = {
        "best_val": float(best_val),
        "global_p_rel_l2": rel_l2(p_ref, p_pred),
        "global_q_rel_l2": rel_l2(q_ref, q_pred),
        "cr_abs_mean": float(np.mean(np.abs(spec_pred[:, 0] - cr_ref))),
        "cr_abs_max": float(np.max(np.abs(spec_pred[:, 0] - cr_ref))),
        "ci_abs_mean": float(np.mean(np.abs(spec_pred[:, 1] - ci_ref))),
        "ci_abs_max": float(np.max(np.abs(spec_pred[:, 1] - ci_ref))),
    }

    print("[done] metrics")
    print(json.dumps(metrics, indent=2))

    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_spec_df = pd.DataFrame({
        "Mach": Mach_fixed,
        "alpha": alpha_anchors,
        "cr_ref": cr_ref,
        "ci_ref": ci_ref,
        "cr_pred": spec_pred[:, 0],
        "ci_pred": spec_pred[:, 1],
        "cr_abs_err": np.abs(spec_pred[:, 0] - cr_ref),
        "ci_abs_err": np.abs(spec_pred[:, 1] - ci_ref),
    })
    pred_spec_df.to_csv(args.run_dir / "spectral_predictions.csv", index=False)

    np.savez_compressed(
        args.run_dir / "modal_predictions_all_rows.npz",
        Mach_fixed=np.array(Mach_fixed),
        row_alpha=row_alpha,
        y=y,
        p_ref_real=p_real,
        p_ref_imag=p_imag,
        q_ref_real=q_real,
        q_ref_imag=q_imag,
        p_pred_real=modal_pred[:, 0],
        p_pred_imag=modal_pred[:, 1],
        q_pred_real=modal_pred[:, 2],
        q_pred_imag=modal_pred[:, 3],
    )

    print("[done] wrote:")
    print(" ", best_path)
    print(" ", args.run_dir / "history.csv")
    print(" ", args.run_dir / "metrics.json")
    print(" ", args.run_dir / "spectral_predictions.csv")
    print(" ", args.run_dir / "modal_predictions_all_rows.npz")


if __name__ == "__main__":
    main()
