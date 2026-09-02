#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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

        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def rel_l2(ref, pred):
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-30))


def batched_predict(model, x, device, batch_size=65536):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=65536)
    args = ap.parse_args()

    run_dir = args.run_dir
    ckpt_path = args.checkpoint or (run_dir / "best_model.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    dataset_path = Path(config["dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    d = np.load(dataset_path, allow_pickle=True)

    width = int(config["width"])
    depth = int(config["depth"])
    activation = config["activation"]

    modal_net = MLP(2, 4, width, depth, activation).to(device)
    spectral_net = MLP(1, 2, max(64, width // 2), max(3, depth - 1), activation).to(device)

    modal_net.load_state_dict(ckpt["modal_net"])
    spectral_net.load_state_dict(ckpt["spectral_net"])

    norm = config["normalization"]
    x_mean = np.array(norm["x_mean"], dtype=np.float64)
    x_std = np.array(norm["x_std"], dtype=np.float64)
    modal_mean = np.array(norm["modal_mean"], dtype=np.float64)
    modal_std = np.array(norm["modal_std"], dtype=np.float64)
    spec_mean = np.array(norm["spec_mean"], dtype=np.float64)
    spec_std = np.array(norm["spec_std"], dtype=np.float64)

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

    alpha_min = float(config["alpha_min"])
    alpha_max = float(config["alpha_max"])
    y_scale = float(config.get("y_scale", 10.0))
    y_transform = config.get("y_transform", "tanh")

    def alpha_to_x(a):
        return 2.0 * (a - alpha_min) / (alpha_max - alpha_min) - 1.0

    if y_transform == "tanh":
        y_feat = np.tanh(y / y_scale)
    elif y_transform == "asinh":
        ymax = max(float(np.max(np.abs(y))), 1.0)
        y_feat = np.arcsinh(y / y_scale) / np.arcsinh(ymax / y_scale)
    elif y_transform == "compact":
        y_feat = y / (y_scale + np.abs(y))
    else:
        raise RuntimeError(f"Unknown y_transform: {y_transform}")

    alpha_feat = alpha_to_x(row_alpha)
    x_modal = np.stack([y_feat, alpha_feat], axis=1)
    x_modal_n = (x_modal - x_mean) / x_std

    alpha_spec_x = alpha_to_x(alpha_anchors)[:, None]
    alpha_spec_x_n = (alpha_spec_x - x_mean[1:2]) / x_std[1:2]

    modal_pred_n = batched_predict(modal_net, x_modal_n, device, batch_size=args.batch_size)
    modal_pred = modal_pred_n * modal_std + modal_mean

    spec_pred_n = batched_predict(spectral_net, alpha_spec_x_n, device, batch_size=args.batch_size)
    spec_pred = spec_pred_n * spec_std + spec_mean

    p_ref = p_real + 1j * p_imag
    q_ref = q_real + 1j * q_imag
    p_pred = modal_pred[:, 0] + 1j * modal_pred[:, 1]
    q_pred = modal_pred[:, 2] + 1j * modal_pred[:, 3]

    metrics = {
        "checkpoint": str(ckpt_path),
        "best_val": float(ckpt.get("best_val", np.nan)),
        "global_p_rel_l2": rel_l2(p_ref, p_pred),
        "global_q_rel_l2": rel_l2(q_ref, q_pred),
        "cr_abs_mean": float(np.mean(np.abs(spec_pred[:, 0] - cr_ref))),
        "cr_abs_max": float(np.max(np.abs(spec_pred[:, 0] - cr_ref))),
        "ci_abs_mean": float(np.mean(np.abs(spec_pred[:, 1] - ci_ref))),
        "ci_abs_max": float(np.max(np.abs(spec_pred[:, 1] - ci_ref))),
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pd.DataFrame({
        "Mach": Mach_fixed,
        "alpha": alpha_anchors,
        "cr_ref": cr_ref,
        "ci_ref": ci_ref,
        "cr_pred": spec_pred[:, 0],
        "ci_pred": spec_pred[:, 1],
        "cr_abs_err": np.abs(spec_pred[:, 0] - cr_ref),
        "ci_abs_err": np.abs(spec_pred[:, 1] - ci_ref),
    }).to_csv(run_dir / "spectral_predictions.csv", index=False)

    np.savez_compressed(
        run_dir / "modal_predictions_all_rows.npz",
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

    print(json.dumps(metrics, indent=2))
    print("[done] wrote metrics/predictions in", run_dir)


if __name__ == "__main__":
    main()
