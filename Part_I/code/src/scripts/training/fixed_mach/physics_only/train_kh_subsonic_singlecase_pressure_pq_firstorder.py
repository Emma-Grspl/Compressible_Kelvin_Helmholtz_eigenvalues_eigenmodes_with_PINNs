#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.scripts.classical.solve_robust_subsonic_shooting import RobustSubsonicShootingSolver


def csqrt_pos(z: complex) -> complex:
    q = np.sqrt(z + 0j)
    if q.real < 0:
        q = -q
    if abs(q.real) < 1e-14 and q.imag < 0:
        q = -q
    return q


def U_np(y):
    return np.tanh(y)


def Up_np(y):
    u = np.tanh(y)
    return 1.0 - u * u


def classical_pressure_reference(alpha: float, Mach: float, ci: float, ymax: float, n_y: int):
    c = 1j * ci
    y = np.linspace(-ymax, ymax, n_y)
    h = y[1] - y[0]

    lam_left = alpha * csqrt_pos(1.0 - Mach**2 * ((-1.0) - c) ** 2)

    state = np.zeros((n_y, 2), dtype=np.complex128)
    state[0, 0] = 1.0 + 0j
    state[0, 1] = lam_left

    def rhs(yy, s):
        p, q = s
        U = math.tanh(float(yy))
        Up = 1.0 - U * U
        denom = U - c
        A = 2.0 * Up / denom
        B = alpha**2 * (1.0 - Mach**2 * denom**2)
        return np.array([q, A * q + B * p], dtype=np.complex128)

    for i in range(n_y - 1):
        yi = y[i]
        s = state[i]
        k1 = rhs(yi, s)
        k2 = rhs(yi + 0.5 * h, s + 0.5 * h * k1)
        k3 = rhs(yi + 0.5 * h, s + 0.5 * h * k2)
        k4 = rhs(yi + h, s + h * k3)
        state[i + 1] = s + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    p = state[:, 0]
    q = state[:, 1]

    p0 = np.interp(0.0, y, p.real) + 1j * np.interp(0.0, y, p.imag)
    if abs(p0) < 1e-14:
        p0 = p[np.argmax(np.abs(p))]

    p = p / p0
    q = q / p0

    return y, p, q


class FourierMLP(nn.Module):
    def __init__(self, ymax: float, width: int = 160, depth: int = 5, n_freq: int = 8):
        super().__init__()
        self.ymax = float(ymax)
        self.n_freq = int(n_freq)

        in_dim = 1 + 2 * n_freq
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.SiLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(width, 4))
        self.net = nn.Sequential(*layers)

        # Start near p=1, q=0, not near exact solution.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)
            self.net[-1].bias.zero_()
            self.net[-1].bias[0] = 1.0

    def features(self, y: torch.Tensor) -> torch.Tensor:
        z = y / self.ymax
        feats = [z]
        for k in range(1, self.n_freq + 1):
            feats.append(torch.sin(math.pi * k * z))
            feats.append(torch.cos(math.pi * k * z))
        return torch.cat(feats, dim=1)

    def forward(self, y: torch.Tensor):
        out = self.net(self.features(y))
        pr, pi, qr, qi = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
        p = torch.complex(pr, pi)
        q = torch.complex(qr, qi)
        return p, q


def complex_mse(z: torch.Tensor) -> torch.Tensor:
    return torch.mean(z.real**2 + z.imag**2)


def grad_complex(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    gr = torch.autograd.grad(z.real.sum(), y, create_graph=True, retain_graph=True)[0]
    gi = torch.autograd.grad(z.imag.sum(), y, create_graph=True, retain_graph=True)[0]
    return torch.complex(gr, gi)


def sample_y(n: int, ymax: float, central_ymax: float, device):
    n1 = n // 2
    n2 = n - n1

    y_uniform = -ymax + 2.0 * ymax * torch.rand(n1, 1, device=device)
    y_central = -central_ymax + 2.0 * central_ymax * torch.rand(n2, 1, device=device)

    y = torch.cat([y_uniform, y_central], dim=0)
    y.requires_grad_(True)
    return y


def torch_constants(alpha, Mach, ci, device, dtype=torch.float64):
    c = torch.complex(
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(ci, device=device, dtype=dtype),
    )
    a = torch.tensor(alpha, device=device, dtype=dtype)
    M = torch.tensor(Mach, device=device, dtype=dtype)
    return a, M, c


def loss_terms(model, y, alpha, Mach, ci, ymax):
    dtype = y.dtype
    device = y.device
    a, M, c = torch_constants(alpha, Mach, ci, device, dtype)

    p, q = model(y)

    dp = grad_complex(p, y)
    dq = grad_complex(q, y)

    U = torch.tanh(y)
    Up = 1.0 - U * U
    Uc = torch.complex(U, torch.zeros_like(U))
    Upc = torch.complex(Up, torch.zeros_like(Up))

    denom = Uc - c
    A = 2.0 * Upc / denom
    B = a**2 * (1.0 - M**2 * denom**2)

    r_compat = dp - q
    r_ode = dq - A * q - B * p

    loss_compat = complex_mse(r_compat)
    loss_ode = complex_mse(r_ode)

    # Center gauge p(0)=1+0i.
    y0 = torch.zeros(1, 1, device=device, dtype=dtype, requires_grad=True)
    p0, q0 = model(y0)
    loss_gauge = torch.mean((p0.real - 1.0) ** 2 + p0.imag**2)

    # Optional center derivative parity: Re q(0)=0.
    loss_q_center = torch.mean(q0.real**2)

    # Robin far-field BC.
    yL = torch.tensor([[-ymax]], device=device, dtype=dtype)
    yR = torch.tensor([[ ymax]], device=device, dtype=dtype)

    pL, qL = model(yL)
    pR, qR = model(yR)

    lam_left = alpha * csqrt_pos(1.0 - Mach**2 * ((-1.0) - 1j * ci) ** 2)
    lam_right = alpha * csqrt_pos(1.0 - Mach**2 * ((1.0) - 1j * ci) ** 2)

    lamL = torch.complex(
        torch.tensor(lam_left.real, device=device, dtype=dtype),
        torch.tensor(lam_left.imag, device=device, dtype=dtype),
    )
    lamR = torch.complex(
        torch.tensor(lam_right.real, device=device, dtype=dtype),
        torch.tensor(lam_right.imag, device=device, dtype=dtype),
    )

    r_bc_left = qL - lamL * pL
    r_bc_right = qR + lamR * pR
    loss_bc = complex_mse(r_bc_left) + complex_mse(r_bc_right)

    return {
        "compat": loss_compat,
        "ode": loss_ode,
        "bc": loss_bc,
        "gauge": loss_gauge,
        "q_center": loss_q_center,
    }


def eval_model(model, y_np, device):
    dtype = torch.float64
    y = torch.tensor(y_np[:, None], device=device, dtype=dtype)
    with torch.no_grad():
        p, q = model(y)
    return p.cpu().numpy().reshape(-1), q.cpu().numpy().reshape(-1)


def align_complex(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> complex:
    num = np.vdot(pred[mask], ref[mask])
    den = np.vdot(pred[mask], pred[mask])
    if abs(den) < 1e-300:
        return 1.0 + 0j
    return num / den


def rel_l2(pred, ref, y, mask):
    num = np.trapz(np.abs(pred[mask] - ref[mask]) ** 2, y[mask])
    den = np.trapz(np.abs(ref[mask]) ** 2, y[mask])
    return float(np.sqrt(num / max(den, 1e-300)))


def fields_from_pq(y, p, q, alpha, Mach, ci):
    c = 1j * ci
    U = np.tanh(y)
    Up = 1.0 - U * U
    denom = U - c

    rho = Mach**2 * p
    v = -q / (1j * alpha * denom)
    u = -(Up * v + 1j * alpha * p) / (1j * alpha * denom)
    gamma = q / np.where(np.abs(p) > 1e-14, p, np.nan + 1j * np.nan)

    return rho, u, v, gamma


def plot_complex_pair(y, ref, pred, title, path, ylabel):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(y, ref.real, label="classic Re")
    axes[0].plot(y, pred.real, "--", label="PINN Re")
    axes[0].set_ylabel(f"Re {ylabel}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(y, ref.imag, label="classic Im")
    axes[1].plot(y, pred.imag, "--", label="PINN Im")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel(f"Im {ylabel}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--mach", type=float, default=0.5)
    ap.add_argument("--ci", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=75.0)
    ap.add_argument("--central-ymax", type=float, default=15.0)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--n-train", type=int, default=3072)
    ap.add_argument("--n-y", type=int, default=4001)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--n-freq", type=int, default=8)
    ap.add_argument("--w-compat", type=float, default=10.0)
    ap.add_argument("--w-ode", type=float, default=1.0)
    ap.add_argument("--w-bc", type=float, default=20.0)
    ap.add_argument("--w-gauge", type=float, default=100.0)
    ap.add_argument("--w-q-center", type=float, default=1.0)
    ap.add_argument("--amp-mask-frac", type=float, default=0.05)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1234)
    np.random.seed(1234)

    if args.ci is None:
        r = RobustSubsonicShootingSolver(alpha=args.alpha, Mach=args.mach).solve()
        ci = float(r.ci)
    else:
        ci = float(args.ci)

    print("Subsonic single-case first-order p/q PINN")
    print(f"alpha={args.alpha} Mach={args.mach} ci={ci} device={device}")
    print(f"output_dir={outdir}")
    print(
        f"weights: compat={args.w_compat} ode={args.w_ode} bc={args.w_bc} "
        f"gauge={args.w_gauge} q_center={args.w_q_center}"
    )

    model = FourierMLP(
        ymax=args.ymax,
        width=args.width,
        depth=args.depth,
        n_freq=args.n_freq,
    ).to(device=device, dtype=torch.float64)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8, foreach=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    history = []
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        y = sample_y(args.n_train, args.ymax, args.central_ymax, device)
        terms = loss_terms(model, y, args.alpha, args.mach, ci, args.ymax)

        loss = (
            args.w_compat * terms["compat"]
            + args.w_ode * terms["ode"]
            + args.w_bc * terms["bc"]
            + args.w_gauge * terms["gauge"]
            + args.w_q_center * terms["q_center"]
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()

        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "compat": float(terms["compat"].detach().cpu()),
            "ode": float(terms["ode"].detach().cpu()),
            "bc": float(terms["bc"].detach().cpu()),
            "gauge": float(terms["gauge"].detach().cpu()),
            "q_center": float(terms["q_center"].detach().cpu()),
            "lr": float(sched.get_last_lr()[0]),
        }
        history.append(row)

        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "alpha": args.alpha,
                    "Mach": args.mach,
                    "ci": ci,
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_loss": best_loss,
                },
                outdir / "model_best.pt",
            )

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Epoch {epoch:5d} | loss={row['loss']:.3e} "
                f"compat={row['compat']:.3e} ode={row['ode']:.3e} "
                f"bc={row['bc']:.3e} gauge={row['gauge']:.3e} "
                f"q_center={row['q_center']:.3e}"
            )

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)

    ckpt = torch.load(outdir / "model_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_ref, p_ref, q_ref = classical_pressure_reference(
        args.alpha, args.mach, ci, args.central_ymax, args.n_y
    )
    p_pred, q_pred = eval_model(model, y_ref, device)

    mask = np.abs(p_ref) >= args.amp_mask_frac * np.nanmax(np.abs(p_ref))
    scale = align_complex(p_pred, p_ref, mask)

    p_pred = scale * p_pred
    q_pred = scale * q_pred

    rho_ref, u_ref, v_ref, gamma_ref = fields_from_pq(y_ref, p_ref, q_ref, args.alpha, args.mach, ci)
    rho_pred, u_pred, v_pred, gamma_pred = fields_from_pq(y_ref, p_pred, q_pred, args.alpha, args.mach, ci)

    metrics = {
        "alpha": args.alpha,
        "Mach": args.mach,
        "ci": ci,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "p_rel": rel_l2(p_pred, p_ref, y_ref, mask),
        "q_rel": rel_l2(q_pred, q_ref, y_ref, mask),
        "p_y_rel": rel_l2(q_pred, q_ref, y_ref, mask),
        "rho_rel": rel_l2(rho_pred, rho_ref, y_ref, mask),
        "u_rel": rel_l2(u_pred, u_ref, y_ref, mask),
        "v_rel": rel_l2(v_pred, v_ref, y_ref, mask),
        "gamma_rel": rel_l2(gamma_pred, gamma_ref, y_ref, mask),
        "amp_mask_frac": args.amp_mask_frac,
        "align_scale_real": scale.real,
        "align_scale_imag": scale.imag,
    }

    pd.DataFrame([metrics]).to_csv(outdir / "diagnostics_summary.csv", index=False)

    with open(outdir / "README.md", "w") as f:
        f.write("# Subsonic single-case first-order p/q PINN\n\n")
        f.write(f"- alpha={args.alpha}\n")
        f.write(f"- Mach={args.mach}\n")
        f.write(f"- ci={ci}\n")
        f.write("- trained outputs: Re p, Im p, Re q, Im q with q ≈ p_y\n")
        f.write("- no classical modal field is used in the training loss\n")
        f.write("- classical solution is used only for post-training diagnostics\n\n")
        for k, v in metrics.items():
            f.write(f"- {k}: {v}\n")

    # Plots.
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ["loss", "compat", "ode", "bc", "gauge", "q_center"]:
        ax.semilogy(h["epoch"], h[col], label=col)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss term")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "01_loss_history.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot_complex_pair(y_ref, p_ref, p_pred, "Pressure p", outdir / "02_pressure_p.png", "p")
    plot_complex_pair(y_ref, q_ref, q_pred, "First-order derivative q = p_y", outdir / "03_derivative_q_py.png", "q")

    plot_complex_pair(y_ref, gamma_ref, gamma_pred, "Gamma = q / p", outdir / "04_gamma.png", "gamma")
    plot_complex_pair(y_ref, v_ref, v_pred, "Velocity v", outdir / "05_velocity_v.png", "v")
    plot_complex_pair(y_ref, u_ref, u_pred, "Velocity u", outdir / "06_velocity_u.png", "u")

    print("\n[OK] wrote", outdir)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
