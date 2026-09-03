#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch import nn


def csqrt_decay(z: complex) -> complex:
    q = np.sqrt(z + 0j)
    if q.real < 0:
        q = -q
    if abs(q.real) < 1e-14 and q.imag < 0:
        q = -q
    return q


def torch_sqrt_decay(z: torch.Tensor) -> torch.Tensor:
    q = torch.sqrt(z)
    q = torch.where(q.real < 0.0, -q, q)
    q = torch.where((q.real.abs() < 1e-14) & (q.imag < 0.0), -q, q)
    return q


def complex_mse(z: torch.Tensor) -> torch.Tensor:
    return torch.mean(z.real**2 + z.imag**2)


def grad_complex(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    gr = torch.autograd.grad(z.real.sum(), y, create_graph=True, retain_graph=True)[0]
    gi = torch.autograd.grad(z.imag.sum(), y, create_graph=True, retain_graph=True)[0]
    return torch.complex(gr, gi)


class FourierMLP(nn.Module):
    def __init__(self, ymax: float, width: int = 192, depth: int = 6, n_freq: int = 10):
        super().__init__()
        self.ymax = float(ymax)
        self.n_freq = int(n_freq)

        in_dim = 1 + 2 * n_freq
        layers = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 4)]
        self.net = nn.Sequential(*layers)

        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)
            self.net[-1].bias.zero_()
            self.net[-1].bias[0] = 1.0  # Re p starts near 1.

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


def sample_y(n: int, ymax: float, central_ymax: float, device):
    dtype = torch.float64
    n1 = n // 2
    n2 = n - n1

    y_uniform = -ymax + 2.0 * ymax * torch.rand(n1, 1, device=device, dtype=dtype)
    y_central = -central_ymax + 2.0 * central_ymax * torch.rand(n2, 1, device=device, dtype=dtype)
    y = torch.cat([y_uniform, y_central], dim=0)
    y.requires_grad_(True)
    return y


def loss_terms(model, y, alpha: float, Mach: float, cr: float, ci: float, ymax: float, device):
    dtype = torch.float64

    p, q = model(y)
    dp = grad_complex(p, y)
    dq = grad_complex(q, y)

    c = torch.complex(
        torch.tensor(cr, device=device, dtype=dtype),
        torch.tensor(ci, device=device, dtype=dtype),
    )

    U = torch.tanh(y)
    Up = 1.0 - U * U
    Uc = torch.complex(U, torch.zeros_like(U))
    Upc = torch.complex(Up, torch.zeros_like(Up))

    denom = Uc - c

    A = 2.0 * Upc / denom
    B = alpha**2 * (1.0 - Mach**2 * denom**2)

    r_compat = dp - q
    r_ode = dq - A * q - B * p

    loss_compat = complex_mse(r_compat)
    loss_ode = complex_mse(r_ode)

    # Gauge p(0)=1.
    y0 = torch.zeros_like(y)
    y0.requires_grad_(True)
    p0, q0 = model(y0)
    loss_gauge = torch.mean((p0.real - 1.0) ** 2 + p0.imag**2)

    # Robin far-field BC.
    yL = torch.full_like(y, -ymax)
    yR = torch.full_like(y, ymax)

    pL, qL = model(yL)
    pR, qR = model(yR)

    minus_one = torch.complex(
        -torch.ones_like(y, dtype=dtype),
        torch.zeros_like(y, dtype=dtype),
    )
    plus_one = torch.complex(
        torch.ones_like(y, dtype=dtype),
        torch.zeros_like(y, dtype=dtype),
    )

    lam_left = alpha * torch_sqrt_decay(1.0 - Mach**2 * (minus_one - c) ** 2)
    lam_right = alpha * torch_sqrt_decay(1.0 - Mach**2 * (plus_one - c) ** 2)

    r_bc_left = qL - lam_left * pL
    r_bc_right = qR + lam_right * pR

    loss_bc = complex_mse(r_bc_left) + complex_mse(r_bc_right)

    # Small regularizer to keep q(0) finite, not a parity constraint.
    loss_q0 = torch.mean(q0.real**2 + q0.imag**2)

    return {
        "compat": loss_compat,
        "ode": loss_ode,
        "bc": loss_bc,
        "gauge": loss_gauge,
        "q0": loss_q0,
    }


def classical_pressure_reference(alpha: float, Mach: float, cr: float, ci: float, ymax: float, n_y: int):
    c = cr + 1j * ci

    y = np.linspace(-ymax, ymax, n_y)
    h = y[1] - y[0]

    lam_left = alpha * csqrt_decay(1.0 - Mach**2 * ((-1.0) - c) ** 2)

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


def eval_model(model, y_np: np.ndarray, device):
    y = torch.tensor(y_np[:, None], device=device, dtype=torch.float64)
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
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    good = mask & np.isfinite(pred.real) & np.isfinite(pred.imag) & np.isfinite(ref.real) & np.isfinite(ref.imag)
    num = np.trapz(np.abs(pred[good] - ref[good]) ** 2, y[good])
    den = np.trapz(np.abs(ref[good]) ** 2, y[good])
    return float(np.sqrt(num / max(den, 1e-300)))


def fields_from_pq(y, p, q, alpha, Mach, cr, ci):
    c = cr + 1j * ci
    U = np.tanh(y)
    Up = 1.0 - U * U
    denom = U - c

    rho = Mach**2 * p
    v = -q / (1j * alpha * denom)
    u = -(Up * v + 1j * alpha * p) / (1j * alpha * denom)

    gamma = np.full_like(p, np.nan + 1j * np.nan)
    ok = np.abs(p) > 1e-14
    gamma[ok] = q[ok] / p[ok]

    return rho, u, v, gamma


def plot_complex_pair(y, ref, pred, title, path, ylabel):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(y, ref.real, label="classic Re")
    axes[0].plot(y, pred.real, "--", label="PINN Re")
    axes[0].set_ylabel(f"Re {ylabel}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(y, ref.imag, label="classic Im")
    axes[1].plot(y, pred.imag, "--", label="PINN Im")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel(f"Im {ylabel}")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def diagnostics(args, model, device, outdir: Path):
    y, p_ref, q_ref = classical_pressure_reference(
        alpha=args.alpha,
        Mach=args.mach,
        cr=args.cr,
        ci=args.ci,
        ymax=args.diagnostic_ymax,
        n_y=args.n_y,
    )

    p_pred, q_pred = eval_model(model, y, device)

    mask = np.abs(p_ref) >= args.amp_mask_frac * np.nanmax(np.abs(p_ref))
    scale = align_complex(p_pred, p_ref, mask)

    p_pred = scale * p_pred
    q_pred = scale * q_pred

    rho_ref, u_ref, v_ref, gamma_ref = fields_from_pq(y, p_ref, q_ref, args.alpha, args.mach, args.cr, args.ci)
    rho_pred, u_pred, v_pred, gamma_pred = fields_from_pq(y, p_pred, q_pred, args.alpha, args.mach, args.cr, args.ci)

    row = {
        "alpha": args.alpha,
        "Mach": args.mach,
        "cr": args.cr,
        "ci": args.ci,
        "p_rel": rel_l2(p_pred, p_ref, y, mask),
        "q_rel": rel_l2(q_pred, q_ref, y, mask),
        "p_y_rel": rel_l2(q_pred, q_ref, y, mask),
        "rho_rel": rel_l2(rho_pred, rho_ref, y, mask),
        "u_rel": rel_l2(u_pred, u_ref, y, mask),
        "v_rel": rel_l2(v_pred, v_ref, y, mask),
        "gamma_rel": rel_l2(gamma_pred, gamma_ref, y, mask),
        "align_scale_real": scale.real,
        "align_scale_imag": scale.imag,
        "amp_mask_frac": args.amp_mask_frac,
    }

    pd.DataFrame([row]).to_csv(outdir / "diagnostics_summary.csv", index=False)

    plot_complex_pair(y, p_ref, p_pred, "Supersonic pressure p", outdir / "02_pressure_p.png", "p")
    plot_complex_pair(y, q_ref, q_pred, "Supersonic derivative q=p_y", outdir / "03_derivative_q_py.png", "q")
    plot_complex_pair(y, gamma_ref, gamma_pred, "Supersonic gamma=q/p", outdir / "04_gamma.png", "gamma")
    plot_complex_pair(y, v_ref, v_pred, "Supersonic velocity v", outdir / "05_velocity_v.png", "v")
    plot_complex_pair(y, u_ref, u_pred, "Supersonic velocity u", outdir / "06_velocity_u.png", "u")

    with open(outdir / "README.md", "w") as f:
        f.write("# Supersonic single-case first-order p/q PINN\n\n")
        f.write(f"- alpha={args.alpha}\n")
        f.write(f"- Mach={args.mach}\n")
        f.write(f"- cr={args.cr}\n")
        f.write(f"- ci={args.ci}\n")
        f.write("- trained outputs: Re p, Im p, Re q, Im q with q approx p_y\n")
        f.write("- c is fixed from the classical spectral point\n")
        f.write("- no classical modal field is used in the training loss\n")
        f.write("- classical pressure/modes are used only for post-training diagnostics\n\n")
        f.write("## Diagnostics\n\n")
        for k, v in row.items():
            f.write(f"- {k}: {v}\n")

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--mach", type=float, required=True)
    ap.add_argument("--cr", type=float, required=True)
    ap.add_argument("--ci", type=float, required=True)

    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--ymax", type=float, default=120.0)
    ap.add_argument("--central-ymax", type=float, default=20.0)
    ap.add_argument("--diagnostic-ymax", type=float, default=120.0)
    ap.add_argument("--n-y", type=int, default=4001)

    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n-freq", type=int, default=10)

    ap.add_argument("--w-compat", type=float, default=10.0)
    ap.add_argument("--w-ode", type=float, default=10.0)
    ap.add_argument("--w-bc", type=float, default=20.0)
    ap.add_argument("--w-gauge", type=float, default=100.0)
    ap.add_argument("--w-q0", type=float, default=0.0)

    ap.add_argument("--amp-mask-frac", type=float, default=0.05)

    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1234)
    np.random.seed(1234)

    print("Supersonic single-case first-order p/q PINN")
    print(f"alpha={args.alpha} mach={args.mach} cr={args.cr} ci={args.ci}")
    print(f"output_dir={outdir}")
    print(f"device={device}")
    print(f"weights: compat={args.w_compat} ode={args.w_ode} bc={args.w_bc} gauge={args.w_gauge} q0={args.w_q0}")
    print(f"ymax={args.ymax} central_ymax={args.central_ymax}")

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

        terms = loss_terms(
            model=model,
            y=y,
            alpha=args.alpha,
            Mach=args.mach,
            cr=args.cr,
            ci=args.ci,
            ymax=args.ymax,
            device=device,
        )

        loss = (
            args.w_compat * terms["compat"]
            + args.w_ode * terms["ode"]
            + args.w_bc * terms["bc"]
            + args.w_gauge * terms["gauge"]
            + args.w_q0 * terms["q0"]
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
            "q0": float(terms["q0"].detach().cpu()),
            "lr": float(sched.get_last_lr()[0]),
        }
        history.append(row)

        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
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
                f"bc={row['bc']:.3e} gauge={row['gauge']:.3e} q0={row['q0']:.3e}"
            )

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)

    ckpt = torch.load(outdir / "model_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ["loss", "compat", "ode", "bc", "gauge", "q0"]:
        ax.semilogy(h["epoch"], h[col], label=col)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss term")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "01_loss_history.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    row = diagnostics(args, model, device, outdir)

    print("\n[OK] wrote", outdir)
    print(f"best_epoch={best_epoch} best_loss={best_loss:.6e}")
    print("\nDiagnostics:")
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
