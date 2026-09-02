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


def complex_mse(z: torch.Tensor) -> torch.Tensor:
    return torch.mean(z.real**2 + z.imag**2)


def real_mse(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x**2)


def grad_complex(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    gr = torch.autograd.grad(z.real.sum(), y, create_graph=True, retain_graph=True)[0]
    gi = torch.autograd.grad(z.imag.sum(), y, create_graph=True, retain_graph=True)[0]
    return torch.complex(gr, gi)


def inv_softplus(x: float) -> float:
    return float(np.log(np.expm1(x)))


def torch_sqrt_pos(z: torch.Tensor) -> torch.Tensor:
    q = torch.sqrt(z)
    q = torch.where(q.real < 0.0, -q, q)
    q = torch.where((q.real.abs() < 1e-14) & (q.imag < 0.0), -q, q)
    return q


class FieldPQNet(nn.Module):
    def __init__(
        self,
        ymax: float,
        alpha_min: float,
        alpha_max: float,
        mach_min: float,
        mach_max: float,
        width: int = 192,
        depth: int = 6,
        n_freq: int = 8,
    ):
        super().__init__()
        self.ymax = float(ymax)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.mach_min = float(mach_min)
        self.mach_max = float(mach_max)
        self.n_freq = int(n_freq)

        in_dim = 3 + 2 * n_freq
        layers = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 4)]
        self.net = nn.Sequential(*layers)

        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)
            self.net[-1].bias.zero_()
            self.net[-1].bias[0] = 1.0  # Re p starts near 1.

    def normalize_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        den = max(self.alpha_max - self.alpha_min, 1e-12)
        return 2.0 * (alpha - self.alpha_min) / den - 1.0

    def normalize_mach(self, mach: torch.Tensor) -> torch.Tensor:
        den = max(self.mach_max - self.mach_min, 1e-12)
        if den < 1e-11:
            return torch.zeros_like(mach)
        return 2.0 * (mach - self.mach_min) / den - 1.0

    def features(self, y: torch.Tensor, alpha: torch.Tensor, mach: torch.Tensor) -> torch.Tensor:
        z = y / self.ymax
        a = self.normalize_alpha(alpha)
        m = self.normalize_mach(mach)

        feats = [z, a, m]
        for k in range(1, self.n_freq + 1):
            feats.append(torch.sin(math.pi * k * z))
            feats.append(torch.cos(math.pi * k * z))
        return torch.cat(feats, dim=1)

    def forward(self, y: torch.Tensor, alpha: torch.Tensor, mach: torch.Tensor):
        out = self.net(self.features(y, alpha, mach))
        pr, pi, qr, qi = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
        p = torch.complex(pr, pi)
        q = torch.complex(qr, qi)
        return p, q


class CiNet(nn.Module):
    def __init__(
        self,
        alpha_min: float,
        alpha_max: float,
        mach_min: float,
        mach_max: float,
        ci_init: float,
        width: int = 64,
        depth: int = 3,
    ):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.mach_min = float(mach_min)
        self.mach_max = float(mach_max)

        in_dim = 5
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    m.weight.mul_(0.05)
                    m.bias.zero_()
            self.net[-1].bias.fill_(inv_softplus(max(ci_init, 1e-4)))

    def normalize_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        den = max(self.alpha_max - self.alpha_min, 1e-12)
        return 2.0 * (alpha - self.alpha_min) / den - 1.0

    def normalize_mach(self, mach: torch.Tensor) -> torch.Tensor:
        den = max(self.mach_max - self.mach_min, 1e-12)
        if den < 1e-11:
            return torch.zeros_like(mach)
        return 2.0 * (mach - self.mach_min) / den - 1.0

    def forward(self, alpha: torch.Tensor, mach: torch.Tensor) -> torch.Tensor:
        a = self.normalize_alpha(alpha)
        m = self.normalize_mach(mach)
        x = torch.cat([a, m, a * a, a * m, m * m], dim=1)
        raw = self.net(x)
        return torch.nn.functional.softplus(raw) + 1e-8


def parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.replace(",", " ").split() if x.strip()]


def build_ci_anchor_table(anchor_alphas: list[float], mach_values: list[float]) -> pd.DataFrame:
    rows = []
    for M in mach_values:
        for a in anchor_alphas:
            r = RobustSubsonicShootingSolver(alpha=float(a), Mach=float(M)).solve()
            rows.append(
                {
                    "alpha": float(a),
                    "Mach": float(M),
                    "ci": float(r.ci),
                    "omega_i": float(a) * float(r.ci),
                }
            )
    return pd.DataFrame(rows)


def sample_batch(n: int, ymax: float, central_ymax: float, alpha_min: float, alpha_max: float, mach_values: list[float], device, train_alphas: list[float] | None = None):
    dtype = torch.float64

    n1 = n // 2
    n2 = n - n1

    y_uniform = -ymax + 2.0 * ymax * torch.rand(n1, 1, device=device, dtype=dtype)
    y_central = -central_ymax + 2.0 * central_ymax * torch.rand(n2, 1, device=device, dtype=dtype)
    y = torch.cat([y_uniform, y_central], dim=0)
    y.requires_grad_(True)

    if train_alphas:
        av = torch.tensor(train_alphas, device=device, dtype=dtype).view(-1)
        ia = torch.randint(0, len(train_alphas), (n,), device=device)
        alpha = av[ia].view(n, 1)
    else:
        alpha = alpha_min + (alpha_max - alpha_min) * torch.rand(n, 1, device=device, dtype=dtype)

    mv = torch.tensor(mach_values, device=device, dtype=dtype).view(-1)
    idx = torch.randint(0, len(mach_values), (n,), device=device)
    mach = mv[idx].view(n, 1)

    return y, alpha, mach


def physics_losses(
    field: FieldPQNet,
    ci_net: CiNet,
    y: torch.Tensor,
    alpha: torch.Tensor,
    mach: torch.Tensor,
    ymax: float,
    detach_ci: bool,
):
    p, q = field(y, alpha, mach)

    ci = ci_net(alpha, mach)
    ci_phys = ci.detach() if detach_ci else ci
    c = torch.complex(torch.zeros_like(ci_phys), ci_phys)

    dp = grad_complex(p, y)
    dq = grad_complex(q, y)

    U = torch.tanh(y)
    Up = 1.0 - U * U
    Uc = torch.complex(U, torch.zeros_like(U))
    Upc = torch.complex(Up, torch.zeros_like(Up))

    denom = Uc - c
    A = 2.0 * Upc / denom
    B = alpha**2 * (1.0 - mach**2 * denom**2)

    r_compat = dp - q
    r_ode = dq - A * q - B * p

    loss_compat = complex_mse(r_compat)
    loss_ode = complex_mse(r_ode)

    y0 = torch.zeros_like(y)
    y0.requires_grad_(True)
    p0, q0 = field(y0, alpha, mach)
    loss_gauge = torch.mean((p0.real - 1.0) ** 2 + p0.imag**2)
    loss_q_center = torch.mean(q0.real**2)

    yL = torch.full_like(y, -ymax)
    yR = torch.full_like(y, ymax)

    pL, qL = field(yL, alpha, mach)
    pR, qR = field(yR, alpha, mach)

    minus_one = torch.complex(-torch.ones_like(alpha), torch.zeros_like(alpha))
    plus_one = torch.complex(torch.ones_like(alpha), torch.zeros_like(alpha))

    lam_left = alpha * torch_sqrt_pos(1.0 - mach**2 * (minus_one - c) ** 2)
    lam_right = alpha * torch_sqrt_pos(1.0 - mach**2 * (plus_one - c) ** 2)

    r_bc_left = qL - lam_left * pL
    r_bc_right = qR + lam_right * pR
    loss_bc = complex_mse(r_bc_left) + complex_mse(r_bc_right)

    return {
        "compat": loss_compat,
        "ode": loss_ode,
        "bc": loss_bc,
        "gauge": loss_gauge,
        "q_center": loss_q_center,
        "ci_mean": ci.mean(),
    }


def parity_loss(field: FieldPQNet, n: int, sym_ymax: float, alpha_min: float, alpha_max: float, mach_values: list[float], device, train_alphas: list[float] | None = None):
    dtype = torch.float64
    y = sym_ymax * torch.rand(n, 1, device=device, dtype=dtype)

    if train_alphas:
        av = torch.tensor(train_alphas, device=device, dtype=dtype).view(-1)
        ia = torch.randint(0, len(train_alphas), (n,), device=device)
        alpha = av[ia].view(n, 1)
    else:
        alpha = alpha_min + (alpha_max - alpha_min) * torch.rand(n, 1, device=device, dtype=dtype)

    mv = torch.tensor(mach_values, device=device, dtype=dtype).view(-1)
    idx = torch.randint(0, len(mach_values), (n,), device=device)
    mach = mv[idx].view(n, 1)

    p_plus, q_plus = field(y, alpha, mach)
    p_minus, q_minus = field(-y, alpha, mach)

    # p: Re even, Im odd. q=p_y: Re odd, Im even.
    return (
        real_mse(p_minus.real - p_plus.real)
        + real_mse(p_minus.imag + p_plus.imag)
        + real_mse(q_minus.real + q_plus.real)
        + real_mse(q_minus.imag - q_plus.imag)
    )


def ci_anchor_loss(ci_net: CiNet, anchor_df: pd.DataFrame, device):
    dtype = torch.float64
    a = torch.tensor(anchor_df["alpha"].to_numpy(), device=device, dtype=dtype).view(-1, 1)
    m = torch.tensor(anchor_df["Mach"].to_numpy(), device=device, dtype=dtype).view(-1, 1)
    ci_ref = torch.tensor(anchor_df["ci"].to_numpy(), device=device, dtype=dtype).view(-1, 1)
    ci_pred = ci_net(a, m)
    # Use a finite scale floor so marginal/near-zero growth-rate anchors do not dominate.
    # This keeps ci supervision meaningful without exploding when ci_ref ≈ 0.
    scale = torch.clamp(ci_ref.abs(), min=5e-2)
    return torch.mean(((ci_pred - ci_ref) / scale) ** 2)


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


def eval_model(field: FieldPQNet, ci_net: CiNet, y_np: np.ndarray, alpha: float, Mach: float, device):
    dtype = torch.float64
    y = torch.tensor(y_np[:, None], device=device, dtype=dtype)
    a = torch.full_like(y, float(alpha))
    m = torch.full_like(y, float(Mach))

    with torch.no_grad():
        p, q = field(y, a, m)
        ci_pred = ci_net(a[:1], m[:1])

    return (
        p.cpu().numpy().reshape(-1),
        q.cpu().numpy().reshape(-1),
        float(ci_pred.cpu().item()),
    )


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


def run_diagnostics(args, field, ci_net, anchor_df, device):
    outdir = Path(args.output_dir)
    diag_rows = []

    eval_alphas = parse_float_list(args.eval_alphas)
    mach_values = parse_float_list(args.mach_values)

    for Mach in mach_values:
        for alpha in eval_alphas:
            robust = RobustSubsonicShootingSolver(alpha=alpha, Mach=Mach).solve()
            ci_ref = float(robust.ci)

            y, p_ref, q_ref = classical_pressure_reference(
                alpha=alpha,
                Mach=Mach,
                ci=ci_ref,
                ymax=args.central_ymax,
                n_y=args.n_y,
            )

            p_pred, q_pred, ci_pred = eval_model(field, ci_net, y, alpha, Mach, device)

            mask = np.abs(p_ref) >= args.amp_mask_frac * np.nanmax(np.abs(p_ref))
            scale = align_complex(p_pred, p_ref, mask)
            p_pred = scale * p_pred
            q_pred = scale * q_pred

            rho_ref, u_ref, v_ref, gamma_ref = fields_from_pq(y, p_ref, q_ref, alpha, Mach, ci_ref)
            rho_pred, u_pred, v_pred, gamma_pred = fields_from_pq(y, p_pred, q_pred, alpha, Mach, ci_pred)

            row = {
                "alpha": alpha,
                "Mach": Mach,
                "ci_ref": ci_ref,
                "ci_pred": ci_pred,
                "ci_abs_err": abs(ci_pred - ci_ref),
                "ci_rel_err": abs(ci_pred - ci_ref) / max(abs(ci_ref), 1e-12),
                "p_rel": rel_l2(p_pred, p_ref, y, mask),
                "q_rel": rel_l2(q_pred, q_ref, y, mask),
                "p_y_rel": rel_l2(q_pred, q_ref, y, mask),
                "rho_rel": rel_l2(rho_pred, rho_ref, y, mask),
                "u_rel": rel_l2(u_pred, u_ref, y, mask),
                "v_rel": rel_l2(v_pred, v_ref, y, mask),
                "gamma_rel": rel_l2(gamma_pred, gamma_ref, y, mask),
                "align_scale_real": scale.real,
                "align_scale_imag": scale.imag,
            }
            diag_rows.append(row)

            tag = f"M{int(round(1000*Mach)):04d}_a{int(round(1000*alpha)):04d}"
            plot_complex_pair(y, p_ref, p_pred, f"Pressure p, M={Mach:g}, alpha={alpha:g}", outdir / f"pressure_p_{tag}.png", "p")
            plot_complex_pair(y, q_ref, q_pred, f"Derivative q=p_y, M={Mach:g}, alpha={alpha:g}", outdir / f"derivative_q_{tag}.png", "q")
            plot_complex_pair(y, u_ref, u_pred, f"Velocity u, M={Mach:g}, alpha={alpha:g}", outdir / f"velocity_u_{tag}.png", "u")
            plot_complex_pair(y, v_ref, v_pred, f"Velocity v, M={Mach:g}, alpha={alpha:g}", outdir / f"velocity_v_{tag}.png", "v")
            plot_complex_pair(y, gamma_ref, gamma_pred, f"Gamma=q/p, M={Mach:g}, alpha={alpha:g}", outdir / f"gamma_{tag}.png", "gamma")

            # Export full diagnostic fields for post-analysis.
            pd.DataFrame({
                "y": y,
                "alpha": float(alpha),
                "Mach": float(Mach),
                "ci_ref": float(ci_ref),
                "ci_pred": float(ci_pred),
                "p_ref_real": np.real(p_ref),
                "p_ref_imag": np.imag(p_ref),
                "p_pred_real": np.real(p_pred),
                "p_pred_imag": np.imag(p_pred),
                "q_ref_real": np.real(q_ref),
                "q_ref_imag": np.imag(q_ref),
                "q_pred_real": np.real(q_pred),
                "q_pred_imag": np.imag(q_pred),
                "rho_ref_real": np.real(rho_ref),
                "rho_ref_imag": np.imag(rho_ref),
                "rho_pred_real": np.real(rho_pred),
                "rho_pred_imag": np.imag(rho_pred),
                "u_ref_real": np.real(u_ref),
                "u_ref_imag": np.imag(u_ref),
                "u_pred_real": np.real(u_pred),
                "u_pred_imag": np.imag(u_pred),
                "v_ref_real": np.real(v_ref),
                "v_ref_imag": np.imag(v_ref),
                "v_pred_real": np.real(v_pred),
                "v_pred_imag": np.imag(v_pred),
                "gamma_ref_real": np.real(gamma_ref),
                "gamma_ref_imag": np.imag(gamma_ref),
                "gamma_pred_real": np.real(gamma_pred),
                "gamma_pred_imag": np.imag(gamma_pred),
            }).to_csv(outdir / f"fields_vs_classic_{tag}.csv", index=False)

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(outdir / "diagnostics_summary.csv", index=False)

    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--mach-values", default="0.5")
    ap.add_argument("--alpha-min", type=float, default=0.3)
    ap.add_argument("--alpha-max", type=float, default=0.7)
    ap.add_argument("--anchor-alphas", default="0.3 0.5 0.7")
    ap.add_argument("--eval-alphas", default="0.3 0.5 0.7")
    ap.add_argument("--train-alphas", default="", help="Optional discrete alpha values for training, e.g. '0.3 0.5 0.7'. If empty, sample alpha continuously.")

    ap.add_argument("--ymax", type=float, default=75.0)
    ap.add_argument("--central-ymax", type=float, default=15.0)
    ap.add_argument("--sym-ymax", type=float, default=15.0)

    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--n-train", type=int, default=3072)
    ap.add_argument("--n-sym", type=int, default=512)
    ap.add_argument("--n-y", type=int, default=4001)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ci-lr", type=float, default=5e-4)
    ap.add_argument("--ci-prefit-steps", type=int, default=2000)

    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n-freq", type=int, default=8)

    ap.add_argument("--w-compat", type=float, default=10.0)
    ap.add_argument("--w-ode", type=float, default=1.0)
    ap.add_argument("--w-bc", type=float, default=20.0)
    ap.add_argument("--w-gauge", type=float, default=100.0)
    ap.add_argument("--w-q-center", type=float, default=1.0)
    ap.add_argument("--w-parity", type=float, default=0.1)
    ap.add_argument("--w-ci", type=float, default=1000.0)

    ap.add_argument("--detach-ci-in-mode-branch", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--amp-mask-frac", type=float, default=0.05)

    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1234)
    np.random.seed(1234)

    mach_values = parse_float_list(args.mach_values)
    anchor_alphas = parse_float_list(args.anchor_alphas)
    train_alphas = parse_float_list(args.train_alphas) if args.train_alphas.strip() else []

    alpha_min = float(args.alpha_min)
    alpha_max = float(args.alpha_max)
    mach_min = min(mach_values)
    mach_max = max(mach_values)

    print("Subsonic mini-2D first-order p/q PINN")
    print(f"output_dir={outdir}")
    print(f"device={device}")
    print(f"mach_values={mach_values}")
    print(f"alpha_range=[{alpha_min}, {alpha_max}]")
    print(f"anchor_alphas={anchor_alphas}")
    print(f"train_alphas={train_alphas if train_alphas else 'continuous'}")
    # Force true no-detach experiment: ci remains coupled to the modal residual.
    # This overrides argparse defaults/legacy flags for the current c_i-coupled runs.
    args.detach_ci_in_mode_branch = False
    print(f"detach_ci_in_mode_branch={args.detach_ci_in_mode_branch}")

    anchor_df = build_ci_anchor_table(anchor_alphas, mach_values)
    anchor_df.to_csv(outdir / "ci_anchor_points.csv", index=False)
    ci_init = float(anchor_df["ci"].mean())

    print("\nCI anchors:")
    print(anchor_df.to_string(index=False))

    field = FieldPQNet(
        ymax=args.ymax,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
        width=args.width,
        depth=args.depth,
        n_freq=args.n_freq,
    ).to(device=device, dtype=torch.float64)

    ci_net = CiNet(
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
        ci_init=ci_init,
    ).to(device=device, dtype=torch.float64)

    # Prefit ci branch on sparse classical anchors only.
    opt_ci = torch.optim.AdamW(ci_net.parameters(), lr=args.ci_lr, weight_decay=1e-10, foreach=False)
    for step in range(1, args.ci_prefit_steps + 1):
        opt_ci.zero_grad(set_to_none=True)
        loss_ci = ci_anchor_loss(ci_net, anchor_df, device)
        loss_ci.backward()
        opt_ci.step()

        if step == 1 or step % 500 == 0:
            print(f"CI prefit {step:5d} | rel_mse={float(loss_ci.detach().cpu()):.3e}")

    opt = torch.optim.AdamW(
        list(field.parameters()) + list(ci_net.parameters()),
        lr=args.lr,
        weight_decay=1e-8,
        foreach=False,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    history = []
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        field.train()
        ci_net.train()
        opt.zero_grad(set_to_none=True)

        y, alpha, mach = sample_batch(
            n=args.n_train,
            ymax=args.ymax,
            central_ymax=args.central_ymax,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            mach_values=mach_values,
            device=device,
            train_alphas=train_alphas,
        )

        terms = physics_losses(
            field=field,
            ci_net=ci_net,
            y=y,
            alpha=alpha,
            mach=mach,
            ymax=args.ymax,
            detach_ci=args.detach_ci_in_mode_branch,
        )

        loss_ci = ci_anchor_loss(ci_net, anchor_df, device)

        if args.w_parity > 0.0:
            loss_parity = parity_loss(
                field=field,
                n=args.n_sym,
                sym_ymax=args.sym_ymax,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                mach_values=mach_values,
                device=device,
                train_alphas=train_alphas,
            )
        else:
            loss_parity = torch.tensor(0.0, device=device, dtype=torch.float64)

        loss = (
            args.w_compat * terms["compat"]
            + args.w_ode * terms["ode"]
            + args.w_bc * terms["bc"]
            + args.w_gauge * terms["gauge"]
            + args.w_q_center * terms["q_center"]
            + args.w_parity * loss_parity
            + args.w_ci * loss_ci
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(field.parameters()) + list(ci_net.parameters()), 10.0)
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
            "parity": float(loss_parity.detach().cpu()),
            "ci_anchor": float(loss_ci.detach().cpu()),
            "ci_batch_mean": float(terms["ci_mean"].detach().cpu()),
            "lr": float(sched.get_last_lr()[0]),
        }
        history.append(row)

        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "field_state_dict": field.state_dict(),
                    "ci_state_dict": ci_net.state_dict(),
                    "args": vars(args),
                    "anchor_df": anchor_df.to_dict(orient="list"),
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
                f"q_center={row['q_center']:.3e} parity={row['parity']:.3e} "
                f"ci_anchor={row['ci_anchor']:.3e}"
            )

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)

    ckpt = torch.load(outdir / "model_best.pt", map_location=device)
    field.load_state_dict(ckpt["field_state_dict"])
    ci_net.load_state_dict(ckpt["ci_state_dict"])
    field.eval()
    ci_net.eval()

    diag = run_diagnostics(args, field, ci_net, anchor_df, device)

    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ["loss", "compat", "ode", "bc", "gauge", "q_center", "parity", "ci_anchor"]:
        ax.semilogy(h["epoch"], h[col], label=col)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss term")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_history.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    with open(outdir / "README.md", "w") as f:
        f.write("# Subsonic mini-2D first-order p/q PINN\n\n")
        f.write(f"- mach_values: {mach_values}\n")
        f.write(f"- alpha_range: [{alpha_min}, {alpha_max}]\n")
        f.write(f"- anchor_alphas: {anchor_alphas}\n")
        f.write(f"- trained outputs: Re p, Im p, Re q, Im q with q approx p_y\n")
        f.write(f"- ci supervision: sparse classical scalar ci anchors only\n")
        f.write(f"- detach_ci_in_mode_branch: {args.detach_ci_in_mode_branch}\n")
        f.write("- no classical modal field p/rho/u/v/gamma/q is used in training loss\n")
        f.write("- classical modal fields are used only for post-training diagnostics\n\n")
        f.write(f"- best_epoch: {best_epoch}\n")
        f.write(f"- best_loss: {best_loss}\n\n")
        f.write("## CI anchors\n\n")
        f.write(anchor_df.to_string(index=False))
        f.write("\n\n## Diagnostics\n\n")
        f.write(diag.to_string(index=False))
        f.write("\n")

    print("\n[OK] wrote", outdir)
    print("\nDiagnostics:")
    print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
