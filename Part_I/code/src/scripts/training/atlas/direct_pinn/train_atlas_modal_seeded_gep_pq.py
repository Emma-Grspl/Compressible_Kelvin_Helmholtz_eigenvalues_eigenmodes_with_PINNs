from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classical_solver.subsonic.robust_subsonic_shooting import RobustSubsonicShootingSolver

from scripts.train_kh_subsonic_2d_pressure_pq_firstorder_mini import (
    FieldPQNet,
    physics_losses,
    classical_pressure_reference,
    eval_model,
    align_complex,
    rel_l2,
    fields_from_pq,
    plot_complex_pair,
    parse_float_list,
    real_mse,
)


def alpha_cut_np(M: float) -> float:
    return float(np.sqrt(max(1.0 - float(M) ** 2, 1e-14)))


def alpha_from_eta_np(eta: float, M: float) -> float:
    return float(eta) * alpha_cut_np(M)


def torch_alpha_cut(mach: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(1.0 - mach**2, min=1e-14))


class CiGridIDW(nn.Module):
    """
    Fixed sparse 2D interpolant for c_i over coordinates (eta, Mach),
    where eta = alpha / sqrt(1-Mach^2).

    This is intentionally non-trainable: sparse scalar c_i data are used
    as fixed spectral information, while the PINN learns p/q continuously.
    """

    def __init__(
        self,
        anchor_df: pd.DataFrame,
        eta_scale: float = 0.25,
        mach_scale: float = 0.25,
        power: float = 4.0,
        eps: float = 1e-18,
    ):
        super().__init__()

        eta = torch.tensor(anchor_df["eta"].to_numpy(), dtype=torch.float64).view(1, -1)
        mach = torch.tensor(anchor_df["Mach"].to_numpy(), dtype=torch.float64).view(1, -1)
        ci = torch.tensor(anchor_df["ci"].to_numpy(), dtype=torch.float64).view(1, -1)

        self.register_buffer("anchor_eta", eta)
        self.register_buffer("anchor_mach", mach)
        self.register_buffer("anchor_ci", ci)

        self.eta_scale = float(eta_scale)
        self.mach_scale = float(mach_scale)
        self.power = float(power)
        self.eps = float(eps)

    def forward(self, alpha: torch.Tensor, mach: torch.Tensor) -> torch.Tensor:
        eta = alpha / torch_alpha_cut(mach)

        d2 = (
            ((eta - self.anchor_eta) / self.eta_scale) ** 2
            + ((mach - self.anchor_mach) / self.mach_scale) ** 2
        )

        weights = 1.0 / torch.clamp(d2, min=self.eps) ** (0.5 * self.power)
        ci_interp = torch.sum(weights * self.anchor_ci, dim=1, keepdim=True) / torch.sum(weights, dim=1, keepdim=True)

        min_d2, idx = torch.min(d2, dim=1, keepdim=True)
        ci_exact = torch.gather(self.anchor_ci.expand(alpha.shape[0], -1), 1, idx)
        ci = torch.where(min_d2 < 1e-20, ci_exact, ci_interp)

        return ci


def build_ci_anchor_table_eta(anchor_mach_values: list[float], anchor_eta_values: list[float]) -> pd.DataFrame:
    rows = []

    for M in anchor_mach_values:
        for eta in anchor_eta_values:
            alpha = alpha_from_eta_np(eta, M)
            r = RobustSubsonicShootingSolver(alpha=float(alpha), Mach=float(M)).solve()
            rows.append(
                {
                    "Mach": float(M),
                    "eta": float(eta),
                    "alpha": float(alpha),
                    "ci": float(r.ci),
                    "omega_i": float(alpha) * float(r.ci),
                    "alpha_cut": alpha_cut_np(M),
                }
            )

    return pd.DataFrame(rows)


def sample_batch_eta(
    n: int,
    ymax: float,
    central_ymax: float,
    mach_min: float,
    mach_max: float,
    eta_min: float,
    eta_max: float,
    device,
):
    dtype = torch.float64

    n1 = n // 2
    n2 = n - n1

    y_uniform = -ymax + 2.0 * ymax * torch.rand(n1, 1, device=device, dtype=dtype)
    y_central = -central_ymax + 2.0 * central_ymax * torch.rand(n2, 1, device=device, dtype=dtype)
    y = torch.cat([y_uniform, y_central], dim=0)
    y.requires_grad_(True)

    mach = mach_min + (mach_max - mach_min) * torch.rand(n, 1, device=device, dtype=dtype)
    eta = eta_min + (eta_max - eta_min) * torch.rand(n, 1, device=device, dtype=dtype)

    alpha = eta * torch_alpha_cut(mach)

    return y, alpha, mach, eta


def parity_loss_eta(
    field: FieldPQNet,
    n: int,
    sym_ymax: float,
    mach_min: float,
    mach_max: float,
    eta_min: float,
    eta_max: float,
    device,
):
    dtype = torch.float64

    y = sym_ymax * torch.rand(n, 1, device=device, dtype=dtype)
    mach = mach_min + (mach_max - mach_min) * torch.rand(n, 1, device=device, dtype=dtype)
    eta = eta_min + (eta_max - eta_min) * torch.rand(n, 1, device=device, dtype=dtype)
    alpha = eta * torch_alpha_cut(mach)

    p_plus, q_plus = field(y, alpha, mach)
    p_minus, q_minus = field(-y, alpha, mach)

    return (
        real_mse(p_minus.real - p_plus.real)
        + real_mse(p_minus.imag + p_plus.imag)
        + real_mse(q_minus.real + q_plus.real)
        + real_mse(q_minus.imag - q_plus.imag)
    )


def ci_anchor_check(ci_provider: nn.Module, anchor_df: pd.DataFrame, device) -> dict:
    dtype = torch.float64
    a = torch.tensor(anchor_df["alpha"].to_numpy(), device=device, dtype=dtype).view(-1, 1)
    m = torch.tensor(anchor_df["Mach"].to_numpy(), device=device, dtype=dtype).view(-1, 1)
    ci_ref = torch.tensor(anchor_df["ci"].to_numpy(), device=device, dtype=dtype).view(-1, 1)

    with torch.no_grad():
        ci_pred = ci_provider(a, m)
        rel = (ci_pred - ci_ref) / torch.clamp(ci_ref.abs(), min=1e-12)

    return {
        "ci_anchor_rel_linf": float(torch.max(torch.abs(rel)).detach().cpu()),
        "ci_anchor_rel_rmse": float(torch.sqrt(torch.mean(rel**2)).detach().cpu()),
    }


def run_diagnostics_eta(args, field, ci_provider, anchor_df, device):
    outdir = Path(args.output_dir)
    diag_rows = []

    eval_mach_values = parse_float_list(args.eval_mach_values)
    eval_eta_values = parse_float_list(args.eval_eta_values)

    for Mach in eval_mach_values:
        for eta in eval_eta_values:
            alpha = alpha_from_eta_np(eta, Mach)

            robust = RobustSubsonicShootingSolver(alpha=alpha, Mach=Mach).solve()
            ci_ref = float(robust.ci)

            y, p_ref, q_ref = classical_pressure_reference(
                alpha=alpha,
                Mach=Mach,
                ci=ci_ref,
                ymax=args.central_ymax,
                n_y=args.n_y,
            )

            p_pred, q_pred, ci_pred = eval_model(field, ci_provider, y, alpha, Mach, device)

            mask = np.abs(p_ref) >= args.amp_mask_frac * np.nanmax(np.abs(p_ref))
            scale = align_complex(p_pred, p_ref, mask)
            p_pred = scale * p_pred
            q_pred = scale * q_pred

            rho_ref, u_ref, v_ref, gamma_ref = fields_from_pq(y, p_ref, q_ref, alpha, Mach, ci_ref)
            rho_pred, u_pred, v_pred, gamma_pred = fields_from_pq(y, p_pred, q_pred, alpha, Mach, ci_pred)

            row = {
                "Mach": float(Mach),
                "eta": float(eta),
                "alpha": float(alpha),
                "alpha_cut": alpha_cut_np(Mach),
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

            tag = f"M{int(round(1000*Mach)):04d}_eta{int(round(1000*eta)):04d}_a{int(round(1000*alpha)):04d}"

            plot_complex_pair(y, p_ref, p_pred, f"Pressure p, M={Mach:g}, eta={eta:g}, alpha={alpha:g}", outdir / f"pressure_p_{tag}.png", "p")
            plot_complex_pair(y, q_ref, q_pred, f"Derivative q=p_y, M={Mach:g}, eta={eta:g}, alpha={alpha:g}", outdir / f"derivative_q_{tag}.png", "q")
            plot_complex_pair(y, u_ref, u_pred, f"Velocity u, M={Mach:g}, eta={eta:g}, alpha={alpha:g}", outdir / f"velocity_u_{tag}.png", "u")
            plot_complex_pair(y, v_ref, v_pred, f"Velocity v, M={Mach:g}, eta={eta:g}, alpha={alpha:g}", outdir / f"velocity_v_{tag}.png", "v")

            pd.DataFrame({
                "y": y,
                "Mach": float(Mach),
                "eta": float(eta),
                "alpha": float(alpha),
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

    ap.add_argument("--mach-min", type=float, default=0.05)
    ap.add_argument("--mach-max", type=float, default=0.95)
    ap.add_argument("--eta-min", type=float, default=0.20)
    ap.add_argument("--eta-max", type=float, default=0.85)

    ap.add_argument("--anchor-mach-values", default="0.1 0.3 0.5 0.7 0.9")
    ap.add_argument("--anchor-eta-values", default="0.25 0.45 0.65 0.80")

    ap.add_argument("--eval-mach-values", default="0.1 0.3 0.5 0.7 0.9")
    ap.add_argument("--eval-eta-values", default="0.25 0.45 0.65 0.80")

    ap.add_argument("--ymax", type=float, default=100.0)
    ap.add_argument("--central-ymax", type=float, default=25.0)
    ap.add_argument("--sym-ymax", type=float, default=25.0)

    ap.add_argument("--epochs", type=int, default=15000)
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--n-sym", type=int, default=768)
    ap.add_argument("--n-y", type=int, default=5001)

    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--n-freq", type=int, default=12)

    ap.add_argument("--w-compat", type=float, default=20.0)
    ap.add_argument("--w-ode", type=float, default=1.0)
    ap.add_argument("--w-bc", type=float, default=30.0)
    ap.add_argument("--w-gauge", type=float, default=100.0)
    ap.add_argument("--w-q-center", type=float, default=2.0)
    ap.add_argument("--w-parity", type=float, default=0.2)

    ap.add_argument("--amp-mask-frac", type=float, default=0.05)

    ap.add_argument("--ci-idw-eta-scale", type=float, default=0.25)
    ap.add_argument("--ci-idw-mach-scale", type=float, default=0.25)
    ap.add_argument("--ci-idw-power", type=float, default=4.0)

    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1234)
    np.random.seed(1234)

    anchor_mach_values = parse_float_list(args.anchor_mach_values)
    anchor_eta_values = parse_float_list(args.anchor_eta_values)

    # Global alpha range induced by eta and Mach ranges.
    alpha_min = alpha_from_eta_np(args.eta_min, args.mach_max)
    alpha_max = alpha_from_eta_np(args.eta_max, args.mach_min)

    print("Subsonic continuous 2D p/q PINN for seed-GEP")
    print(f"output_dir={outdir}")
    print(f"device={device}")
    print(f"Mach range=[{args.mach_min}, {args.mach_max}]")
    print(f"eta range=[{args.eta_min}, {args.eta_max}]")
    print(f"induced alpha range approx=[{alpha_min}, {alpha_max}]")
    print(f"anchor_mach_values={anchor_mach_values}")
    print(f"anchor_eta_values={anchor_eta_values}")

    anchor_df = build_ci_anchor_table_eta(anchor_mach_values, anchor_eta_values)
    anchor_df.to_csv(outdir / "ci_anchor_points.csv", index=False)

    print("\nCI anchors:")
    print(anchor_df.to_string(index=False))

    field = FieldPQNet(
        ymax=args.ymax,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        mach_min=args.mach_min,
        mach_max=args.mach_max,
        width=args.width,
        depth=args.depth,
        n_freq=args.n_freq,
    ).to(device=device, dtype=torch.float64)

    ci_provider = CiGridIDW(
        anchor_df=anchor_df,
        eta_scale=args.ci_idw_eta_scale,
        mach_scale=args.ci_idw_mach_scale,
        power=args.ci_idw_power,
    ).to(device=device, dtype=torch.float64)

    ci_check = ci_anchor_check(ci_provider, anchor_df, device)
    print("\nCI interpolant anchor check:")
    print(ci_check)

    opt = torch.optim.AdamW(
        field.parameters(),
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
        opt.zero_grad(set_to_none=True)

        y, alpha, mach, eta = sample_batch_eta(
            n=args.n_train,
            ymax=args.ymax,
            central_ymax=args.central_ymax,
            mach_min=args.mach_min,
            mach_max=args.mach_max,
            eta_min=args.eta_min,
            eta_max=args.eta_max,
            device=device,
        )

        terms = physics_losses(
            field=field,
            ci_net=ci_provider,
            y=y,
            alpha=alpha,
            mach=mach,
            ymax=args.ymax,
            detach_ci=False,
        )

        if args.w_parity > 0.0:
            loss_parity = parity_loss_eta(
                field=field,
                n=args.n_sym,
                sym_ymax=args.sym_ymax,
                mach_min=args.mach_min,
                mach_max=args.mach_max,
                eta_min=args.eta_min,
                eta_max=args.eta_max,
                device=device,
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
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 10.0)
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
                    "args": vars(args),
                    "anchor_df": anchor_df.to_dict(orient="list"),
                    "ci_check": ci_check,
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
                f"ci_mean={row['ci_batch_mean']:.3e}"
            )

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)

    ckpt = torch.load(outdir / "model_best.pt", map_location=device)
    field.load_state_dict(ckpt["field_state_dict"])
    field.eval()
    ci_provider.eval()

    diag = run_diagnostics_eta(args, field, ci_provider, anchor_df, device)

    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ["loss", "compat", "ode", "bc", "gauge", "q_center", "parity"]:
        ax.semilogy(h["epoch"], h[col], label=col)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss term")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_history.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    with open(outdir / "README.md", "w") as f:
        f.write("# Continuous subsonic 2D p/q PINN for seed-GEP\n\n")
        f.write(f"- Mach range: [{args.mach_min}, {args.mach_max}]\n")
        f.write(f"- eta range: [{args.eta_min}, {args.eta_max}]\n")
        f.write("- eta = alpha / sqrt(1-Mach^2)\n")
        f.write("- alpha = eta * sqrt(1-Mach^2)\n")
        f.write("- trained outputs: Re p, Im p, Re q, Im q\n")
        f.write("- c_i source: fixed sparse 2D IDW interpolation from scalar classical anchors\n")
        f.write("- no dense modal field supervision on p/rho/u/v/gamma/q during training\n\n")
        f.write(f"- best_epoch: {best_epoch}\n")
        f.write(f"- best_loss: {best_loss}\n")
        f.write(f"- ci_check: {ci_check}\n\n")
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
