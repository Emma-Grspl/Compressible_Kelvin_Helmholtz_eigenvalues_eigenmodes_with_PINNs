#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch


BASE = Path(__file__).resolve().parent / "train_kh_subsonic_2d_pressure_pq_firstorder_mini.py"
spec = importlib.util.spec_from_file_location("pqmini", BASE)
pqmini = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pqmini)


def complex_mse(z: torch.Tensor) -> torch.Tensor:
    return torch.mean(z.real**2 + z.imag**2)


def grad_complex(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    gr = torch.autograd.grad(z.real.sum(), y, create_graph=True, retain_graph=True)[0]
    gi = torch.autograd.grad(z.imag.sum(), y, create_graph=True, retain_graph=True)[0]
    return torch.complex(gr, gi)


def second_order_pressure_losses(field, ci_net, y, alpha, mach, ymax: float, detach_ci: bool):
    p, q = field(y, alpha, mach)

    ci = ci_net(alpha, mach)
    ci_phys = ci.detach() if detach_ci else ci
    c = torch.complex(torch.zeros_like(ci_phys), ci_phys)

    dp = grad_complex(p, y)
    d2p = grad_complex(dp, y)

    U = torch.tanh(y)
    Up = 1.0 - U * U
    Uc = torch.complex(U, torch.zeros_like(U))
    Upc = torch.complex(Up, torch.zeros_like(Up))

    denom = Uc - c
    A = 2.0 * Upc / denom
    B = alpha**2 * (1.0 - mach**2 * denom**2)

    # Pressure-only second-order residual:
    # p_yy = A p_y + B p
    r_p2 = d2p - A * dp - B * p

    # Initialize q as p_y, but do not let q drive p yet.
    r_compat = dp - q

    y0 = torch.zeros_like(y)
    y0.requires_grad_(True)
    p0, q0 = field(y0, alpha, mach)
    loss_gauge = torch.mean((p0.real - 1.0) ** 2 + p0.imag**2)
    loss_q_center = torch.mean(q0.real**2)

    # Robin BC using dp, not q, during pressure-only warm-start.
    yL = torch.full_like(y, -ymax)
    yR = torch.full_like(y, ymax)
    yL.requires_grad_(True)
    yR.requires_grad_(True)

    pL, _ = field(yL, alpha, mach)
    pR, _ = field(yR, alpha, mach)

    dpL = grad_complex(pL, yL)
    dpR = grad_complex(pR, yR)

    minus_one = torch.complex(-torch.ones_like(alpha), torch.zeros_like(alpha))
    plus_one = torch.complex(torch.ones_like(alpha), torch.zeros_like(alpha))

    lam_left = alpha * pqmini.torch_sqrt_pos(1.0 - mach**2 * (minus_one - c) ** 2)
    lam_right = alpha * pqmini.torch_sqrt_pos(1.0 - mach**2 * (plus_one - c) ** 2)

    r_bc_left = dpL - lam_left * pL
    r_bc_right = dpR + lam_right * pR

    return {
        "p2": complex_mse(r_p2),
        "compat": complex_mse(r_compat),
        "bc": complex_mse(r_bc_left) + complex_mse(r_bc_right),
        "gauge": loss_gauge,
        "q_center": loss_q_center,
        "ci_mean": ci.mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--mach-values", default="0.5")
    ap.add_argument("--alpha-min", type=float, default=0.3)
    ap.add_argument("--alpha-max", type=float, default=0.7)
    ap.add_argument("--train-alphas", default="0.3 0.5 0.7")
    ap.add_argument("--warm-train-alphas", default="", help="Optional weighted alpha list for pressure-only warm-start. Example: '0.3 0.3 0.3 0.5 0.7'. If empty, uses --train-alphas.")
    ap.add_argument("--anchor-alphas", default="0.3 0.5 0.7")
    ap.add_argument("--eval-alphas", default="0.3 0.5 0.7")

    ap.add_argument("--ymax", type=float, default=75.0)
    ap.add_argument("--central-ymax", type=float, default=15.0)
    ap.add_argument("--sym-ymax", type=float, default=15.0)

    ap.add_argument("--warm-epochs", type=int, default=4000)
    ap.add_argument("--pq-epochs", type=int, default=6000)
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--n-sym", type=int, default=512)
    ap.add_argument("--n-y", type=int, default=4001)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ci-lr", type=float, default=5e-4)
    ap.add_argument("--ci-prefit-steps", type=int, default=2000)

    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n-freq", type=int, default=8)

    # Warm pressure-only phase.
    ap.add_argument("--warm-w-p2", type=float, default=1.0)
    ap.add_argument("--warm-w-compat", type=float, default=10.0)
    ap.add_argument("--warm-w-bc", type=float, default=20.0)
    ap.add_argument("--warm-w-gauge", type=float, default=100.0)
    ap.add_argument("--warm-w-q-center", type=float, default=1.0)
    ap.add_argument("--warm-w-parity", type=float, default=0.1)
    ap.add_argument("--warm-w-ci", type=float, default=1000.0)

    # First-order p/q phase.
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

    mach_values = pqmini.parse_float_list(args.mach_values)
    anchor_alphas = pqmini.parse_float_list(args.anchor_alphas)
    train_alphas = pqmini.parse_float_list(args.train_alphas) if args.train_alphas.strip() else []
    warm_train_alphas = pqmini.parse_float_list(args.warm_train_alphas) if args.warm_train_alphas.strip() else train_alphas

    alpha_min = float(args.alpha_min)
    alpha_max = float(args.alpha_max)
    mach_min = min(mach_values)
    mach_max = max(mach_values)

    print("Subsonic mini-2D first-order p/q PINN with pressure-only warm-start")
    print(f"output_dir={outdir}")
    print(f"device={device}")
    print(f"mach_values={mach_values}")
    print(f"alpha_range=[{alpha_min}, {alpha_max}]")
    print(f"train_alphas={train_alphas}")
    print(f"warm_train_alphas={warm_train_alphas}")
    print(f"anchor_alphas={anchor_alphas}")
    # Force true no-detach experiment: ci remains coupled to the modal residual.
    # This overrides argparse defaults/legacy flags for the current c_i-coupled runs.
    args.detach_ci_in_mode_branch = False
    print(f"detach_ci_in_mode_branch={args.detach_ci_in_mode_branch}")

    anchor_df = pqmini.build_ci_anchor_table(anchor_alphas, mach_values)
    anchor_df.to_csv(outdir / "ci_anchor_points.csv", index=False)
    ci_init = float(anchor_df["ci"].mean())

    print("\nCI anchors:")
    print(anchor_df.to_string(index=False))

    field = pqmini.FieldPQNet(
        ymax=args.ymax,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
        width=args.width,
        depth=args.depth,
        n_freq=args.n_freq,
    ).to(device=device, dtype=torch.float64)

    ci_net = pqmini.CiNet(
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        mach_min=mach_min,
        mach_max=mach_max,
        ci_init=ci_init,
    ).to(device=device, dtype=torch.float64)

    # Prefit ci branch on sparse scalar classical anchors.
    opt_ci = torch.optim.AdamW(ci_net.parameters(), lr=args.ci_lr, weight_decay=1e-10, foreach=False)
    for step in range(1, args.ci_prefit_steps + 1):
        opt_ci.zero_grad(set_to_none=True)
        loss_ci = pqmini.ci_anchor_loss(ci_net, anchor_df, device)
        loss_ci.backward()
        opt_ci.step()
        if step == 1 or step % 500 == 0:
            print(f"CI prefit {step:5d} | rel_mse={float(loss_ci.detach().cpu()):.3e}")

    history = []
    best_loss = float("inf")
    best_epoch = -1
    best_phase = "none"

    # -------------------------
    # Phase A: pressure-only warm start.
    # -------------------------
    print("\n[Phase A] pressure-only warm-start")
    opt = torch.optim.AdamW(
        list(field.parameters()) + list(ci_net.parameters()),
        lr=args.lr,
        weight_decay=1e-8,
        foreach=False,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.warm_epochs, 1), eta_min=args.lr * 0.05
    )

    for epoch in range(1, args.warm_epochs + 1):
        field.train()
        ci_net.train()
        opt.zero_grad(set_to_none=True)

        y, alpha, mach = pqmini.sample_batch(
            n=args.n_train,
            ymax=args.ymax,
            central_ymax=args.central_ymax,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            mach_values=mach_values,
            device=device,
            train_alphas=warm_train_alphas,
        )

        terms = second_order_pressure_losses(
            field=field,
            ci_net=ci_net,
            y=y,
            alpha=alpha,
            mach=mach,
            ymax=args.ymax,
            detach_ci=args.detach_ci_in_mode_branch,
        )

        loss_ci = pqmini.ci_anchor_loss(ci_net, anchor_df, device)

        loss_parity = pqmini.parity_loss(
            field=field,
            n=args.n_sym,
            sym_ymax=args.sym_ymax,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            mach_values=mach_values,
            device=device,
            train_alphas=warm_train_alphas,
        ) if args.warm_w_parity > 0 else torch.tensor(0.0, device=device, dtype=torch.float64)

        loss = (
            args.warm_w_p2 * terms["p2"]
            + args.warm_w_compat * terms["compat"]
            + args.warm_w_bc * terms["bc"]
            + args.warm_w_gauge * terms["gauge"]
            + args.warm_w_q_center * terms["q_center"]
            + args.warm_w_parity * loss_parity
            + args.warm_w_ci * loss_ci
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(field.parameters()) + list(ci_net.parameters()), 10.0)
        opt.step()
        sched.step()

        row = {
            "phase": "warm_p_only",
            "epoch": epoch,
            "global_epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "p2": float(terms["p2"].detach().cpu()),
            "compat": float(terms["compat"].detach().cpu()),
            "ode": np.nan,
            "bc": float(terms["bc"].detach().cpu()),
            "gauge": float(terms["gauge"].detach().cpu()),
            "q_center": float(terms["q_center"].detach().cpu()),
            "parity": float(loss_parity.detach().cpu()),
            "ci_anchor": float(loss_ci.detach().cpu()),
            "lr": float(sched.get_last_lr()[0]),
        }
        history.append(row)

        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_epoch = epoch
            best_phase = "warm_p_only"
            torch.save(
                {
                    "field_state_dict": field.state_dict(),
                    "ci_state_dict": ci_net.state_dict(),
                    "args": vars(args),
                    "anchor_df": anchor_df.to_dict(orient="list"),
                    "best_epoch": best_epoch,
                    "best_phase": best_phase,
                    "best_loss": best_loss,
                },
                outdir / "model_best.pt",
            )

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Warm {epoch:5d} | loss={row['loss']:.3e} "
                f"p2={row['p2']:.3e} compat={row['compat']:.3e} "
                f"bc={row['bc']:.3e} gauge={row['gauge']:.3e} "
                f"q_center={row['q_center']:.3e} parity={row['parity']:.3e} "
                f"ci_anchor={row['ci_anchor']:.3e}"
            )

    torch.save(
        {
            "field_state_dict": field.state_dict(),
            "ci_state_dict": ci_net.state_dict(),
            "args": vars(args),
            "anchor_df": anchor_df.to_dict(orient="list"),
        },
        outdir / "model_after_warmstart.pt",
    )

    # -------------------------
    # Phase B: first-order p/q fine-tune.
    # -------------------------
    print("\n[Phase B] first-order p/q fine-tune")
    opt = torch.optim.AdamW(
        list(field.parameters()) + list(ci_net.parameters()),
        lr=args.lr,
        weight_decay=1e-8,
        foreach=False,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.pq_epochs, 1), eta_min=args.lr * 0.05
    )

    for epoch in range(1, args.pq_epochs + 1):
        field.train()
        ci_net.train()
        opt.zero_grad(set_to_none=True)

        y, alpha, mach = pqmini.sample_batch(
            n=args.n_train,
            ymax=args.ymax,
            central_ymax=args.central_ymax,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            mach_values=mach_values,
            device=device,
            train_alphas=train_alphas,
        )

        terms = pqmini.physics_losses(
            field=field,
            ci_net=ci_net,
            y=y,
            alpha=alpha,
            mach=mach,
            ymax=args.ymax,
            detach_ci=args.detach_ci_in_mode_branch,
        )

        loss_ci = pqmini.ci_anchor_loss(ci_net, anchor_df, device)

        loss_parity = pqmini.parity_loss(
            field=field,
            n=args.n_sym,
            sym_ymax=args.sym_ymax,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            mach_values=mach_values,
            device=device,
            train_alphas=train_alphas,
        ) if args.w_parity > 0 else torch.tensor(0.0, device=device, dtype=torch.float64)

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

        global_epoch = args.warm_epochs + epoch
        row = {
            "phase": "first_order_pq",
            "epoch": epoch,
            "global_epoch": global_epoch,
            "loss": float(loss.detach().cpu()),
            "p2": np.nan,
            "compat": float(terms["compat"].detach().cpu()),
            "ode": float(terms["ode"].detach().cpu()),
            "bc": float(terms["bc"].detach().cpu()),
            "gauge": float(terms["gauge"].detach().cpu()),
            "q_center": float(terms["q_center"].detach().cpu()),
            "parity": float(loss_parity.detach().cpu()),
            "ci_anchor": float(loss_ci.detach().cpu()),
            "lr": float(sched.get_last_lr()[0]),
        }
        history.append(row)

        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_epoch = global_epoch
            best_phase = "first_order_pq"
            torch.save(
                {
                    "field_state_dict": field.state_dict(),
                    "ci_state_dict": ci_net.state_dict(),
                    "args": vars(args),
                    "anchor_df": anchor_df.to_dict(orient="list"),
                    "best_epoch": best_epoch,
                    "best_phase": best_phase,
                    "best_loss": best_loss,
                },
                outdir / "model_best.pt",
            )

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"PQ {epoch:5d} | loss={row['loss']:.3e} "
                f"compat={row['compat']:.3e} ode={row['ode']:.3e} "
                f"bc={row['bc']:.3e} gauge={row['gauge']:.3e} "
                f"q_center={row['q_center']:.3e} parity={row['parity']:.3e} "
                f"ci_anchor={row['ci_anchor']:.3e}"
            )

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)

    # Use final state, not necessarily lowest scalar training loss.
    torch.save(
        {
            "field_state_dict": field.state_dict(),
            "ci_state_dict": ci_net.state_dict(),
            "args": vars(args),
            "anchor_df": anchor_df.to_dict(orient="list"),
            "best_epoch": args.warm_epochs + args.pq_epochs,
            "best_phase": "final",
            "best_loss": float(history[-1]["loss"]),
        },
        outdir / "model_final.pt",
    )

    # Diagnostics on final state.
    field.eval()
    ci_net.eval()
    diag = pqmini.run_diagnostics(args, field, ci_net, anchor_df, device)

    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ["loss", "p2", "compat", "ode", "bc", "gauge", "q_center", "parity", "ci_anchor"]:
        if col in h.columns:
            vals = pd.to_numeric(h[col], errors="coerce")
            if vals.notna().any():
                ax.semilogy(h["global_epoch"], vals, label=col)
    ax.axvline(args.warm_epochs, linestyle="--", linewidth=1)
    ax.set_xlabel("global epoch")
    ax.set_ylabel("loss term")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_history_bootp.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    with open(outdir / "README.md", "w") as f:
        f.write("# Subsonic mini-2D p/q with pressure-only warm-start\n\n")
        f.write(f"- mach_values: {mach_values}\n")
        f.write(f"- alpha_range: [{alpha_min}, {alpha_max}]\n")
        f.write(f"- train_alphas: {train_alphas}\n")
        f.write(f"- warm_train_alphas: {warm_train_alphas}\n")
        f.write(f"- anchor_alphas: {anchor_alphas}\n")
        f.write("- phase A: pressure-only second-order warm-start for p, with q approx p_y\n")
        f.write("- phase B: first-order p/q fine-tuning\n")
        f.write("- ci supervision: sparse scalar classical ci anchors only\n")
        f.write("- no classical modal field p/rho/u/v/gamma/q is used in training loss\n\n")
        f.write(f"- best_phase_recorded_by_loss: {best_phase}\n")
        f.write(f"- best_epoch_recorded_by_loss: {best_epoch}\n")
        f.write(f"- best_loss_recorded: {best_loss}\n\n")
        f.write("## CI anchors\n\n")
        f.write(anchor_df.to_string(index=False))
        f.write("\n\n## Diagnostics on final state\n\n")
        f.write(diag.to_string(index=False))
        f.write("\n")

    print("\n[OK] wrote", outdir)
    print("\nDiagnostics on final state:")
    print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
