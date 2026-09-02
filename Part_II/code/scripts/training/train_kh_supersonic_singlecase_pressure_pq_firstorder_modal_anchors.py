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


BASE = Path(__file__).resolve().parent / "train_kh_supersonic_singlecase_pressure_pq_firstorder.py"
spec = importlib.util.spec_from_file_location("sup_pq_base", BASE)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


def interp_complex(x_new, x, z):
    return np.interp(x_new, x, z.real) + 1j * np.interp(x_new, x, z.imag)


def build_modal_anchors(args, device):
    if args.anchor_points <= 0:
        return None

    y_ref, p_ref, q_ref = base.classical_pressure_reference(
        alpha=args.alpha,
        Mach=args.mach,
        cr=args.cr,
        ci=args.ci,
        ymax=args.diagnostic_ymax,
        n_y=max(args.n_y, 4001),
    )

    if args.anchor_points == 1:
        y_anchor = np.array([0.0], dtype=np.float64)
    else:
        y_anchor = np.linspace(-args.anchor_ymax, args.anchor_ymax, args.anchor_points)

    p_anchor = interp_complex(y_anchor, y_ref, p_ref)
    q_anchor = interp_complex(y_anchor, y_ref, q_ref)

    y_t = torch.tensor(y_anchor[:, None], device=device, dtype=torch.float64)
    p_t = torch.tensor(p_anchor[:, None], device=device, dtype=torch.complex128)
    q_t = torch.tensor(q_anchor[:, None], device=device, dtype=torch.complex128)

    return {
        "y_np": y_anchor,
        "p_np": p_anchor,
        "q_np": q_anchor,
        "y": y_t,
        "p": p_t,
        "q": q_t,
    }


def complex_rel_mse(pred, ref):
    diff = pred - ref
    num = torch.mean(diff.real**2 + diff.imag**2)
    den = torch.mean(ref.real**2 + ref.imag**2).clamp_min(1e-12)
    return num / den


def modal_anchor_loss(model, anchors):
    if anchors is None:
        z = next(model.parameters()).sum() * 0.0
        return z, z

    p_pred, q_pred = model(anchors["y"])
    loss_p = complex_rel_mse(p_pred, anchors["p"])
    loss_q = complex_rel_mse(q_pred, anchors["q"])
    return loss_p, loss_q


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

    ap.add_argument("--anchor-points", type=int, default=0)
    ap.add_argument("--anchor-ymax", type=float, default=12.0)
    ap.add_argument("--w-anchor-p", type=float, default=10.0)
    ap.add_argument("--w-anchor-q", type=float, default=0.0)

    ap.add_argument("--amp-mask-frac", type=float, default=0.05)

    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1234)
    np.random.seed(1234)

    print("Supersonic single-case first-order p/q PINN with sparse modal anchors")
    print(f"alpha={args.alpha} mach={args.mach} cr={args.cr} ci={args.ci}")
    print(f"output_dir={outdir}")
    print(f"device={device}")
    print(
        f"weights: compat={args.w_compat} ode={args.w_ode} bc={args.w_bc} "
        f"gauge={args.w_gauge} q0={args.w_q0} "
        f"anchor_p={args.w_anchor_p} anchor_q={args.w_anchor_q}"
    )
    print(f"anchor_points={args.anchor_points} anchor_ymax={args.anchor_ymax}")

    model = base.FourierMLP(
        ymax=args.ymax,
        width=args.width,
        depth=args.depth,
        n_freq=args.n_freq,
    ).to(device=device, dtype=torch.float64)

    anchors = build_modal_anchors(args, device)

    if anchors is not None:
        anchor_df = pd.DataFrame({
            "y": anchors["y_np"],
            "p_ref_real": anchors["p_np"].real,
            "p_ref_imag": anchors["p_np"].imag,
            "q_ref_real": anchors["q_np"].real,
            "q_ref_imag": anchors["q_np"].imag,
        })
        anchor_df.to_csv(outdir / "modal_anchor_points.csv", index=False)
        print("\nModal anchors:")
        print(anchor_df.to_string(index=False))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8, foreach=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    history = []
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        y = base.sample_y(args.n_train, args.ymax, args.central_ymax, device)

        terms = base.loss_terms(
            model=model,
            y=y,
            alpha=args.alpha,
            Mach=args.mach,
            cr=args.cr,
            ci=args.ci,
            ymax=args.ymax,
            device=device,
        )

        loss_anchor_p, loss_anchor_q = modal_anchor_loss(model, anchors)

        loss = (
            args.w_compat * terms["compat"]
            + args.w_ode * terms["ode"]
            + args.w_bc * terms["bc"]
            + args.w_gauge * terms["gauge"]
            + args.w_q0 * terms["q0"]
            + args.w_anchor_p * loss_anchor_p
            + args.w_anchor_q * loss_anchor_q
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
            "anchor_p": float(loss_anchor_p.detach().cpu()),
            "anchor_q": float(loss_anchor_q.detach().cpu()),
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
                f"bc={row['bc']:.3e} gauge={row['gauge']:.3e} "
                f"anchor_p={row['anchor_p']:.3e} anchor_q={row['anchor_q']:.3e}"
            )

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)

    ckpt = torch.load(outdir / "model_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ["loss", "compat", "ode", "bc", "gauge", "q0", "anchor_p", "anchor_q"]:
        ax.semilogy(h["epoch"], h[col], label=col)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss term")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "01_loss_history.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    row = base.diagnostics(args, model, device, outdir)

    with open(outdir / "README.md", "a") as f:
        f.write("\n## Sparse modal anchor configuration\n\n")
        f.write(f"- anchor_points: {args.anchor_points}\n")
        f.write(f"- anchor_ymax: {args.anchor_ymax}\n")
        f.write(f"- w_anchor_p: {args.w_anchor_p}\n")
        f.write(f"- w_anchor_q: {args.w_anchor_q}\n")
        f.write(f"- best_epoch: {best_epoch}\n")
        f.write(f"- best_loss: {best_loss}\n")

    print("\n[OK] wrote", outdir)
    print(f"best_epoch={best_epoch} best_loss={best_loss:.6e}")
    print("\nDiagnostics:")
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
