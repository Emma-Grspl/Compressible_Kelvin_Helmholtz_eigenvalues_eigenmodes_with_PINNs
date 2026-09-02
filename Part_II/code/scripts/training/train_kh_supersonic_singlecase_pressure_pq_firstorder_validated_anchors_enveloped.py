#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.training.train_kh_supersonic_singlecase_pressure_pq_firstorder import (
    FourierMLP,
    sample_y,
    loss_terms,
)


class EnvelopedFourierMLP(FourierMLP):
    def __init__(
        self,
        ymax,
        width=192,
        depth=6,
        n_freq=10,
        envelope_y_right=50.0,
        envelope_k_right=0.075,
        envelope_y_left=-80.0,
        envelope_k_left=0.010,
    ):
        super().__init__(ymax=ymax, width=width, depth=depth, n_freq=n_freq)
        self.envelope_y_right = float(envelope_y_right)
        self.envelope_k_right = float(envelope_k_right)
        self.envelope_y_left = float(envelope_y_left)
        self.envelope_k_left = float(envelope_k_left)

    def envelope(self, y):
        right = torch.relu(y - self.envelope_y_right)
        left = torch.relu(self.envelope_y_left - y)
        return torch.exp(-self.envelope_k_right * right - self.envelope_k_left * left)

    def forward(self, y):
        p_raw, q_raw = super().forward(y)
        e = self.envelope(y).to(dtype=p_raw.real.dtype)
        return e * p_raw, e * q_raw




def load_validated_anchors(ref_csv, mach, alpha, anchor_ymax, anchor_points, amp_mask_frac):
    df = pd.read_csv(ref_csv, low_memory=False)
    sub = df[
        np.isclose(df["Mach"].astype(float), float(mach), atol=1e-10)
        & np.isclose(df["alpha"].astype(float), float(alpha), atol=1e-10)
    ].copy()

    if sub.empty:
        raise RuntimeError(f"No validated reference for M={mach}, alpha={alpha} in {ref_csv}")

    sub = sub.sort_values("y").drop_duplicates("y", keep="first").reset_index(drop=True)

    y = sub["y"].to_numpy(float)
    p = sub["p_real"].to_numpy(float) + 1j * sub["p_imag"].to_numpy(float)

    if {"q_real", "q_imag"}.issubset(sub.columns):
        q = sub["q_real"].to_numpy(float) + 1j * sub["q_imag"].to_numpy(float)
    else:
        q = np.gradient(p, y)

    p0 = np.interp(0.0, y, p.real) + 1j * np.interp(0.0, y, p.imag)
    if abs(p0) < 1e-12:
        p0 = p[np.nanargmax(np.abs(p))]

    p = p / p0
    q = q / p0

    central = np.abs(y) <= float(anchor_ymax)
    amp = np.abs(p)
    amp_max = np.nanmax(amp[central]) if np.any(central) else np.nanmax(amp)

    mask = central & (amp >= float(amp_mask_frac) * amp_max)
    if mask.sum() < min(20, int(anchor_points)):
        mask = central

    idx_all = np.where(mask)[0]
    if len(idx_all) == 0:
        raise RuntimeError("No anchor points selected")

    n = min(int(anchor_points), len(idx_all))
    take = np.unique(np.round(np.linspace(0, len(idx_all) - 1, n)).astype(int))
    idx = idx_all[take]

    return y[idx], p[idx], q[idx], {
        "n_ref_rows": len(sub),
        "n_anchor": len(idx),
        "p0_real": float(p0.real),
        "p0_imag": float(p0.imag),
        "anchor_y_min": float(y[idx].min()),
        "anchor_y_max": float(y[idx].max()),
    }


def load_validated_tail_anchors(ref_csv, mach, alpha, tail_ymin, tail_ymax, tail_points):
    df = pd.read_csv(ref_csv, low_memory=False)
    sub = df[
        np.isclose(df["Mach"].astype(float), float(mach), atol=1e-10)
        & np.isclose(df["alpha"].astype(float), float(alpha), atol=1e-10)
    ].copy()

    if sub.empty:
        raise RuntimeError(f"No validated reference for M={mach}, alpha={alpha} in {ref_csv}")

    sub = sub.sort_values("y").drop_duplicates("y", keep="first").reset_index(drop=True)

    y = sub["y"].to_numpy(float)
    p = sub["p_real"].to_numpy(float) + 1j * sub["p_imag"].to_numpy(float)

    if {"q_real", "q_imag"}.issubset(sub.columns):
        q = sub["q_real"].to_numpy(float) + 1j * sub["q_imag"].to_numpy(float)
    else:
        q = np.gradient(p, y)

    p0 = np.interp(0.0, y, p.real) + 1j * np.interp(0.0, y, p.imag)
    if abs(p0) < 1e-12:
        p0 = p[np.nanargmax(np.abs(p))]

    p = p / p0
    q = q / p0

    mask = (np.abs(y) >= float(tail_ymin)) & (np.abs(y) <= float(tail_ymax))
    idx_all = np.where(mask)[0]
    if len(idx_all) == 0:
        raise RuntimeError("No tail anchor points selected")

    n = min(int(tail_points), len(idx_all))
    take = np.unique(np.round(np.linspace(0, len(idx_all) - 1, n)).astype(int))
    idx = idx_all[take]

    return y[idx], p[idx], q[idx], {
        "n_tail": len(idx),
        "tail_y_min": float(y[idx].min()),
        "tail_y_max": float(y[idx].max()),
        "tail_ref_max_abs_p": float(np.nanmax(np.abs(p[idx]))),
        "tail_ref_max_abs_q": float(np.nanmax(np.abs(q[idx]))),
    }


def complex_mse_rel(pred, ref):
    num = torch.mean(torch.abs(pred - ref) ** 2)
    den = torch.mean(torch.abs(ref) ** 2).clamp_min(1e-14)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--mach", type=float, required=True)
    ap.add_argument("--cr", type=float, required=True)
    ap.add_argument("--ci", type=float, required=True)
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--ref-csv", default="assets/classic_supersonic/shooting/supersonic_reference_core_local_modal_fields.csv")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ymax", type=float, default=500.0)
    ap.add_argument("--central-ymax", type=float, default=80.0)
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

    ap.add_argument("--anchor-points", type=int, default=256)
    ap.add_argument("--anchor-ymax", type=float, default=80.0)
    ap.add_argument("--w-anchor-p", type=float, default=2.0)
    ap.add_argument("--w-anchor-q", type=float, default=0.5)
    ap.add_argument("--amp-mask-frac", type=float, default=0.05)

    ap.add_argument("--tail-ymin", type=float, default=80.0)
    ap.add_argument("--tail-ymax", type=float, default=500.0)
    ap.add_argument("--tail-points", type=int, default=256)
    ap.add_argument("--w-tail-p", type=float, default=50.0)
    ap.add_argument("--w-tail-q", type=float, default=10.0)
    ap.add_argument("--init-checkpoint", default=None)

    ap.add_argument("--envelope-y-right", type=float, default=50.0)
    ap.add_argument("--envelope-k-right", type=float, default=0.075)
    ap.add_argument("--envelope-y-left", type=float, default=-80.0)
    ap.add_argument("--envelope-k-left", type=float, default=0.010)

    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1234)
    np.random.seed(1234)

    ya_np, pa_np, qa_np, anchor_meta = load_validated_anchors(
        args.ref_csv,
        args.mach,
        args.alpha,
        args.anchor_ymax,
        args.anchor_points,
        args.amp_mask_frac,
    )

    yt_np, pt_np, qt_np, tail_meta = load_validated_tail_anchors(
        args.ref_csv,
        args.mach,
        args.alpha,
        args.tail_ymin,
        args.tail_ymax,
        args.tail_points,
    )

    ya = torch.tensor(ya_np[:, None], dtype=torch.float64, device=device)
    pa = torch.tensor(pa_np[:, None], dtype=torch.complex128, device=device)
    qa = torch.tensor(qa_np[:, None], dtype=torch.complex128, device=device)

    yt = torch.tensor(yt_np[:, None], dtype=torch.float64, device=device)
    pt = torch.tensor(pt_np[:, None], dtype=torch.complex128, device=device)
    qt = torch.tensor(qt_np[:, None], dtype=torch.complex128, device=device)

    model = EnvelopedFourierMLP(
        ymax=args.ymax,
        width=args.width,
        depth=args.depth,
        n_freq=args.n_freq,
        envelope_y_right=args.envelope_y_right,
        envelope_k_right=args.envelope_k_right,
        envelope_y_left=args.envelope_y_left,
        envelope_k_left=args.envelope_k_left,
    ).to(device=device, dtype=torch.float64)

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[OK] initialized from {args.init_checkpoint}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8, foreach=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    print("Supersonic p/q PINN with VALIDATED modal anchors")
    print(f"alpha={args.alpha} mach={args.mach} cr={args.cr} ci={args.ci}")
    print(f"output_dir={outdir}")
    print(f"ref_csv={args.ref_csv}")
    print(f"anchor_meta={anchor_meta}")
    print(f"tail_meta={tail_meta}")
    print(
        f"envelope: y_right={args.envelope_y_right}, k_right={args.envelope_k_right}, "
        f"y_left={args.envelope_y_left}, k_left={args.envelope_k_left}"
    )
    print(
        f"weights: compat={args.w_compat} ode={args.w_ode} bc={args.w_bc} "
        f"gauge={args.w_gauge} q0={args.w_q0} "
        f"anchor_p={args.w_anchor_p} anchor_q={args.w_anchor_q} "
        f"tail_p={args.w_tail_p} tail_q={args.w_tail_q}"
    )

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

        p_anchor, q_anchor = model(ya)
        loss_anchor_p = complex_mse_rel(p_anchor, pa)
        loss_anchor_q = complex_mse_rel(q_anchor, qa)

        p_tail, q_tail = model(yt)

        # Queue loss: absolute MSE normalized by central scale p(0)=1.
        # This prevents tiny reference amplitudes from disappearing in the averaged anchor loss.
        loss_tail_p = torch.mean(torch.abs(p_tail - pt) ** 2)
        loss_tail_q = torch.mean(torch.abs(q_tail - qt) ** 2)

        loss = (
            args.w_compat * terms["compat"]
            + args.w_ode * terms["ode"]
            + args.w_bc * terms["bc"]
            + args.w_gauge * terms["gauge"]
            + args.w_q0 * terms["q0"]
            + args.w_anchor_p * loss_anchor_p
            + args.w_anchor_q * loss_anchor_q
            + args.w_tail_p * loss_tail_p
            + args.w_tail_q * loss_tail_q
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
            "tail_p": float(loss_tail_p.detach().cpu()),
            "tail_q": float(loss_tail_q.detach().cpu()),
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
                    "anchor_meta": anchor_meta,
                    "tail_meta": tail_meta,
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
                f"anch_p={row['anchor_p']:.3e} anch_q={row['anchor_q']:.3e} "
                f"tail_p={row['tail_p']:.3e} tail_q={row['tail_q']:.3e}"
            )

    h = pd.DataFrame(history)
    h.to_csv(outdir / "history.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ["loss", "compat", "ode", "bc", "gauge", "q0", "anchor_p", "anchor_q", "tail_p", "tail_q"]:
        ax.semilogy(h["epoch"], h[col], label=col)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss term")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "01_loss_history.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame([anchor_meta]).to_csv(outdir / "anchor_meta.csv", index=False)
    pd.DataFrame([tail_meta]).to_csv(outdir / "tail_meta.csv", index=False)

    with open(outdir / "README.md", "w") as f:
        f.write("# Supersonic p/q PINN with validated modal anchors\n\n")
        f.write(f"- alpha={args.alpha}\n")
        f.write(f"- Mach={args.mach}\n")
        f.write(f"- cr={args.cr}\n")
        f.write(f"- ci={args.ci}\n")
        f.write(f"- ref_csv={args.ref_csv}\n")
        f.write(f"- best_epoch={best_epoch}\n")
        f.write(f"- best_loss={best_loss:.8e}\n")
        f.write(f"- anchor_meta={anchor_meta}\n")
        f.write(f"- tail_meta={tail_meta}\n")

    print(f"[OK] wrote {outdir}")
    print(f"best_epoch={best_epoch} best_loss={best_loss:.8e}")


if __name__ == "__main__":
    main()
