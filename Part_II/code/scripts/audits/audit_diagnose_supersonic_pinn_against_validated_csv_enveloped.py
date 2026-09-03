#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.training.train_kh_supersonic_singlecase_pressure_pq_firstorder import FourierMLP



def apply_checkpoint_envelope(y_t, p_t, q_t, ckpt):
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if not all(k in args for k in ["envelope_y_right", "envelope_k_right", "envelope_y_left", "envelope_k_left"]):
        return p_t, q_t

    y_right = float(args["envelope_y_right"])
    k_right = float(args["envelope_k_right"])
    y_left = float(args["envelope_y_left"])
    k_left = float(args["envelope_k_left"])

    right = torch.relu(y_t - y_right)
    left = torch.relu(y_left - y_t)
    env = torch.exp(-k_right * right - k_left * left).to(dtype=p_t.real.dtype)

    return env * p_t, env * q_t


def complex_align(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray):
    """Return scale s minimizing ||s*pred - ref|| over mask."""
    p = pred[mask]
    r = ref[mask]
    den = np.vdot(p, p)
    if not np.isfinite(den) or abs(den) < 1e-30:
        return np.nan + 1j * np.nan
    return np.vdot(p, r) / den


def rel_l2(pred: np.ndarray, ref: np.ndarray, y: np.ndarray, mask: np.ndarray):
    e = pred[mask] - ref[mask]
    r = ref[mask]
    yy = y[mask]
    num = np.trapz(np.abs(e) ** 2, yy)
    den = np.trapz(np.abs(r) ** 2, yy)
    if not np.isfinite(den) or den <= 1e-30:
        return np.nan
    return float(np.sqrt(num / den))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ref-csv", default="assets/classic_supersonic/shooting/supersonic_reference_core_local_modal_fields.csv")
    ap.add_argument("--mach", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=0.1625)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n-freq", type=int, default=10)
    ap.add_argument("--ymax", type=float, default=500.0)
    ap.add_argument("--central-ymax", type=float, default=80.0)
    ap.add_argument("--amp-threshold", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "validated_csv_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = pd.read_csv(args.ref_csv, low_memory=False)
    ref = ref[
        np.isclose(ref["Mach"].astype(float), float(args.mach), atol=1e-10)
        & np.isclose(ref["alpha"].astype(float), float(args.alpha), atol=1e-10)
    ].copy()

    if ref.empty:
        raise RuntimeError(f"No reference rows for M={args.mach}, alpha={args.alpha}")

    ref = ref.sort_values("y").drop_duplicates("y", keep="first").reset_index(drop=True)

    y = ref["y"].to_numpy(float)
    p_ref = ref["p_real"].to_numpy(float) + 1j * ref["p_imag"].to_numpy(float)

    if {"q_real", "q_imag"}.issubset(ref.columns):
        q_ref = ref["q_real"].to_numpy(float) + 1j * ref["q_imag"].to_numpy(float)
    else:
        q_ref = np.gradient(p_ref, y)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    model = FourierMLP(width=args.width, depth=args.depth, n_freq=args.n_freq, ymax=args.ymax).to(device=device, dtype=torch.float64)
    ckpt = torch.load(run_dir / "model_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        yt = torch.tensor(y, dtype=torch.float64, device=device).reshape(-1, 1)
        p_t, q_t = model(yt)
    p_t, q_t = apply_checkpoint_envelope(yt, p_t, q_t, ckpt)

    p_pred = p_t.detach().cpu().numpy().reshape(-1)
    q_pred = q_t.detach().cpu().numpy().reshape(-1)
    p_y_pred_num = np.gradient(p_pred, y)

    amp = np.abs(p_ref)
    amp_max = float(np.nanmax(amp))
    mask = (
        np.isfinite(y)
        & np.isfinite(p_ref.real)
        & np.isfinite(p_ref.imag)
        & np.isfinite(p_pred.real)
        & np.isfinite(p_pred.imag)
        & (np.abs(y) <= float(args.central_ymax))
        & (amp >= float(args.amp_threshold) * amp_max)
    )

    if mask.sum() < 20:
        mask = (
            np.isfinite(y)
            & np.isfinite(p_ref.real)
            & np.isfinite(p_ref.imag)
            & np.isfinite(p_pred.real)
            & np.isfinite(p_pred.imag)
            & (np.abs(y) <= float(args.central_ymax))
        )

    scale = complex_align(p_pred, p_ref, mask)
    p_aligned = scale * p_pred
    q_aligned = scale * q_pred
    p_y_pred_num_aligned = scale * p_y_pred_num

    row = {
        "alpha": float(args.alpha),
        "Mach": float(args.mach),
        "n_ref": int(len(ref)),
        "n_mask": int(mask.sum()),
        "central_ymax": float(args.central_ymax),
        "amp_threshold": float(args.amp_threshold),
        "ref_max_abs_p": amp_max,
        "pred_max_abs_p_raw": float(np.nanmax(np.abs(p_pred))),
        "align_scale_real": float(np.real(scale)),
        "align_scale_imag": float(np.imag(scale)),
        "align_scale_abs": float(np.abs(scale)),
        "p_rel": rel_l2(p_aligned, p_ref, y, mask),
        "q_rel": rel_l2(q_aligned, q_ref, y, mask),
        "p_y_num_rel": rel_l2(p_y_pred_num_aligned, q_ref, y, mask),
        "q_vs_dpred_rel": rel_l2(q_aligned, p_y_pred_num_aligned, y, mask),
    }

    pd.DataFrame([row]).to_csv(out_dir / "diagnostics_vs_validated_csv.csv", index=False)

    field = pd.DataFrame({
        "y": y,
        "mask": mask.astype(int),
        "p_ref_real": p_ref.real,
        "p_ref_imag": p_ref.imag,
        "q_ref_real": q_ref.real,
        "q_ref_imag": q_ref.imag,
        "p_pred_raw_real": p_pred.real,
        "p_pred_raw_imag": p_pred.imag,
        "q_pred_raw_real": q_pred.real,
        "q_pred_raw_imag": q_pred.imag,
        "p_pred_aligned_real": p_aligned.real,
        "p_pred_aligned_imag": p_aligned.imag,
        "q_pred_aligned_real": q_aligned.real,
        "q_pred_aligned_imag": q_aligned.imag,
        "p_y_pred_num_aligned_real": p_y_pred_num_aligned.real,
        "p_y_pred_num_aligned_imag": p_y_pred_num_aligned.imag,
    })
    field.to_csv(out_dir / "fields_vs_validated_csv.csv", index=False)

    print(pd.DataFrame([row]).to_string(index=False))
    print("[OK] wrote", out_dir)

    # Plot p.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(y, p_ref.real / amp_max, label="Re p_ref / max|p_ref|", linewidth=1.4)
    ax.plot(y, p_aligned.real / amp_max, "--", label="Re p_PINN aligned / max|p_ref|", linewidth=1.2)
    ax.plot(y, np.abs(p_ref) / amp_max, label="|p_ref| / max|p_ref|", linewidth=1.2)
    ax.plot(y, np.abs(p_aligned) / amp_max, "--", label="|p_PINN aligned| / max|p_ref|", linewidth=1.2)
    ax.axvspan(-args.central_ymax, args.central_ymax, alpha=0.08)
    ax.set_xlim(-args.central_ymax * 1.5, args.central_ymax * 1.5)
    ax.set_xlabel("y")
    ax.set_ylabel("normalized p")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"PINN vs validated CSV: p, M={args.mach}, alpha={args.alpha}")
    fig.tight_layout()
    fig.savefig(out_dir / "p_vs_validated_csv_central.png", dpi=220)
    plt.close(fig)

    # Plot q.
    qnorm = float(np.nanmax(np.abs(q_ref[mask]))) if mask.any() else float(np.nanmax(np.abs(q_ref)))
    if not np.isfinite(qnorm) or qnorm <= 0:
        qnorm = 1.0

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(y, q_ref.real / qnorm, label="Re q_ref", linewidth=1.4)
    ax.plot(y, q_aligned.real / qnorm, "--", label="Re q_PINN aligned", linewidth=1.2)
    ax.plot(y, np.abs(q_ref) / qnorm, label="|q_ref|", linewidth=1.2)
    ax.plot(y, np.abs(q_aligned) / qnorm, "--", label="|q_PINN aligned|", linewidth=1.2)
    ax.axvspan(-args.central_ymax, args.central_ymax, alpha=0.08)
    ax.set_xlim(-args.central_ymax * 1.5, args.central_ymax * 1.5)
    ax.set_xlabel("y")
    ax.set_ylabel("normalized q")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"PINN vs validated CSV: q=p_y, M={args.mach}, alpha={args.alpha}")
    fig.tight_layout()
    fig.savefig(out_dir / "q_vs_validated_csv_central.png", dpi=220)
    plt.close(fig)

    # Full-domain amplitude sanity check.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(y, np.abs(p_ref) / amp_max + 1e-16, label="|p_ref|")
    ax.semilogy(y, np.abs(p_aligned) / amp_max + 1e-16, "--", label="|p_PINN aligned|")
    ax.set_xlabel("y")
    ax.set_ylabel("normalized |p|")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Full-domain amplitude")
    fig.tight_layout()
    fig.savefig(out_dir / "p_amplitude_full_semilogy.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
