#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import cmath

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def csqrt_decay(z: complex) -> complex:
    q = cmath.sqrt(z + 0j)
    if q.real < 0:
        q = -q
    if abs(q.real) < 1e-14 and q.imag < 0:
        q = -q
    return q


def rhs(y, s, alpha, Mach, c):
    p, q = s
    U = math.tanh(float(y))
    Up = 1.0 - U * U
    denom = U - c
    A = 2.0 * Up / denom
    B = alpha**2 * (1.0 - Mach**2 * denom**2)
    return np.array([q, A * q + B * p], dtype=np.complex128)


def rk4_integrate(y_grid, s0, alpha, Mach, c):
    state = np.zeros((len(y_grid), 2), dtype=np.complex128)
    state[0] = s0

    for i in range(len(y_grid) - 1):
        y = y_grid[i]
        h = y_grid[i + 1] - y_grid[i]
        s = state[i]
        k1 = rhs(y, s, alpha, Mach, c)
        k2 = rhs(y + 0.5 * h, s + 0.5 * h * k1, alpha, Mach, c)
        k3 = rhs(y + 0.5 * h, s + 0.5 * h * k2, alpha, Mach, c)
        k4 = rhs(y + h, s + h * k3, alpha, Mach, c)
        state[i + 1] = s + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return state


def build_twosided(alpha, Mach, cr, ci, ymax, n_y):
    c = cr + 1j * ci

    n_half = n_y // 2 + 1
    y_left = np.linspace(-ymax, 0.0, n_half)
    y_right_desc = np.linspace(ymax, 0.0, n_half)

    lam_left = alpha * csqrt_decay(1.0 - Mach**2 * ((-1.0) - c) ** 2)
    lam_right = alpha * csqrt_decay(1.0 - Mach**2 * ((+1.0) - c) ** 2)

    # Left decaying/outgoing branch: q = +lambda_left p at y=-L.
    left = rk4_integrate(
        y_left,
        np.array([1.0 + 0j, lam_left], dtype=np.complex128),
        alpha, Mach, c,
    )

    # Right decaying/outgoing branch: q = -lambda_right p at y=+L.
    right_desc = rk4_integrate(
        y_right_desc,
        np.array([1.0 + 0j, -lam_right], dtype=np.complex128),
        alpha, Mach, c,
    )

    p0_left, q0_left = left[-1, 0], left[-1, 1]
    p0_right, q0_right = right_desc[-1, 0], right_desc[-1, 1]

    gamma_left = q0_left / p0_left
    gamma_right = q0_right / p0_right

    left = left / p0_left
    right_desc = right_desc / p0_right

    y_right = y_right_desc[::-1]
    right = right_desc[::-1]

    y = np.concatenate([y_left[:-1], y_right])
    p = np.concatenate([left[:-1, 0], right[:, 0]])
    q = np.concatenate([left[:-1, 1], right[:, 1]])

    info = {
        "lambda_left": lam_left,
        "lambda_right": lam_right,
        "p0_left": p0_left,
        "p0_right": p0_right,
        "gamma_left_at_0": gamma_left,
        "gamma_right_at_0": gamma_right,
        "gamma_jump_abs": abs(gamma_left - gamma_right),
        "gamma_jump_rel": abs(gamma_left - gamma_right) / max(abs(gamma_left), abs(gamma_right), 1e-300),
    }

    return y, p, q, info


def build_left_to_right(alpha, Mach, cr, ci, ymax, n_y):
    c = cr + 1j * ci
    y = np.linspace(-ymax, ymax, n_y)

    lam_left = alpha * csqrt_decay(1.0 - Mach**2 * ((-1.0) - c) ** 2)
    state = rk4_integrate(
        y,
        np.array([1.0 + 0j, lam_left], dtype=np.complex128),
        alpha, Mach, c,
    )

    p = state[:, 0]
    q = state[:, 1]

    p0 = np.interp(0.0, y, p.real) + 1j * np.interp(0.0, y, p.imag)
    p = p / p0
    q = q / p0
    return y, p, q


def plot_pair(y, z, title, path, ylabel):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(y, z.real)
    axes[0].set_ylabel(f"Re {ylabel}")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(y, z.imag)
    axes[1].set_ylabel(f"Im {ylabel}")
    axes[1].set_xlabel("y")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_compare(y_a, z_a, y_b, z_b, title, path, ylabel):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(y_a, z_a.real, label="two-sided Re")
    axes[0].plot(y_b, z_b.real, "--", label="left-to-right Re")
    axes[0].set_ylabel(f"Re {ylabel}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(y_a, z_a.imag, label="two-sided Im")
    axes[1].plot(y_b, z_b.imag, "--", label="left-to-right Im")
    axes[1].set_ylabel(f"Im {ylabel}")
    axes[1].set_xlabel("y")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--mach", type=float, default=1.6)
    ap.add_argument("--cr", type=float, default=0.299805)
    ap.add_argument("--ci", type=float, default=0.029118)
    ap.add_argument("--ymax", type=float, default=120.0)
    ap.add_argument("--n-y", type=int, default=4001)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    y2, p2, q2, info = build_twosided(args.alpha, args.mach, args.cr, args.ci, args.ymax, args.n_y)
    y1, p1, q1 = build_left_to_right(args.alpha, args.mach, args.cr, args.ci, args.ymax, args.n_y)

    df = pd.DataFrame({
        "y": y2,
        "p_twosided_real": p2.real,
        "p_twosided_imag": p2.imag,
        "q_twosided_real": q2.real,
        "q_twosided_imag": q2.imag,
    })
    df.to_csv(outdir / "twosided_reference.csv", index=False)

    info_rows = []
    for k, v in info.items():
        if isinstance(v, complex):
            info_rows.append({"name": k, "real": v.real, "imag": v.imag, "abs": abs(v)})
        else:
            info_rows.append({"name": k, "real": float(v), "imag": 0.0, "abs": abs(float(v))})
    pd.DataFrame(info_rows).to_csv(outdir / "twosided_reference_info.csv", index=False)

    plot_pair(y2, p2, "two-sided pressure p", outdir / "twosided_pressure_p.png", "p")
    plot_pair(y2, q2, "two-sided derivative q=p_y", outdir / "twosided_derivative_q.png", "q")

    plot_compare(y2, p2, y1, p1, "two-sided vs left-to-right pressure p", outdir / "compare_pressure_p.png", "p")
    plot_compare(y2, q2, y1, q1, "two-sided vs left-to-right derivative q", outdir / "compare_derivative_q.png", "q")

    print("[OK] wrote", outdir)
    print("Two-sided matching info:")
    for r in info_rows:
        print(r)


if __name__ == "__main__":
    main()
