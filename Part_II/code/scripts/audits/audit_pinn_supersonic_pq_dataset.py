#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def l2_norm(y, f):
    order = np.argsort(y)
    y = y[order]
    f = f[order]
    return float(np.sqrt(np.trapz(np.abs(f) ** 2, y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    d = np.load(args.dataset, allow_pickle=True)

    alpha = d["row_alpha"]
    y = d["y"]
    p = d["p_real"] + 1j * d["p_imag"]
    q = d["q_real"] + 1j * d["q_imag"]

    print("[audit] file:", args.dataset)
    print("[audit] rows:", len(y))
    print("[audit] unique alpha:", len(np.unique(np.round(alpha, 12))))
    print("[audit] y range:", float(y.min()), float(y.max()))

    print("\n[audit] global magnitudes")
    print("  max |p|:", float(np.max(np.abs(p))))
    print("  max |q|:", float(np.max(np.abs(q))))
    print("  median |p|:", float(np.median(np.abs(p))))
    print("  median |q|:", float(np.median(np.abs(q))))
    print("  99.9% |q|:", float(np.quantile(np.abs(q), 0.999)))
    print("  99.99% |q|:", float(np.quantile(np.abs(q), 0.9999)))

    print("\n[audit] top |q| rows")
    idx_top = np.argsort(np.abs(q))[-args.top:][::-1]
    for j in idx_top:
        print(
            f"  alpha={alpha[j]:.12g} "
            f"y={y[j]: .16e} "
            f"|p|={abs(p[j]):.6e} "
            f"|q|={abs(q[j]):.6e} "
            f"p=({p[j].real:.6e},{p[j].imag:.6e}) "
            f"q=({q[j].real:.6e},{q[j].imag:.6e})"
        )

    print("\n[audit] per-alpha grid and norm diagnostics")
    rows = []
    for a in np.unique(np.round(alpha, 12)):
        m = np.isclose(alpha, a, atol=1e-12)
        ya = y[m]
        pa = p[m]
        qa = q[m]

        ys = np.sort(ya)
        dy = np.diff(ys)
        dy_pos = dy[dy > 0]

        rows.append({
            "alpha": float(a),
            "n": int(m.sum()),
            "n_unique_y": int(len(np.unique(ys))),
            "n_duplicate_y": int(len(ys) - len(np.unique(ys))),
            "min_dy": float(dy_pos.min()) if len(dy_pos) else np.nan,
            "p_l2": l2_norm(ya, pa),
            "q_l2": l2_norm(ya, qa),
            "max_abs_q": float(np.max(np.abs(qa))),
            "y_at_max_q": float(ya[np.argmax(np.abs(qa))]),
        })

    print(
        "  alpha        n   unique_y dup_y     min_dy           ||p||          ||q||          max|q|       y(max|q|)"
    )
    for r in rows[:15]:
        print(
            f"  {r['alpha']:<10.6g} "
            f"{r['n']:6d} "
            f"{r['n_unique_y']:8d} "
            f"{r['n_duplicate_y']:5d} "
            f"{r['min_dy']:.3e} "
            f"{r['p_l2']:.6e} "
            f"{r['q_l2']:.6e} "
            f"{r['max_abs_q']:.6e} "
            f"{r['y_at_max_q']:.6e}"
        )

    print("\n[audit] worst alpha by ||q||")
    rows_sorted = sorted(rows, key=lambda r: r["q_l2"], reverse=True)
    for r in rows_sorted[:10]:
        print(
            f"  alpha={r['alpha']:.12g} "
            f"||p||={r['p_l2']:.6e} "
            f"||q||={r['q_l2']:.6e} "
            f"max|q|={r['max_abs_q']:.6e} "
            f"y(max|q|)={r['y_at_max_q']:.6e} "
            f"min_dy={r['min_dy']:.3e} "
            f"dup_y={r['n_duplicate_y']}"
        )


if __name__ == "__main__":
    main()
