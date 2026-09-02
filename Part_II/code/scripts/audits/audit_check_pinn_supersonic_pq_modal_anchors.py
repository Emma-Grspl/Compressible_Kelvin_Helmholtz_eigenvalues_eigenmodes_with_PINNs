#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    args = ap.parse_args()

    d = np.load(args.dataset, allow_pickle=True)

    print("[check] file:", args.dataset)
    print("[check] keys:")
    for k in d.files:
        print(" ", k)

    alpha = d["alpha_anchors"]
    row_alpha = d["row_alpha"]
    y = d["y"]

    p = d["p_real"] + 1j * d["p_imag"]
    q = d["q_real"] + 1j * d["q_imag"]

    print("\n[check] spectral anchors")
    print("  Mach fixed:", float(d["Mach_fixed"]))
    print("  n alpha anchors:", len(alpha))
    print("  alpha min/max:", float(alpha.min()), float(alpha.max()))
    print("  cr min/max:", float(d["cr_ref"].min()), float(d["cr_ref"].max()))
    print("  ci min/max:", float(d["ci_ref"].min()), float(d["ci_ref"].max()))

    if len(alpha) > 1:
        gaps = np.diff(np.sort(alpha))
        print("  alpha gap median:", float(np.median(gaps)))
        print("  alpha gap max:", float(np.max(gaps)))

    print("\n[check] modal rows")
    print("  n rows:", len(y))
    print("  unique row alphas:", len(np.unique(np.round(row_alpha, 12))))
    print("  y min/max:", float(np.min(y)), float(np.max(y)))

    print("\n[check] finite")
    for name, arr in [
        ("p", p),
        ("q", q),
        ("gamma_real", d["gamma_real"]),
        ("gamma_imag", d["gamma_imag"]),
        ("Q_real", d["Q_real"]),
        ("Q_imag", d["Q_imag"]),
    ]:
        finite = np.isfinite(arr).mean()
        print(f"  {name:12s}: {100*finite:.3f}% finite")

    print("\n[check] norms by alpha")
    for a in np.unique(np.round(row_alpha, 12))[:10]:
        m = np.isclose(row_alpha, a, atol=1e-12)
        pnorm = np.sqrt(np.trapz(np.abs(p[m]) ** 2, y[m]))
        qnorm = np.sqrt(np.trapz(np.abs(q[m]) ** 2, y[m]))
        print(f"  alpha={a:.12g}  rows={m.sum():5d}  ||p||={pnorm:.6e}  ||q||={qnorm:.6e}")

    if "metadata_json" in d.files:
        print("\n[check] metadata")
        print(str(d["metadata_json"]))


if __name__ == "__main__":
    main()
