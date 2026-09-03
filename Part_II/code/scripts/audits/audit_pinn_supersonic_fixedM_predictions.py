#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def rel_l2(a, b):
    return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    pred_path = args.run_dir / "modal_predictions_all_rows.npz"
    spec_path = args.run_dir / "spectral_predictions.csv"

    d = np.load(pred_path, allow_pickle=True)
    spec = pd.read_csv(spec_path)

    alpha = d["row_alpha"]
    y = d["y"]

    p_ref = d["p_ref_real"] + 1j * d["p_ref_imag"]
    q_ref = d["q_ref_real"] + 1j * d["q_ref_imag"]
    p_pred = d["p_pred_real"] + 1j * d["p_pred_imag"]
    q_pred = d["q_pred_real"] + 1j * d["q_pred_imag"]

    rows = []

    for a in sorted(np.unique(np.round(alpha, 12))):
        m = np.isclose(alpha, a, atol=1e-12)

        rows.append({
            "alpha": float(a),
            "n": int(m.sum()),
            "p_rel_l2": rel_l2(p_ref[m], p_pred[m]),
            "q_rel_l2": rel_l2(q_ref[m], q_pred[m]),
            "p_max_abs_err": float(np.max(np.abs(p_pred[m] - p_ref[m]))),
            "q_max_abs_err": float(np.max(np.abs(q_pred[m] - q_ref[m]))),
            "p_ref_norm": float(np.linalg.norm(p_ref[m])),
            "q_ref_norm": float(np.linalg.norm(q_ref[m])),
            "y_at_p_max_err": float(y[m][np.argmax(np.abs(p_pred[m] - p_ref[m]))]),
            "y_at_q_max_err": float(y[m][np.argmax(np.abs(q_pred[m] - q_ref[m]))]),
        })

    out = pd.DataFrame(rows)
    out_path = args.run_dir / "modal_error_by_alpha.csv"
    out.to_csv(out_path, index=False)

    print("[audit] wrote:", out_path)

    print("\n[audit] worst p_rel_l2")
    print(out.sort_values("p_rel_l2", ascending=False).head(args.top).to_string(index=False))

    print("\n[audit] worst q_rel_l2")
    print(out.sort_values("q_rel_l2", ascending=False).head(args.top).to_string(index=False))

    print("\n[audit] spectral summary")
    for col in ["cr_abs_err", "ci_abs_err"]:
        if col in spec.columns:
            print(
                f"  {col}: mean={spec[col].mean():.6e} "
                f"max={spec[col].max():.6e}"
            )

    print("\n[audit] worst spectral rows")
    print(spec.sort_values("ci_abs_err", ascending=False).head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
