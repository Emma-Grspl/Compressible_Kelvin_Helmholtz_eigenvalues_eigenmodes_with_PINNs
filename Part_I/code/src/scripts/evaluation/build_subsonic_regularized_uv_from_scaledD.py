#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def cplx(df, re, im):
    return pd.to_numeric(df[re], errors="coerce").to_numpy(float) + 1j * pd.to_numeric(df[im], errors="coerce").to_numpy(float)


def rel_l2(a, b, mask):
    num = np.linalg.norm((a - b)[mask])
    den = max(np.linalg.norm(b[mask]), 1e-14)
    return float(num / den)


def align(pred, ref, mask):
    p = pred[mask]
    r = ref[mask]
    den = np.vdot(p, p)
    if abs(den) < 1e-30:
        return 1.0 + 0.0j
    return np.vdot(p, r) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--central-ymax", type=float, default=20.0)
    ap.add_argument("--eps-fracs", type=str, default="0,0.1,0.2,0.3,0.5,0.8,1.0")
    args = ap.parse_args()

    run = args.run_dir
    out = run / "regularized_uv_scaledD"
    out.mkdir(parents=True, exist_ok=True)

    rows = []

    files = sorted(run.glob("fields_vs_classic_*.csv"))
    if not files:
        raise SystemExit(f"[FAIL] no fields_vs_classic_*.csv found in {run}")

    for f in files:
        try:
            print(f"[READ] {f}", flush=True)
            df = pd.read_csv(f)
        except OSError as e:
            print(f"[SKIP_IO_ERROR] {f}: {e}", flush=True)
            continue
        except Exception as e:
            print(f"[SKIP_READ_ERROR] {f}: {type(e).__name__}: {e}", flush=True)
            continue

        y = pd.to_numeric(df["y"], errors="coerce").to_numpy(float)
        alpha = float(pd.to_numeric(df["alpha"], errors="coerce").dropna().iloc[0])
        mach = float(pd.to_numeric(df["Mach"], errors="coerce").dropna().iloc[0])
        ci = float(pd.to_numeric(df["ci_pred"], errors="coerce").dropna().iloc[0])
        cr = 0.0

        U = np.tanh(y)
        D = U - (cr + 1j * ci)
        absD = np.abs(D)
        min_abs_D = float(np.nanmin(absD))

        p_pred = cplx(df, "p_pred_real", "p_pred_imag")
        q_pred = cplx(df, "q_pred_real", "q_pred_imag")

        u_ref = cplx(df, "u_ref_real", "u_ref_imag")
        v_ref = cplx(df, "v_ref_real", "v_ref_imag")
        u_pred = cplx(df, "u_pred_real", "u_pred_imag")
        v_pred = cplx(df, "v_pred_real", "v_pred_imag")

        mask = np.isfinite(y) & (np.abs(y) <= float(args.central_ymax))

        # Variables régulières directement depuis p/q.
        Uy = 1.0 - U * U
        vhat = 1j * q_pred / alpha
        uhat = -D * p_pred - Uy * q_pred / (alpha**2)

        # Alignement global sur les références, pour comparaison propre.
        s_u = align(u_pred, u_ref, mask)
        s_v = align(v_pred, v_ref, mask)

        u_raw_a = s_u * u_pred
        v_raw_a = s_v * v_pred

        # Align aussi les variables scaled-D depuis p/q.
        s_uhat = align(uhat, D**2 * u_ref, mask)
        s_vhat = align(vhat, D * v_ref, mask)

        uhat_a = s_uhat * uhat
        vhat_a = s_vhat * vhat

        base = df.copy()
        base["D_real"] = D.real
        base["D_imag"] = D.imag
        base["absD"] = absD
        base["uhat_from_pq_real"] = uhat_a.real
        base["uhat_from_pq_imag"] = uhat_a.imag
        base["vhat_from_pq_real"] = vhat_a.real
        base["vhat_from_pq_imag"] = vhat_a.imag

        for eps_frac in [float(x) for x in args.eps_fracs.split(",")]:
            eps = eps_frac * min_abs_D

            # Inversion régularisée :
            # v = vhat / D  devient  v_reg = vhat * conj(D) / (|D|^2 + eps^2)
            # u = uhat / D^2 devient u_reg = uhat * conj(D)^2 / (|D|^2 + eps^2)^2
            denom1 = absD**2 + eps**2
            denom2 = denom1**2

            v_reg = vhat_a * np.conj(D) / denom1
            u_reg = uhat_a * (np.conj(D) ** 2) / denom2

            tag = f"epsfrac{eps_frac:.3f}".replace(".", "p")

            base[f"u_reg_{tag}_real"] = u_reg.real
            base[f"u_reg_{tag}_imag"] = u_reg.imag
            base[f"v_reg_{tag}_real"] = v_reg.real
            base[f"v_reg_{tag}_imag"] = v_reg.imag

            rows.append({
                "file": str(f),
                "alpha": alpha,
                "Mach": mach,
                "ci": ci,
                "min_abs_D": min_abs_D,
                "eps_frac": eps_frac,
                "eps": eps,
                "u_raw_rel_central": rel_l2(u_raw_a, u_ref, mask),
                "u_reg_rel_central": rel_l2(u_reg, u_ref, mask),
                "v_raw_rel_central": rel_l2(v_raw_a, v_ref, mask),
                "v_reg_rel_central": rel_l2(v_reg, v_ref, mask),
                "u_hat_rel_central": rel_l2(uhat_a, D**2 * u_ref, mask),
                "v_hat_rel_central": rel_l2(vhat_a, D * v_ref, mask),
                "max_abs_u_raw": float(np.nanmax(np.abs(u_raw_a))),
                "max_abs_u_reg": float(np.nanmax(np.abs(u_reg))),
                "max_abs_v_raw": float(np.nanmax(np.abs(v_raw_a))),
                "max_abs_v_reg": float(np.nanmax(np.abs(v_reg))),
            })

        out_csv = out / f.name.replace(".csv", "_regularized_uv.csv")
        base.to_csv(out_csv, index=False)
        print("[OK] wrote", out_csv)

    summary = pd.DataFrame(rows).sort_values(["alpha", "eps_frac"])
    summary_path = out / "regularized_uv_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n[OK] wrote", summary_path)
    print("\n[SUMMARY]")
    cols = [
        "alpha",
        "eps_frac",
        "u_raw_rel_central",
        "u_reg_rel_central",
        "v_raw_rel_central",
        "v_reg_rel_central",
        "u_hat_rel_central",
        "v_hat_rel_central",
        "max_abs_u_raw",
        "max_abs_u_reg",
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
