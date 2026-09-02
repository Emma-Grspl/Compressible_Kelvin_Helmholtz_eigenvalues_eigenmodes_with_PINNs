#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve


def cplx(df, re, im):
    return (
        pd.to_numeric(df[re], errors="coerce").to_numpy(float)
        + 1j * pd.to_numeric(df[im], errors="coerce").to_numpy(float)
    )


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


def second_diff_penalty_sparse(n):
    if n < 3:
        return sparse.csr_matrix((n, n), dtype=float)

    e = np.ones(n)
    L = sparse.diags(
        diagonals=[e[:-2], -2.0 * e[:-2], e[:-2]],
        offsets=[0, 1, 2],
        shape=(n - 2, n),
        format="csr",
    )
    return (L.T @ L).tocsr()


def global_smooth_invert_sparse(G, h, R, lam, bc_weight=1e-6):
    n = len(h)

    data_diag = np.abs(G) ** 2
    med_data = max(float(np.nanmedian(data_diag)), 1e-16)

    R_diag = R.diagonal()
    med_R = max(float(np.nanmedian(R_diag[R_diag > 0])) if np.any(R_diag > 0) else 1.0, 1e-16)

    bc = np.zeros(n, dtype=float)
    bc[0] = float(bc_weight) * med_data
    bc[-1] = float(bc_weight) * med_data

    A = sparse.diags(data_diag + bc, 0, format="csr", dtype=np.complex128)
    if lam > 0:
        A = A + (float(lam) * med_data / med_R) * R.astype(np.complex128)

    b = np.conj(G) * h

    return spsolve(A.tocsc(), b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--central-ymax", type=float, default=20.0)
    ap.add_argument("--bc-weight", type=float, default=1e-6)
    ap.add_argument(
        "--lambdas",
        type=str,
        default="0,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1",
    )
    args = ap.parse_args()

    run = args.run_dir
    out = run / "global_scaledD_reconstruction_FAST"
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(run.glob("fields_vs_classic_*.csv"))
    lambdas = [float(x) for x in args.lambdas.split(",")]

    all_rows = []

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

        print(f"[INFO] alpha={alpha} n_y={len(y)} n_lambda={len(lambdas)}", flush=True)

        U = np.tanh(y)
        Uy = 1.0 - U * U
        D = U - (cr + 1j * ci)

        p_pred = cplx(df, "p_pred_real", "p_pred_imag")
        q_pred = cplx(df, "q_pred_real", "q_pred_imag")

        u_ref = cplx(df, "u_ref_real", "u_ref_imag")
        v_ref = cplx(df, "v_ref_real", "v_ref_imag")
        u_raw = cplx(df, "u_pred_real", "u_pred_imag")
        v_raw = cplx(df, "v_pred_real", "v_pred_imag")

        mask = np.isfinite(y) & (np.abs(y) <= float(args.central_ymax))

        vhat = 1j * q_pred / alpha
        uhat = -D * p_pred - Uy * q_pred / (alpha ** 2)

        uhat = align(uhat, D**2 * u_ref, mask) * uhat
        vhat = align(vhat, D * v_ref, mask) * vhat

        R = second_diff_penalty_sparse(len(y))

        u_raw_a = align(u_raw, u_ref, mask) * u_raw
        v_raw_a = align(v_raw, v_ref, mask) * v_raw

        best_u = None
        best_v = None

        for lam in lambdas:
            print(f"[SOLVE] alpha={alpha} lambda={lam:.3e}", flush=True)

            u_global = global_smooth_invert_sparse(
                D**2, uhat, R, lam, bc_weight=args.bc_weight
            )
            v_global = global_smooth_invert_sparse(
                D, vhat, R, lam, bc_weight=args.bc_weight
            )

            u_global = align(u_global, u_ref, mask) * u_global
            v_global = align(v_global, v_ref, mask) * v_global

            row = {
                "file": str(f),
                "alpha": alpha,
                "Mach": mach,
                "ci": ci,
                "lambda": lam,
                "u_raw_rel_central": rel_l2(u_raw_a, u_ref, mask),
                "u_global_rel_central": rel_l2(u_global, u_ref, mask),
                "v_raw_rel_central": rel_l2(v_raw_a, v_ref, mask),
                "v_global_rel_central": rel_l2(v_global, v_ref, mask),
                "u_scaled_rel_central": rel_l2(D**2 * u_global, D**2 * u_ref, mask),
                "v_scaled_rel_central": rel_l2(D * v_global, D * v_ref, mask),
                "max_abs_u_raw": float(np.nanmax(np.abs(u_raw_a))),
                "max_abs_u_global": float(np.nanmax(np.abs(u_global))),
                "max_abs_v_raw": float(np.nanmax(np.abs(v_raw_a))),
                "max_abs_v_global": float(np.nanmax(np.abs(v_global))),
            }
            all_rows.append(row)

            if best_u is None or row["u_global_rel_central"] < best_u[0]["u_global_rel_central"]:
                best_u = (row, u_global)

            if best_v is None or row["v_global_rel_central"] < best_v[0]["v_global_rel_central"]:
                best_v = (row, v_global)

        best_u_row, best_u_field = best_u
        best_v_row, best_v_field = best_v

        out_df = df.copy()
        out_df["D_real"] = D.real
        out_df["D_imag"] = D.imag
        out_df["absD"] = np.abs(D)
        out_df["u_global_real"] = best_u_field.real
        out_df["u_global_imag"] = best_u_field.imag
        out_df["v_global_real"] = best_v_field.real
        out_df["v_global_imag"] = best_v_field.imag
        out_df["best_lambda_u"] = best_u_row["lambda"]
        out_df["best_lambda_v"] = best_v_row["lambda"]

        out_csv = out / f.name.replace(".csv", "_global_scaledD_FAST.csv")
        out_df.to_csv(out_csv, index=False)

        print("[OK] wrote", out_csv)
        print(
            f"[BEST] alpha={alpha:.3f} "
            f"u_raw={best_u_row['u_raw_rel_central']:.4f} "
            f"u_global={best_u_row['u_global_rel_central']:.4f} "
            f"lambda_u={best_u_row['lambda']:.3e} | "
            f"v_raw={best_v_row['v_raw_rel_central']:.4f} "
            f"v_global={best_v_row['v_global_rel_central']:.4f} "
            f"lambda_v={best_v_row['lambda']:.3e}",
            flush=True,
        )

    if not all_rows:
        raise SystemExit("[FAIL] no usable CSV processed")

    summary = pd.DataFrame(all_rows).sort_values(["alpha", "lambda"])
    summary_path = out / "global_scaledD_reconstruction_summary_FAST.csv"
    summary.to_csv(summary_path, index=False)

    best = (
        summary.sort_values(["alpha", "u_global_rel_central"])
        .groupby(["alpha", "Mach"], as_index=False)
        .first()
    )
    best_path = out / "global_scaledD_best_by_alpha_FAST.csv"
    best.to_csv(best_path, index=False)

    print("\n[OK] wrote", summary_path)
    print("[OK] wrote", best_path)
    print("\n[BEST BY ALPHA]")
    cols = [
        "alpha",
        "lambda",
        "u_raw_rel_central",
        "u_global_rel_central",
        "v_raw_rel_central",
        "v_global_rel_central",
        "u_scaled_rel_central",
        "v_scaled_rel_central",
        "max_abs_u_raw",
        "max_abs_u_global",
    ]
    print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
