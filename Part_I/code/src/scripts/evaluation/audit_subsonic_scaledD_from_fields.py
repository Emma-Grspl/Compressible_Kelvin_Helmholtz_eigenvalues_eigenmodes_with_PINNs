#!/usr/bin/env python3
from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd


def pick_col(df, candidates, contains=None, required=True):
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}

    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]

    if contains:
        for c in cols:
            lc = c.lower()
            if all(s.lower() in lc for s in contains):
                return c

    if required:
        raise KeyError(
            f"Could not find column among candidates={candidates}, contains={contains}. "
            f"Available columns={cols}"
        )
    return None


def complex_from_cols(df, re_candidates, im_candidates, re_contains=None, im_contains=None, required=True):
    re_col = pick_col(df, re_candidates, re_contains, required=required)
    im_col = pick_col(df, im_candidates, im_contains, required=False)

    if re_col is None:
        if required:
            raise KeyError("missing real column")
        return None

    re = pd.to_numeric(df[re_col], errors="coerce").to_numpy(float)

    if im_col is None:
        im = np.zeros_like(re)
    else:
        im = pd.to_numeric(df[im_col], errors="coerce").to_numpy(float)

    return re + 1j * im


def base_U(y):
    # Kelvin-Helmholtz tanh shear layer used in the current scripts.
    return np.tanh(y)


def base_Uy(y):
    U = np.tanh(y)
    return 1.0 - U * U


def rel_l2(a, b, mask=None):
    if mask is None:
        mask = np.isfinite(a.real) & np.isfinite(a.imag) & np.isfinite(b.real) & np.isfinite(b.imag)
    if mask.sum() < 3:
        return np.nan
    num = np.linalg.norm((a - b)[mask])
    den = max(np.linalg.norm(b[mask]), 1e-14)
    return float(num / den)


def complex_align(pred, ref, mask):
    p = pred[mask]
    r = ref[mask]
    den = np.vdot(p, p)
    if abs(den) < 1e-30:
        return 1.0 + 0.0j
    return np.vdot(p, r) / den


def find_field_csvs(root):
    csvs = sorted(root.rglob("*.csv"))
    out = []
    for f in csvs:
        name = f.name.lower()
        if any(k in name for k in ["field", "profile", "mode", "diagnostic"]):
            try:
                df = pd.read_csv(f, nrows=5)
            except Exception:
                continue
            cols = " ".join(df.columns).lower()
            if "alpha" in cols and ("y" in cols or "xi" in cols):
                out.append(f)
    return out


def infer_alpha_mach(df, fallback_alpha=None, fallback_mach=None):
    alpha_col = pick_col(df, ["alpha", "a"], contains=["alpha"], required=False)
    mach_col = pick_col(df, ["Mach", "mach", "M"], contains=["mach"], required=False)

    if alpha_col is not None:
        alpha = float(pd.to_numeric(df[alpha_col], errors="coerce").dropna().iloc[0])
    elif fallback_alpha is not None:
        alpha = float(fallback_alpha)
    else:
        raise KeyError("missing alpha")

    if mach_col is not None:
        mach = float(pd.to_numeric(df[mach_col], errors="coerce").dropna().iloc[0])
    elif fallback_mach is not None:
        mach = float(fallback_mach)
    else:
        mach = 0.5

    return alpha, mach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--central-ymax", type=float, default=20.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = args.run_dir
    if not root.exists():
        raise SystemExit(f"[FAIL] run dir does not exist: {root}")

    csvs = find_field_csvs(root)
    print(f"[INFO] found {len(csvs)} candidate field/profile CSVs under {root}")

    rows = []

    for f in csvs:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[SKIP] {f}: cannot read CSV: {e}")
            continue

        if args.verbose:
            print("\n" + "=" * 120)
            print("[FILE]", f)
            print("[COLUMNS]", list(df.columns))

        try:
            y_col = pick_col(df, ["y", "Y", "xi", "x"], required=True)
            y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)

            alpha, mach = infer_alpha_mach(df)

            ci_col = pick_col(
                df,
                ["ci_pred", "ci", "c_i", "cimag", "c_imag"],
                contains=["ci"],
                required=False,
            )
            cr_col = pick_col(
                df,
                ["cr_pred", "cr", "c_r", "creal", "c_real"],
                contains=["cr"],
                required=False,
            )

            if ci_col is None:
                print(f"[SKIP] {f}: missing ci column")
                continue

            ci = float(pd.to_numeric(df[ci_col], errors="coerce").dropna().iloc[0])
            cr = 0.0 if cr_col is None else float(pd.to_numeric(df[cr_col], errors="coerce").dropna().iloc[0])

            p_pred = complex_from_cols(
                df,
                ["p_pred_real", "p_real_pred", "pinn_p_real", "p_real", "p_re", "p"],
                ["p_pred_imag", "p_imag_pred", "pinn_p_imag", "p_imag", "p_im"],
                re_contains=["p", "real"],
                im_contains=["p", "imag"],
            )

            q_pred = complex_from_cols(
                df,
                ["q_pred_real", "q_real_pred", "p_y_pred_real", "py_pred_real", "p_y_real", "q_real"],
                ["q_pred_imag", "q_imag_pred", "p_y_pred_imag", "py_pred_imag", "p_y_imag", "q_imag"],
                re_contains=["q", "real"],
                im_contains=["q", "imag"],
                required=False,
            )

            u_pred = complex_from_cols(
                df,
                ["u_pred_real", "u_real_pred", "pinn_u_real", "u_real", "u_re"],
                ["u_pred_imag", "u_imag_pred", "pinn_u_imag", "u_imag", "u_im"],
                re_contains=["u", "real"],
                im_contains=["u", "imag"],
                required=False,
            )

            v_pred = complex_from_cols(
                df,
                ["v_pred_real", "v_real_pred", "pinn_v_real", "v_real", "v_re"],
                ["v_pred_imag", "v_imag_pred", "pinn_v_imag", "v_imag", "v_im"],
                re_contains=["v", "real"],
                im_contains=["v", "imag"],
                required=False,
            )

            p_ref = complex_from_cols(
                df,
                ["p_ref_real", "ref_p_real", "classic_p_real", "p_classic_real"],
                ["p_ref_imag", "ref_p_imag", "classic_p_imag", "p_classic_imag"],
                re_contains=["p", "ref", "real"],
                im_contains=["p", "ref", "imag"],
                required=False,
            )

            u_ref = complex_from_cols(
                df,
                ["u_ref_real", "ref_u_real", "classic_u_real", "u_classic_real"],
                ["u_ref_imag", "ref_u_imag", "classic_u_imag", "u_classic_imag"],
                re_contains=["u", "ref", "real"],
                im_contains=["u", "ref", "imag"],
                required=False,
            )

            v_ref = complex_from_cols(
                df,
                ["v_ref_real", "ref_v_real", "classic_v_real", "v_classic_real"],
                ["v_ref_imag", "ref_v_imag", "classic_v_imag", "v_classic_imag"],
                re_contains=["v", "ref", "real"],
                im_contains=["v", "ref", "imag"],
                required=False,
            )

            U = base_U(y)
            Uy = base_Uy(y)
            c = cr + 1j * ci
            D = U - c

            mask = np.isfinite(y) & (np.abs(y) <= float(args.central_ymax))

            row = {
                "file": str(f),
                "alpha": alpha,
                "Mach": mach,
                "cr": cr,
                "ci": ci,
                "min_abs_D": float(np.nanmin(np.abs(D))),
                "median_abs_D": float(np.nanmedian(np.abs(D))),
                "central_ymax": float(args.central_ymax),
            }

            if u_pred is not None and v_pred is not None:
                uhat_pred = D**2 * u_pred
                vhat_pred = D * v_pred

                if u_ref is not None:
                    scale_u = complex_align(u_pred, u_ref, mask)
                    u_pred_a = scale_u * u_pred
                    uhat_pred_a = D**2 * u_pred_a
                    uhat_ref = D**2 * u_ref

                    row["u_rel_central"] = rel_l2(u_pred_a, u_ref, mask)
                    row["u_hat_D2_rel_central"] = rel_l2(uhat_pred_a, uhat_ref, mask)
                    row["u_align_real"] = float(scale_u.real)
                    row["u_align_imag"] = float(scale_u.imag)

                if v_ref is not None:
                    scale_v = complex_align(v_pred, v_ref, mask)
                    v_pred_a = scale_v * v_pred
                    vhat_pred_a = D * v_pred_a
                    vhat_ref = D * v_ref

                    row["v_rel_central"] = rel_l2(v_pred_a, v_ref, mask)
                    row["v_hat_D_rel_central"] = rel_l2(vhat_pred_a, vhat_ref, mask)
                    row["v_align_real"] = float(scale_v.real)
                    row["v_align_imag"] = float(scale_v.imag)

            if q_pred is not None:
                # Algebraic stable scaled variables directly from p/q.
                # For rho0=1:
                #   D v = i q / alpha
                #   D^2 u = -D p - Uy q / alpha^2
                vhat_from_pq = 1j * q_pred / alpha
                uhat_from_pq = -D * p_pred - Uy * q_pred / (alpha**2)

                row["max_abs_vhat_from_pq"] = float(np.nanmax(np.abs(vhat_from_pq)))
                row["max_abs_uhat_from_pq"] = float(np.nanmax(np.abs(uhat_from_pq)))

                if v_ref is not None:
                    vhat_ref = D * v_ref
                    scale = complex_align(vhat_from_pq, vhat_ref, mask)
                    row["v_hat_from_pq_rel_central"] = rel_l2(scale * vhat_from_pq, vhat_ref, mask)

                if u_ref is not None:
                    uhat_ref = D**2 * u_ref
                    scale = complex_align(uhat_from_pq, uhat_ref, mask)
                    row["u_hat_from_pq_rel_central"] = rel_l2(scale * uhat_from_pq, uhat_ref, mask)

            rows.append(row)

        except Exception as e:
            print(f"[SKIP] {f}: {type(e).__name__}: {e}")
            if args.verbose:
                print("[COLUMNS]", list(df.columns))
            continue

    if not rows:
        raise SystemExit(
            "[FAIL] no usable field CSV found. Run with --verbose and check column names."
        )

    outdir = root / "scaledD_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(rows).sort_values(["alpha", "Mach", "file"])
    out_csv = outdir / "scaledD_diagnostics_summary.csv"
    summary.to_csv(out_csv, index=False)

    print("\n[OK] wrote", out_csv)
    print("\n[SUMMARY]")
    cols = [
        "alpha",
        "Mach",
        "ci",
        "min_abs_D",
        "u_rel_central",
        "u_hat_D2_rel_central",
        "u_hat_from_pq_rel_central",
        "v_rel_central",
        "v_hat_D_rel_central",
        "v_hat_from_pq_rel_central",
        "file",
    ]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
