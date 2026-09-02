#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import root
import matplotlib.pyplot as plt


def U(y: np.ndarray) -> np.ndarray:
    return np.tanh(y)


def Up(y: np.ndarray) -> np.ndarray:
    u = np.tanh(y)
    return 1.0 - u * u


def csqrt_pos(z: complex) -> complex:
    q = np.sqrt(z + 0j)
    if q.real < 0:
        q = -q
    if abs(q.real) < 1e-14 and q.imag < 0:
        q = -q
    return q


def rhs_pressure(y: float, state: np.ndarray, alpha: float, Mach: float, c: complex) -> np.ndarray:
    p, dp = state
    uy = math.tanh(float(y))
    up = 1.0 - uy * uy

    denom = uy - c
    # Pressure ODE:
    # p'' - 2 U'/(U-c) p' - alpha^2 [1 - M^2 (U-c)^2] p = 0
    ddp = (2.0 * up / denom) * dp + alpha**2 * (1.0 - Mach**2 * denom**2) * p
    return np.array([dp, ddp], dtype=np.complex128)


def integrate_rk4(alpha: float, Mach: float, c: complex, L: float, N: int):
    y = np.linspace(-L, L, N)
    h = y[1] - y[0]

    U_left = -1.0
    q_left = alpha * csqrt_pos(1.0 - Mach**2 * (U_left - c) ** 2)

    state = np.zeros((N, 2), dtype=np.complex128)
    state[0, 0] = 1.0 + 0j
    state[0, 1] = q_left

    for i in range(N - 1):
        yi = y[i]
        s = state[i]

        k1 = rhs_pressure(yi, s, alpha, Mach, c)
        k2 = rhs_pressure(yi + 0.5 * h, s + 0.5 * h * k1, alpha, Mach, c)
        k3 = rhs_pressure(yi + 0.5 * h, s + 0.5 * h * k2, alpha, Mach, c)
        k4 = rhs_pressure(yi + h, s + h * k3, alpha, Mach, c)

        state[i + 1] = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if not np.all(np.isfinite(state[i + 1].real)) or not np.all(np.isfinite(state[i + 1].imag)):
            raise FloatingPointError("non-finite integration state")

    return y, state[:, 0], state[:, 1]


def shooting_residual_vec(x: np.ndarray, alpha: float, Mach: float, L: float, N: int) -> np.ndarray:
    c = complex(float(x[0]), float(x[1]))

    try:
        _, p, dp = integrate_rk4(alpha, Mach, c, L, N)
        U_right = 1.0
        q_right = alpha * csqrt_pos(1.0 - Mach**2 * (U_right - c) ** 2)

        # Right decay condition: p' + q_right p = 0.
        res = dp[-1] + q_right * p[-1]

        scale = max(1.0, abs(dp[-1]), abs(q_right * p[-1]))
        res = res / scale

        return np.array([res.real, res.imag], dtype=float)
    except Exception:
        return np.array([1e6, 1e6], dtype=float)


@dataclass
class SolveResult:
    alpha: float
    Mach: float
    L: float
    N: int
    cr: float
    ci: float
    omega_i: float
    residual_norm: float
    success: bool
    message: str
    nfev: int


def solve_point(alpha: float, Mach: float, cr0: float, ci0: float, L: float, N: int) -> SolveResult:
    sol = root(
        lambda x: shooting_residual_vec(x, alpha, Mach, L, N),
        np.array([cr0, ci0], dtype=float),
        method="hybr",
        tol=1e-11,
        options={"maxfev": 80},
    )

    cr, ci = map(float, sol.x)
    res = shooting_residual_vec(np.array([cr, ci]), alpha, Mach, L, N)
    residual_norm = float(np.linalg.norm(res))

    ok = bool(sol.success and np.isfinite(cr) and np.isfinite(ci) and ci > 0.0 and residual_norm < 1e-7)

    return SolveResult(
        alpha=alpha,
        Mach=Mach,
        L=L,
        N=N,
        cr=cr,
        ci=ci,
        omega_i=alpha * ci,
        residual_norm=residual_norm,
        success=ok,
        message=str(sol.message),
        nfev=int(sol.nfev),
    )


def normalize_mode(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p0 = np.interp(0.0, y, p.real) + 1j * np.interp(0.0, y, p.imag)
    if abs(p0) < 1e-14:
        idx = int(np.argmax(np.abs(p)))
        p0 = p[idx]
    return p / p0


def relative_l2_on_common_domain(
    y: np.ndarray,
    p: np.ndarray,
    y_ref: np.ndarray,
    p_ref: np.ndarray,
    L_common: float,
    n_common: int = 2001,
) -> float:
    yy = np.linspace(-L_common, L_common, n_common)

    pr = np.interp(yy, y, p.real) + 1j * np.interp(yy, y, p.imag)
    rr = np.interp(yy, y_ref, p_ref.real) + 1j * np.interp(yy, y_ref, p_ref.imag)

    num = np.trapz(np.abs(pr - rr) ** 2, yy)
    den = np.trapz(np.abs(rr) ** 2, yy)
    return float(np.sqrt(num / max(den, 1e-300)))


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_points(args) -> pd.DataFrame:
    if args.reference_csv:
        ref = pd.read_csv(args.reference_csv)

        alpha_col = "alpha"
        mach_col = "Mach" if "Mach" in ref.columns else "M"

        cr_col = None
        for c in ["reference_cr", "cr", "c_r", "cr_ref"]:
            if c in ref.columns:
                cr_col = c
                break

        ci_col = None
        for c in ["reference_ci", "ci", "c_i", "ci_ref"]:
            if c in ref.columns:
                ci_col = c
                break

        if cr_col is None or ci_col is None:
            raise ValueError(f"Cannot find cr/ci columns in {args.reference_csv}. Columns: {list(ref.columns)}")

        ref[alpha_col] = pd.to_numeric(ref[alpha_col], errors="coerce")
        ref[mach_col] = pd.to_numeric(ref[mach_col], errors="coerce")
        ref[cr_col] = pd.to_numeric(ref[cr_col], errors="coerce")
        ref[ci_col] = pd.to_numeric(ref[ci_col], errors="coerce")

        if args.alpha_list:
            alphas = set(round(x, 10) for x in parse_float_list(args.alpha_list))
            ref = ref[ref[alpha_col].round(10).isin(alphas)]

        if args.mach_list:
            machs = set(round(x, 10) for x in parse_float_list(args.mach_list))
            ref = ref[ref[mach_col].round(10).isin(machs)]

        out = ref[[alpha_col, mach_col, cr_col, ci_col]].copy()
        out.columns = ["alpha", "Mach", "cr0", "ci0"]
        out = out.dropna().drop_duplicates(["alpha", "Mach"])
        return out.sort_values(["Mach", "alpha"]).reset_index(drop=True)

    if args.alpha is None or args.Mach is None or args.cr0 is None or args.ci0 is None:
        raise ValueError("Either --reference-csv or explicit --alpha --Mach --cr0 --ci0 is required.")

    return pd.DataFrame([{
        "alpha": float(args.alpha),
        "Mach": float(args.Mach),
        "cr0": float(args.cr0),
        "ci0": float(args.ci0),
    }])


def safe_tag(x: float, ndigits: int = 3) -> str:
    return f"{x:.{ndigits}f}".replace(".", "p").replace("-", "m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-csv", type=str, default=None)
    ap.add_argument("--alpha-list", type=str, default=None)
    ap.add_argument("--mach-list", type=str, default=None)

    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--Mach", type=float, default=None)
    ap.add_argument("--cr0", type=float, default=None)
    ap.add_argument("--ci0", type=float, default=None)

    ap.add_argument("--L-list", type=str, default="20,30,40,60,80")
    ap.add_argument("--N-list", type=str, default="1001,2001,4001,8001")
    ap.add_argument("--outdir", type=str, default="assets/classic_subsonic/convergence_box_resolution")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    points = load_points(args)
    Ls = parse_float_list(args.L_list)
    Ns = parse_int_list(args.N_list)

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for _, row in points.iterrows():
        alpha = float(row["alpha"])
        Mach = float(row["Mach"])
        cr0 = float(row["cr0"])
        ci0 = float(row["ci0"])

        point_tag = f"a{safe_tag(alpha)}_M{safe_tag(Mach)}"
        point_dir = out_root / point_tag
        point_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== POINT alpha={alpha:.6g}, Mach={Mach:.6g}, initial c={cr0:.12g}+{ci0:.12g}i ===")

        rows = []
        modes = {}

        for L in Ls:
            for N in Ns:
                print(f"Solving L={L:g}, N={N} ...", flush=True)
                r = solve_point(alpha, Mach, cr0, ci0, L, N)

                rows.append(r.__dict__)
                all_rows.append({**r.__dict__, "point_tag": point_tag})

                print(
                    f"  success={r.success} cr={r.cr:.12g} ci={r.ci:.12g} "
                    f"omega_i={r.omega_i:.12g} residual={r.residual_norm:.3e}"
                )

                if r.success:
                    try:
                        y, p, dp = integrate_rk4(alpha, Mach, complex(r.cr, r.ci), L, N)
                        p = normalize_mode(y, p)
                        modes[(L, N)] = (y, p)
                    except Exception as e:
                        print(f"  mode integration failed: {e}")

        df = pd.DataFrame(rows)

        ok = df[df["success"]].copy()
        if ok.empty:
            df.to_csv(point_dir / "convergence_raw.csv", index=False)
            print(f"[WARN] no successful solve for {point_tag}")
            continue

        # Reference = largest L, largest N among successful runs.
        ok = ok.sort_values(["L", "N"])
        ref_row = ok.iloc[-1]
        L_ref = float(ref_row["L"])
        N_ref = int(ref_row["N"])
        c_ref = complex(float(ref_row["cr"]), float(ref_row["ci"]))
        omega_i_ref = float(ref_row["omega_i"])

        if (L_ref, N_ref) not in modes:
            y_ref, p_ref, _ = integrate_rk4(alpha, Mach, c_ref, L_ref, N_ref)
            p_ref = normalize_mode(y_ref, p_ref)
        else:
            y_ref, p_ref = modes[(L_ref, N_ref)]

        eig_err_abs = []
        eig_err_rel = []
        omega_i_err_abs = []
        mode_l2 = []

        for _, rr in df.iterrows():
            if not bool(rr["success"]):
                eig_err_abs.append(np.nan)
                eig_err_rel.append(np.nan)
                omega_i_err_abs.append(np.nan)
                mode_l2.append(np.nan)
                continue

            c = complex(float(rr["cr"]), float(rr["ci"]))
            eig_err_abs.append(abs(c - c_ref))
            eig_err_rel.append(abs(c - c_ref) / max(abs(c_ref), 1e-300))
            omega_i_err_abs.append(abs(float(rr["omega_i"]) - omega_i_ref))

            key = (float(rr["L"]), int(rr["N"]))
            if key in modes:
                y, p = modes[key]
            else:
                y, p, _ = integrate_rk4(alpha, Mach, c, float(rr["L"]), int(rr["N"]))
                p = normalize_mode(y, p)

            L_common = min(float(rr["L"]), L_ref, 20.0)
            mode_l2.append(relative_l2_on_common_domain(y, p, y_ref, p_ref, L_common=L_common))

        df["reference_L"] = L_ref
        df["reference_N"] = N_ref
        df["reference_cr"] = c_ref.real
        df["reference_ci"] = c_ref.imag
        df["reference_omega_i"] = omega_i_ref
        df["eig_abs_error"] = eig_err_abs
        df["eig_rel_error"] = eig_err_rel
        df["omega_i_abs_error"] = omega_i_err_abs
        df["mode_p_rel_l2_on_common_domain"] = mode_l2

        df.to_csv(point_dir / "convergence_raw.csv", index=False)

        summary_cols = [
            "alpha", "Mach", "L", "N", "cr", "ci", "omega_i",
            "residual_norm", "success",
            "eig_abs_error", "eig_rel_error", "omega_i_abs_error",
            "mode_p_rel_l2_on_common_domain",
        ]
        df[summary_cols].to_csv(point_dir / "convergence_summary.csv", index=False)

        with open(point_dir / "reference_solution.json", "w") as f:
            json.dump({
                "alpha": alpha,
                "Mach": Mach,
                "reference_L": L_ref,
                "reference_N": N_ref,
                "reference_cr": c_ref.real,
                "reference_ci": c_ref.imag,
                "reference_omega_i": omega_i_ref,
            }, f, indent=2)

        # Plot eigenvalue convergence versus N, grouped by L.
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for L in sorted(df["L"].unique()):
            sub = df[(df["L"] == L) & (df["success"])]
            if len(sub):
                ax.loglog(sub["N"], sub["eig_abs_error"], marker="o", label=f"L={L:g}")
        ax.set_xlabel("N")
        ax.set_ylabel(r"$|c_{L,N} - c_{\mathrm{ref}}|$")
        ax.set_title(f"Subsonic eigenvalue convergence, alpha={alpha:g}, M={Mach:g}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(point_dir / "eigenvalue_convergence_vs_N.png", dpi=200)
        plt.close(fig)

        # Plot box convergence versus L, grouped by N.
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for N in sorted(df["N"].unique()):
            sub = df[(df["N"] == N) & (df["success"])]
            if len(sub):
                ax.semilogy(sub["L"], sub["eig_abs_error"], marker="o", label=f"N={N}")
        ax.set_xlabel("L")
        ax.set_ylabel(r"$|c_{L,N} - c_{\mathrm{ref}}|$")
        ax.set_title(f"Subsonic box convergence, alpha={alpha:g}, M={Mach:g}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(point_dir / "eigenvalue_convergence_vs_L.png", dpi=200)
        plt.close(fig)

        # Plot modal convergence.
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for L in sorted(df["L"].unique()):
            sub = df[(df["L"] == L) & (df["success"])]
            if len(sub):
                ax.loglog(sub["N"], sub["mode_p_rel_l2_on_common_domain"], marker="o", label=f"L={L:g}")
        ax.set_xlabel("N")
        ax.set_ylabel(r"relative $L^2$ error of normalized $p$")
        ax.set_title(f"Subsonic modal convergence, alpha={alpha:g}, M={Mach:g}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(point_dir / "modal_p_convergence_vs_N.png", dpi=200)
        plt.close(fig)

        print(f"[OK] wrote {point_dir}")

    pd.DataFrame(all_rows).to_csv(out_root / "all_points_raw.csv", index=False)
    print(f"\n[done] output root: {out_root}")


if __name__ == "__main__":
    main()
