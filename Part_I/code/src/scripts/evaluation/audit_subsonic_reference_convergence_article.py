#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.scripts.classical.solve_robust_subsonic_shooting import RobustSubsonicShootingSolver
from src.scripts.classical.solve_shooting_subsonic import SubsonicShootingSolver
from src.scripts.classical.solve_mstab17_subsonic_solver import Mstab17SubsonicSolver


def parse_floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def make_solver(cls, **kwargs):
    sig = inspect.signature(cls)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**accepted)


def result_to_dict(r):
    if hasattr(r, "__dict__"):
        return dict(r.__dict__)
    if isinstance(r, dict):
        return dict(r)
    return {k: getattr(r, k) for k in dir(r) if not k.startswith("_")}


def solve_primary(alpha, Mach, L, n_scan, rtol, atol):
    solver = make_solver(
        SubsonicShootingSolver,
        alpha=alpha,
        Mach=Mach,
        L=L,
        y_limit=L,
        ci_max=1.0,
        n_scan=n_scan,
        rtol=rtol,
        atol=atol,
    )
    r = solver.solve()
    d = result_to_dict(r)
    ci = float(d.get("ci", d.get("primary_ci", np.nan)))
    omega_i = float(d.get("omega_i", alpha * ci))
    success = bool(d.get("success", np.isfinite(ci)))
    mismatch = float(d.get("mismatch", d.get("residual", np.nan)))
    return {
        "solver": "primary_riccati",
        "alpha": alpha,
        "Mach": Mach,
        "L": L,
        "n_scan": n_scan,
        "rtol": rtol,
        "atol": atol,
        "cr": 0.0,
        "ci": ci,
        "omega_i": omega_i,
        "success": success,
        "mismatch": mismatch,
    }


def solve_secondary(alpha, Mach, y_limit, n_scan, rtol, atol):
    solver = make_solver(
        Mstab17SubsonicSolver,
        alpha=alpha,
        Mach=Mach,
        y_limit=y_limit,
        L=y_limit,
        n_scan=n_scan,
        rtol=rtol,
        atol=atol,
    )
    r = solver.solve()
    d = result_to_dict(r)
    ci = float(d.get("ci", d.get("secondary_ci", np.nan)))
    omega_i = float(d.get("omega_i", alpha * ci))
    success = bool(d.get("success", np.isfinite(ci)))
    mismatch = float(d.get("stage2_mismatch", d.get("mismatch", d.get("residual", np.nan))))
    return {
        "solver": "secondary_mstab17",
        "alpha": alpha,
        "Mach": Mach,
        "L": y_limit,
        "n_scan": n_scan,
        "rtol": rtol,
        "atol": atol,
        "cr": 0.0,
        "ci": ci,
        "omega_i": omega_i,
        "success": success,
        "mismatch": mismatch,
    }


def solve_robust(alpha, Mach):
    r = RobustSubsonicShootingSolver(alpha=alpha, Mach=Mach).solve()
    d = result_to_dict(r)
    ci = float(d.get("ci", np.nan))
    return {
        "alpha": alpha,
        "Mach": Mach,
        "reference_cr": 0.0,
        "reference_ci": ci,
        "reference_omega_i": alpha * ci,
        "reference_source": d.get("source", "robust"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.3,0.5,0.7")
    ap.add_argument("--machs", default="0.5")
    ap.add_argument("--L-list", default="10,15,20,30,40,60,80")
    ap.add_argument("--n-scan-list", default="41,61,81,121,161")
    ap.add_argument("--rtol-list", default="1e-7,1e-8,1e-9,1e-10")
    ap.add_argument("--atol-factor", type=float, default=1e-2)
    ap.add_argument("--outdir", default="assets/classic_subsonic/convergence_box_resolution/article_solver_exact")
    args = ap.parse_args()

    alphas = parse_floats(args.alphas)
    machs = parse_floats(args.machs)
    Ls = parse_floats(args.L_list)
    n_scans = parse_ints(args.n_scan_list)
    rtols = parse_floats(args.rtol_list)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    ref_rows = []

    for Mach in machs:
        for alpha in alphas:
            print(f"\n=== alpha={alpha}, Mach={Mach} ===", flush=True)

            ref = solve_robust(alpha, Mach)
            ref_rows.append(ref)
            ci_ref = ref["reference_ci"]
            omega_ref = ref["reference_omega_i"]

            for L in Ls:
                for n_scan in n_scans:
                    rtol = 1e-8
                    atol = rtol * args.atol_factor

                    for fn in [solve_secondary]:
                        try:
                            row = fn(alpha, Mach, L, n_scan, rtol, atol)
                            row["ci_abs_error_vs_robust"] = abs(row["ci"] - ci_ref)
                            row["omega_i_abs_error_vs_robust"] = abs(row["omega_i"] - omega_ref)
                            rows.append(row)
                            print(
                                f"{row['solver']:18s} L={L:5g} n_scan={n_scan:4d} "
                                f"ci={row['ci']:.12g} err={row['ci_abs_error_vs_robust']:.3e} "
                                f"ok={row['success']}",
                                flush=True,
                            )
                        except Exception as e:
                            rows.append({
                                "solver": fn.__name__,
                                "alpha": alpha,
                                "Mach": Mach,
                                "L": L,
                                "n_scan": n_scan,
                                "rtol": rtol,
                                "atol": atol,
                                "cr": 0.0,
                                "ci": np.nan,
                                "omega_i": np.nan,
                                "success": False,
                                "mismatch": np.nan,
                                "ci_abs_error_vs_robust": np.nan,
                                "omega_i_abs_error_vs_robust": np.nan,
                                "error": repr(e),
                            })
                            print(f"[FAIL] {fn.__name__} L={L} n_scan={n_scan}: {e}", flush=True)

            # tolerance sweep at largest box and fixed n_scan
            L = max(Ls)
            n_scan = max(n_scans)
            for rtol in rtols:
                atol = rtol * args.atol_factor
                for fn in [solve_secondary]:
                    try:
                        row = fn(alpha, Mach, L, n_scan, rtol, atol)
                        row["sweep"] = "tolerance"
                        row["ci_abs_error_vs_robust"] = abs(row["ci"] - ci_ref)
                        row["omega_i_abs_error_vs_robust"] = abs(row["omega_i"] - omega_ref)
                        rows.append(row)
                    except Exception as e:
                        rows.append({
                            "solver": fn.__name__,
                            "alpha": alpha,
                            "Mach": Mach,
                            "L": L,
                            "n_scan": n_scan,
                            "rtol": rtol,
                            "atol": atol,
                            "cr": 0.0,
                            "ci": np.nan,
                            "omega_i": np.nan,
                            "success": False,
                            "mismatch": np.nan,
                            "ci_abs_error_vs_robust": np.nan,
                            "omega_i_abs_error_vs_robust": np.nan,
                            "sweep": "tolerance",
                            "error": repr(e),
                        })

    df = pd.DataFrame(rows)
    refdf = pd.DataFrame(ref_rows)

    df.to_csv(outdir / "subsonic_classical_convergence_raw.csv", index=False)
    refdf.to_csv(outdir / "subsonic_classical_robust_reference_points.csv", index=False)

    summary = (
        df[df["success"].astype(str).str.lower().isin(["true", "1"])]
        .groupby(["solver", "alpha", "Mach"], as_index=False)
        .agg(
            min_ci_abs_error=("ci_abs_error_vs_robust", "min"),
            min_omega_i_abs_error=("omega_i_abs_error_vs_robust", "min"),
            max_L=("L", "max"),
            max_n_scan=("n_scan", "max"),
        )
    )
    summary.to_csv(outdir / "subsonic_classical_convergence_summary.csv", index=False)

    for (solver, alpha, Mach), sub in df.groupby(["solver", "alpha", "Mach"]):
        sub = sub[sub["success"].astype(str).str.lower().isin(["true", "1"])].copy()
        if sub.empty:
            continue

        tag = f"{solver}_a{alpha:.3f}_M{Mach:.3f}".replace(".", "p")

        fig, ax = plt.subplots(figsize=(7, 5))
        for L, ss in sub.groupby("L"):
            ss = ss.sort_values("n_scan")
            ax.loglog(ss["n_scan"], ss["ci_abs_error_vs_robust"], marker="o", label=f"L={L:g}")
        ax.set_xlabel("n_scan")
        ax.set_ylabel(r"$|c_i - c_{i,\mathrm{robust}}|$")
        ax.set_title(f"{solver}, alpha={alpha:g}, M={Mach:g}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{tag}_ci_error_vs_nscan.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        for n_scan, ss in sub.groupby("n_scan"):
            ss = ss.sort_values("L")
            ax.semilogy(ss["L"], ss["ci_abs_error_vs_robust"], marker="o", label=f"n={n_scan}")
        ax.set_xlabel("box half-size L")
        ax.set_ylabel(r"$|c_i - c_{i,\mathrm{robust}}|$")
        ax.set_title(f"{solver}, alpha={alpha:g}, M={Mach:g}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{tag}_ci_error_vs_box.png", dpi=200)
        plt.close(fig)

    print("\n[OK] wrote:", outdir)


if __name__ == "__main__":
    main()
