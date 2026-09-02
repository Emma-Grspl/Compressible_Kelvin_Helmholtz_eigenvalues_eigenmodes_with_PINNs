#!/usr/bin/env python3
"""
Benchmark final coeur subsonique :

local PINN atlas
  -> ci_pred
  -> dense GEP branch selection
  -> modal comparison against classical reference

This script deliberately uses ci_pred from each selected chart diagnostics.
It does not require evaluating the neural networks again.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def alpha_from_eta(M: float, eta: float) -> float:
    return float(eta * math.sqrt(max(0.0, 1.0 - M * M)))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    den = np.linalg.norm(b)
    if den == 0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / den)


def best_complex_scale(pred: np.ndarray, ref: np.ndarray) -> complex:
    den = np.vdot(pred, pred)
    if abs(den) == 0:
        return 1.0 + 0.0j
    return np.vdot(pred, ref) / den


def align(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return best_complex_scale(pred, ref) * pred


def unpack_classic_full_mode(classic, Mach: float):
    """
    Accepts either:
      - dict with keys y,p,rho,u,v,ci/c
      - tuple/list from older load_classic_full_mode implementations.

    For tuple/list, it detects y as the monotone real grid and detects
    the p/rho pair using rho ~= M^2 p. The remaining two field arrays,
    in original order, are interpreted as u,v.
    """
    if isinstance(classic, dict):
        y = np.asarray(classic["y"]).reshape(-1)
        p = np.asarray(classic["p"]).reshape(-1)
        rho = np.asarray(classic.get("rho", (Mach * Mach) * p)).reshape(-1)
        u = np.asarray(classic["u"]).reshape(-1)
        v = np.asarray(classic["v"]).reshape(-1)

        ci = None
        if "ci" in classic:
            ci = float(classic["ci"])
        elif "c" in classic:
            ci = float(np.imag(classic["c"]))

        return y, p, rho, u, v, ci

    if not isinstance(classic, (tuple, list)):
        raise TypeError(f"Unsupported classic mode type: {type(classic)}")

    items = list(classic)

    # Scalars: possible ci or c.
    ci = None
    for x in items:
        arr = np.asarray(x)
        if arr.shape == ():
            z = complex(arr.item())
            if abs(z.imag) > 0:
                ci = float(z.imag)
            else:
                ci = float(z.real)

    # Candidate arrays.
    arrays = []
    for i, x in enumerate(items):
        arr = np.asarray(x)
        if arr.ndim == 1 and arr.size > 10:
            arrays.append((i, arr.reshape(-1)))

    if not arrays:
        raise RuntimeError("Could not find array entries in classic tuple.")

    # Detect y: real, finite, monotone increasing or decreasing.
    y_idx = None
    y = None
    for i, arr in arrays:
        if np.max(np.abs(np.imag(arr))) > 1e-12:
            continue
        yr = np.real(arr)
        d = np.diff(yr)
        if np.all(np.isfinite(yr)) and (np.all(d > 0) or np.all(d < 0)):
            y_idx = i
            y = yr
            break

    if y is None:
        # Fallback: first real array.
        for i, arr in arrays:
            if np.max(np.abs(np.imag(arr))) < 1e-12:
                y_idx = i
                y = np.real(arr)
                break

    if y is None:
        raise RuntimeError("Could not detect y grid in classic tuple.")

    n = len(y)
    fields = [(i, np.asarray(arr, dtype=np.complex128).reshape(-1))
              for i, arr in arrays
              if i != y_idx and len(arr) == n]

    if len(fields) < 3:
        raise RuntimeError(
            f"Not enough field arrays in classic tuple. "
            f"Tuple length={len(items)}, field_count={len(fields)}"
        )

    # Detect p/rho pair from rho ~= M^2 p.
    best = None
    M2 = Mach * Mach

    for ia, a in fields:
        for ib, b in fields:
            if ia == ib:
                continue

            # hypothesis: a = rho, b = p
            den = np.linalg.norm(a) + 1e-300
            score = np.linalg.norm(a - M2 * b) / den

            if best is None or score < best[0]:
                best = (score, ia, ib)  # rho index, p index

    score, rho_i, p_i = best
    fmap = {i: arr for i, arr in fields}

    if score > 1e-2:
        print(
            f"[WARN] weak p/rho detection in classic tuple: score={score:.3e}. "
            "Continuing with best candidate."
        )

    rho = fmap[rho_i]
    p = fmap[p_i]

    remaining = [(i, arr) for i, arr in fields if i not in {rho_i, p_i}]
    remaining = sorted(remaining, key=lambda t: t[0])

    if len(remaining) < 2:
        raise RuntimeError("Could not identify u/v arrays in classic tuple.")

    u = remaining[0][1]
    v = remaining[1][1]

    print(
        f"[classic tuple] y_idx={y_idx}, p_idx={p_i}, rho_idx={rho_i}, "
        f"u_idx={remaining[0][0]}, v_idx={remaining[1][0]}, "
        f"rho~M2p score={score:.3e}"
    )

    return y, p, rho, u, v, ci


def load_selected_atlas_points(manifest_path: Path) -> pd.DataFrame:
    man = pd.read_csv(manifest_path)

    all_rows = []

    for _, chart in man.iterrows():
        diag_path = Path(chart["path"]) / "diagnostics_summary.csv"
        if not diag_path.exists():
            print(f"[WARN] missing diagnostics: {diag_path}", file=sys.stderr)
            continue

        df = pd.read_csv(diag_path)
        df["chart_id"] = chart["chart_id"]
        df["chart_status"] = chart["status"]
        df["chart_priority"] = int(chart["priority"])
        df["chart_path"] = chart["path"]
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No diagnostics_summary.csv found from manifest charts.")

    data = pd.concat(all_rows, ignore_index=True)

    # Normalize keys for grouping.
    data["Mach_key"] = data["Mach"].round(6)
    data["eta_key"] = data["eta"].round(6)

    # Select the highest priority chart when several charts evaluated the same point.
    data = data.sort_values(["Mach_key", "eta_key", "chart_priority"], ascending=[True, True, False])
    selected = data.drop_duplicates(subset=["Mach_key", "eta_key"], keep="first").copy()

    selected = selected.sort_values(["Mach", "eta"]).reset_index(drop=True)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="assets/pinn_subsonic/local_atlas_v1/atlas_manifest.csv",
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        default="assets/pinn_subsonic/local_atlas_v1/gep_core_atlas_ci_seeded",
        type=str,
    )
    parser.add_argument("--N", default=301, type=int)
    parser.add_argument("--ymax", default=80.0, type=float)
    parser.add_argument("--ci-weight", default=2.0, type=float)
    parser.add_argument("--max-points", default=None, type=int)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected = load_selected_atlas_points(Path(args.manifest))

    if args.max_points is not None:
        selected = selected.head(args.max_points).copy()

    print(f"Selected atlas diagnostic points: {len(selected)}")
    print(selected[["Mach", "eta", "alpha", "chart_id", "chart_status", "ci_ref", "ci_pred"]].to_string(index=False))

    # Imports kept here to give clearer error messages if module paths changed.
    try:
        from src.scripts.gep.selection.solve_dense_gep_notebook_style import NotebookStyleDenseGEPSolver
    except Exception as e:
        raise RuntimeError(
            "Could not import NotebookStyleDenseGEPSolver. "
            "Check code/src/scripts/gep/selection/solve_dense_gep_notebook_style.py"
        ) from e

    try:
        from src.scripts.evaluation.evaluate_kh_subsonic_fixed_mach_modal_candidates import load_classic_full_mode
    except Exception as e:
        raise RuntimeError(
            "Could not import load_classic_full_mode from "
            "code/src/scripts/evaluation/evaluate_kh_subsonic_fixed_mach_modal_candidates.py"
        ) from e

    rows = []

    for i, r in selected.iterrows():
        M = float(r["Mach"])
        eta = float(r["eta"])
        alpha = float(r["alpha"]) if "alpha" in r and np.isfinite(r["alpha"]) else alpha_from_eta(M, eta)

        ci_seed = float(r["ci_pred"])
        ci_ref = float(r["ci_ref"]) if "ci_ref" in r and np.isfinite(r["ci_ref"]) else np.nan

        print(f"\n[{i+1}/{len(selected)}] M={M:.6f} eta={eta:.6f} alpha={alpha:.6f} chart={r['chart_id']} ci_seed={ci_seed:.8e}")

        try:
            classic = load_classic_full_mode(Mach=M, alpha=alpha, n_y=5001, ymax=max(args.ymax, 80.0))
        except TypeError:
            # Older helper variants may use positional args.
            classic = load_classic_full_mode(M, alpha)

        y_ref, p_ref, rho_ref, u_ref, v_ref, ci_classic = unpack_classic_full_mode(classic, M)

        if np.isfinite(ci_ref) is False and ci_classic is not None:
            ci_ref = float(ci_classic)

        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=M,
            N=args.N,
            ymax=args.ymax,
        )

        mode = solver.get_nearest_mode_to_target(
            target_guess=(0.0, ci_seed),
            prefer_positive_cr=False,
            ci_weight=args.ci_weight,
        )

        c_gep = complex(mode["c"])
        ci_gep = float(np.imag(c_gep))
        cr_gep = float(np.real(c_gep))

        vec = np.asarray(mode["vector"])
        n = vec.size // 3
        u_gep = vec[:n]
        v_gep = vec[n:2*n]
        p_gep = vec[2*n:3*n]
        rho_gep = (M * M) * p_gep

        # Solver grid naming may differ.
        if hasattr(solver, "y"):
            y_gep = np.asarray(solver.y)
        elif hasattr(solver, "y_grid"):
            y_gep = np.asarray(solver.y_grid)
        else:
            y_gep = np.linspace(-args.ymax, args.ymax, n)

        # Interpolate reference onto GEP grid if needed.
        def interp_complex(y_src, f_src, y_dst):
            return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))

        p_ref_i = interp_complex(y_ref, p_ref, y_gep)
        u_ref_i = interp_complex(y_ref, u_ref, y_gep)
        v_ref_i = interp_complex(y_ref, v_ref, y_gep)
        rho_ref_i = interp_complex(y_ref, rho_ref, y_gep)

        p_gep_a = align(p_gep, p_ref_i)
        scale = best_complex_scale(p_gep, p_ref_i)
        u_gep_a = scale * u_gep
        v_gep_a = scale * v_gep
        rho_gep_a = scale * rho_gep

        p_rel = rel_l2(p_gep_a, p_ref_i)
        u_rel = rel_l2(u_gep_a, u_ref_i)
        v_rel = rel_l2(v_gep_a, v_ref_i)
        rho_rel = rel_l2(rho_gep_a, rho_ref_i)

        ci_abs_err = abs(ci_gep - ci_ref)
        ci_rel_err = ci_abs_err / max(abs(ci_ref), 1e-14)

        rows.append({
            "Mach": M,
            "eta": eta,
            "alpha": alpha,
            "chart_id": r["chart_id"],
            "chart_status": r["chart_status"],
            "ci_ref": ci_ref,
            "ci_seed": ci_seed,
            "ci_gep": ci_gep,
            "cr_gep": cr_gep,
            "ci_seed_rel_err": abs(ci_seed - ci_ref) / max(abs(ci_ref), 1e-14),
            "ci_gep_abs_err": ci_abs_err,
            "ci_gep_rel_err": ci_rel_err,
            "p_rel_gep": p_rel,
            "rho_rel_gep": rho_rel,
            "u_rel_gep": u_rel,
            "v_rel_gep": v_rel,
        })

        print(
            f"  gep ci={ci_gep:.8e} rel={ci_rel_err:.3e} "
            f"p={p_rel:.3e} u={u_rel:.3e} v={v_rel:.3e}"
        )

    res = pd.DataFrame(rows)
    csv_path = outdir / "atlas_core_ci_seeded_gep_summary.csv"
    res.to_csv(csv_path, index=False)

    print("\n===== summary by chart =====")
    print(
        res.groupby("chart_id")[[
            "ci_seed_rel_err",
            "ci_gep_rel_err",
            "p_rel_gep",
            "rho_rel_gep",
            "u_rel_gep",
            "v_rel_gep",
        ]].mean().to_string()
    )

    print("\n===== global means =====")
    print(
        res[[
            "ci_seed_rel_err",
            "ci_gep_rel_err",
            "p_rel_gep",
            "rho_rel_gep",
            "u_rel_gep",
            "v_rel_gep",
        ]].mean().to_string()
    )

    print("\n===== worst GEP p/u/v =====")
    print(
        res.sort_values("p_rel_gep", ascending=False)
        .head(12)[[
            "Mach", "eta", "alpha", "chart_id",
            "ci_seed_rel_err", "ci_gep_rel_err",
            "p_rel_gep", "u_rel_gep", "v_rel_gep",
        ]].to_string(index=False)
    )

    print("\nCSV:", csv_path)


if __name__ == "__main__":
    main()
