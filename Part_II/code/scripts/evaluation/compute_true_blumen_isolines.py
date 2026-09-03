#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq


# ============================================================
# Helpers
# ============================================================

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def parse_ci_from_string(text: str) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None

    # cas usuels: "ci=0.02", "c_i = 0.02", "level=0.02", "ci_sup=0"
    patterns = [
        r'(?:^|[^a-z])ci\s*=?\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
        r'(?:^|[^a-z])c_i\s*=?\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
        r'(?:^|[^a-z])ci_sup\s*=?\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
        r'(?:^|[^a-z])level\s*=?\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
        r'(?:^|[^a-z])value\s*=?\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return float(m.group(1))

    return None


def load_blumen_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    mach_col = find_column(df, ["Mach", "M", "mach"])
    alpha_col = find_column(df, ["alpha", "Alpha"])
    ci_col = find_column(
        df,
        [
            "ci",
            "c_i",
            "ci_level",
            "ci_value",
            "ci_digitized",
            "contour_ci",
            "contour_level",
            "level",
            "value",
        ],
    )
    curve_id_col = find_column(df, ["curve_id", "curve", "line_id"])
    curve_label_col = find_column(df, ["curve_label", "label", "name", "source"])

    if mach_col is None or alpha_col is None:
        raise ValueError(
            f"Impossible de trouver les colonnes Mach/alpha dans {path} ; "
            f"colonnes observées: {list(df.columns)}"
        )

    out = df.copy()
    out["Mach"] = pd.to_numeric(out[mach_col], errors="coerce")
    out["alpha"] = pd.to_numeric(out[alpha_col], errors="coerce")

    if ci_col is not None:
        out["target_ci"] = pd.to_numeric(out[ci_col], errors="coerce")
    else:
        out["target_ci"] = np.nan

    if curve_id_col is not None:
        out["curve_id_key"] = out[curve_id_col].astype(str)
    else:
        out["curve_id_key"] = np.nan

    if curve_label_col is not None:
        out["curve_label"] = out[curve_label_col].astype(str)
    else:
        out["curve_label"] = ""

    # Si target_ci manque, on tente de le parser depuis le label
    mask_missing_ci = ~np.isfinite(out["target_ci"].to_numpy(float))
    if mask_missing_ci.any():
        parsed = []
        for _, row in out.loc[mask_missing_ci].iterrows():
            value = parse_ci_from_string(row.get("curve_label", ""))
            if value is None:
                value = parse_ci_from_string(row.get("curve_id_key", ""))
            parsed.append(value)
        out.loc[mask_missing_ci, "target_ci"] = parsed

    out = out.dropna(subset=["Mach", "alpha", "target_ci"]).copy()

    # clé de courbe
    def make_curve_key(row):
        cid = str(row.get("curve_id_key", ""))
        clb = str(row.get("curve_label", ""))
        tci = float(row["target_ci"])
        return f"{cid}__{clb}__ci_{tci:.12g}"

    out["curve_key"] = out.apply(make_curve_key, axis=1)

    # ordonner proprement
    out = out.sort_values(["target_ci", "Mach", "alpha"]).reset_index(drop=True)
    return out


def load_reference(path: Path) -> pd.DataFrame:
    ref = pd.read_csv(path)
    for col in ["Mach", "alpha", "cr", "ci"]:
        if col not in ref.columns:
            raise ValueError(f"Colonne manquante dans {path}: {col}")
        ref[col] = pd.to_numeric(ref[col], errors="coerce")
    ref = ref.dropna(subset=["Mach", "alpha", "cr", "ci"]).copy()
    return ref


@dataclass
class SolveResult:
    ok: bool
    Mach: float
    alpha: float
    cr: Optional[float]
    ci: Optional[float]
    residual_norm: Optional[float]
    delta_ci_to_target: Optional[float]
    accepted: bool
    workdir: Optional[Path]
    message: str


def default_Ly(Mach: float) -> float:
    return 2000.0 if Mach >= 1.8 else 500.0


def nearest_seed(
    ref: pd.DataFrame,
    Mach: float,
    alpha: float,
    previous: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    if previous is not None:
        return previous

    rr = ref.copy()
    rr["dist2"] = (rr["Mach"] - Mach) ** 2 + (rr["alpha"] - alpha) ** 2
    row = rr.sort_values("dist2").iloc[0]
    return float(row["cr"]), float(row["ci"])


def run_fixed_alpha_solver(
    repo: Path,
    search_script: Path,
    work_root: Path,
    Mach: float,
    alpha: float,
    target_ci: float,
    seed_cr: float,
    seed_ci: float,
    Ly: float,
    matching_y: float,
    max_step: float,
    rtol: float,
    atol: float,
    method: str,
    optimizer_xtol: float,
    optimizer_ftol: float,
    python_exe: str,
    reference_parquet: Optional[Path] = None,
) -> SolveResult:
    workdir = work_root / f"M{Mach:.6f}_a{alpha:.8f}_{uuid.uuid4().hex[:8]}"
    workdir.mkdir(parents=True, exist_ok=True)

    # bornes assez larges mais centrées autour du seed
    cr_lower = max(-1.0, seed_cr - 0.15)
    cr_upper = seed_cr + 0.15
    ci_lower = max(0.0, min(seed_ci, target_ci) - 0.03)
    ci_upper = max(seed_ci, target_ci, 0.02) + 0.06

    cmd = [
        python_exe,
        "-u",
        str(search_script),
        "--repo", str(repo),
        "--Mach", f"{Mach:.15g}",
        "--alpha", f"{alpha:.15g}",
        "--seed-cr", f"{seed_cr:.17g}",
        "--seed-ci", f"{seed_ci:.17g}",
        "--reference-cr", f"{seed_cr:.17g}",
        "--reference-ci", f"{seed_ci:.17g}",
        "--cr-lower", f"{cr_lower:.17g}",
        "--cr-upper", f"{cr_upper:.17g}",
        "--ci-lower", f"{ci_lower:.17g}",
        "--ci-upper", f"{ci_upper:.17g}",
        "--Ly", f"{Ly:.15g}",
        "--matching-y", f"{matching_y:.15g}",
        "--max-step", f"{max_step:.15g}",
        "--rtol", f"{rtol:.15g}",
        "--atol", f"{atol:.15g}",
        "--method", method,
        "--optimizer-xtol", f"{optimizer_xtol:.15g}",
        "--optimizer-ftol", f"{optimizer_ftol:.15g}",
        "--output-dir", str(workdir),
    ]
    if reference_parquet is not None and reference_parquet.is_file():
        cmd += ["--reference-parquet", str(reference_parquet)]

    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        return SolveResult(
            ok=False,
            Mach=Mach,
            alpha=alpha,
            cr=None,
            ci=None,
            residual_norm=None,
            delta_ci_to_target=None,
            accepted=False,
            workdir=workdir,
            message=f"subprocess exception: {exc}",
        )

    summary = workdir / "summary.json"
    if proc.returncode != 0 or not summary.is_file():
        msg = (
            f"solver returncode={proc.returncode}\n"
            f"stdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )
        return SolveResult(
            ok=False,
            Mach=Mach,
            alpha=alpha,
            cr=None,
            ci=None,
            residual_norm=None,
            delta_ci_to_target=None,
            accepted=False,
            workdir=workdir,
            message=msg,
        )

    data = json.loads(summary.read_text())

    opt = data.get("optimized_eigenvalue", {})
    root_test = data.get("root_test", {})
    cr = float(opt["cr"])
    ci = float(opt["ci"])
    residual = float(root_test.get("residual_norm", np.nan))
    accepted = bool(root_test.get("accepted", False))

    return SolveResult(
        ok=True,
        Mach=Mach,
        alpha=alpha,
        cr=cr,
        ci=ci,
        residual_norm=residual,
        delta_ci_to_target=ci - target_ci,
        accepted=accepted,
        workdir=workdir,
        message="OK",
    )


def solve_curve_point(
    repo: Path,
    search_script: Path,
    work_root: Path,
    ref: pd.DataFrame,
    Mach: float,
    alpha_guess: float,
    target_ci: float,
    python_exe: str,
    reference_parquet: Optional[Path],
    previous_seed: Optional[Tuple[float, float]],
    Ly: Optional[float],
    matching_y: float,
    max_step: float,
    rtol: float,
    atol: float,
    method: str,
    optimizer_xtol: float,
    optimizer_ftol: float,
    alpha_scan_halfwidth: float,
    alpha_scan_step: float,
    root_tol: float,
) -> Dict[str, object]:
    Ly_eff = default_Ly(Mach) if Ly is None else Ly

    seed_cr, seed_ci = nearest_seed(ref, Mach, alpha_guess, previous_seed)

    cache: Dict[float, SolveResult] = {}

    def evaluate(alpha: float) -> SolveResult:
        key = float(np.round(alpha, 12))
        if key not in cache:
            cache[key] = run_fixed_alpha_solver(
                repo=repo,
                search_script=search_script,
                work_root=work_root,
                Mach=Mach,
                alpha=alpha,
                target_ci=target_ci,
                seed_cr=seed_cr,
                seed_ci=seed_ci,
                Ly=Ly_eff,
                matching_y=matching_y,
                max_step=max_step,
                rtol=rtol,
                atol=atol,
                method=method,
                optimizer_xtol=optimizer_xtol,
                optimizer_ftol=optimizer_ftol,
                python_exe=python_exe,
                reference_parquet=reference_parquet,
            )
        return cache[key]

    center = evaluate(alpha_guess)
    ci_at_blumen_alpha = center.ci if center.ok else np.nan
    delta_ci_at_blumen_alpha = (
        center.delta_ci_to_target if center.ok else np.nan
    )

    # si déjà bon
    if center.ok and center.accepted and center.delta_ci_to_target is not None:
        if abs(center.delta_ci_to_target) <= root_tol:
            return {
                "status": "converged_center",
                "Mach": Mach,
                "target_ci": target_ci,
                "alpha_blumen": alpha_guess,
                "alpha_classical": center.alpha,
                "classical_cr": center.cr,
                "classical_ci": center.ci,
                "residual_norm": center.residual_norm,
                "accepted": center.accepted,
                "ci_at_blumen_alpha": ci_at_blumen_alpha,
                "delta_ci_at_blumen_alpha": delta_ci_at_blumen_alpha,
                "delta_alpha": center.alpha - alpha_guess,
                "solver_workdir": str(center.workdir),
            }

    # chercher un bracket
    scan_offsets = [0.0]
    n_expand = int(math.ceil(alpha_scan_halfwidth / alpha_scan_step))
    for k in range(1, n_expand + 1):
        scan_offsets.extend([-k * alpha_scan_step, k * alpha_scan_step])

    sampled = []
    for off in scan_offsets:
        a = alpha_guess + off
        if a <= 0:
            continue
        rr = evaluate(a)
        if rr.ok and rr.accepted and rr.delta_ci_to_target is not None:
            sampled.append((a, rr.delta_ci_to_target))

    sampled = sorted(sampled, key=lambda t: t[0])

    bracket = None
    for (a0, f0), (a1, f1) in zip(sampled[:-1], sampled[1:]):
        if np.sign(f0) == 0:
            bracket = (a0, a0)
            break
        if np.sign(f1) == 0:
            bracket = (a1, a1)
            break
        if np.sign(f0) != np.sign(f1):
            bracket = (a0, a1)
            break

    if bracket is None:
        return {
            "status": "no_bracket",
            "Mach": Mach,
            "target_ci": target_ci,
            "alpha_blumen": alpha_guess,
            "alpha_classical": np.nan,
            "classical_cr": np.nan,
            "classical_ci": np.nan,
            "residual_norm": np.nan,
            "accepted": False,
            "ci_at_blumen_alpha": ci_at_blumen_alpha,
            "delta_ci_at_blumen_alpha": delta_ci_at_blumen_alpha,
            "delta_alpha": np.nan,
            "solver_workdir": str(center.workdir) if center.workdir is not None else "",
        }

    a_lo, a_hi = bracket

    if a_lo == a_hi:
        rr = evaluate(a_lo)
        return {
            "status": "converged_bracket_endpoint",
            "Mach": Mach,
            "target_ci": target_ci,
            "alpha_blumen": alpha_guess,
            "alpha_classical": rr.alpha,
            "classical_cr": rr.cr,
            "classical_ci": rr.ci,
            "residual_norm": rr.residual_norm,
            "accepted": rr.accepted,
            "ci_at_blumen_alpha": ci_at_blumen_alpha,
            "delta_ci_at_blumen_alpha": delta_ci_at_blumen_alpha,
            "delta_alpha": rr.alpha - alpha_guess,
            "solver_workdir": str(rr.workdir),
        }

    def f_root(alpha: float) -> float:
        rr = evaluate(alpha)
        if (not rr.ok) or (not rr.accepted) or (rr.delta_ci_to_target is None):
            raise RuntimeError(
                f"Échec solveur à alpha={alpha:.12g}, Mach={Mach:.12g}"
            )
        return float(rr.delta_ci_to_target)

    try:
        alpha_star = brentq(f_root, a_lo, a_hi, xtol=1e-6, rtol=1e-8, maxiter=40)
    except Exception:
        return {
            "status": "root_fail",
            "Mach": Mach,
            "target_ci": target_ci,
            "alpha_blumen": alpha_guess,
            "alpha_classical": np.nan,
            "classical_cr": np.nan,
            "classical_ci": np.nan,
            "residual_norm": np.nan,
            "accepted": False,
            "ci_at_blumen_alpha": ci_at_blumen_alpha,
            "delta_ci_at_blumen_alpha": delta_ci_at_blumen_alpha,
            "delta_alpha": np.nan,
            "solver_workdir": "",
        }

    rr = evaluate(alpha_star)
    return {
        "status": "converged_root",
        "Mach": Mach,
        "target_ci": target_ci,
        "alpha_blumen": alpha_guess,
        "alpha_classical": rr.alpha,
        "classical_cr": rr.cr,
        "classical_ci": rr.ci,
        "residual_norm": rr.residual_norm,
        "accepted": rr.accepted,
        "ci_at_blumen_alpha": ci_at_blumen_alpha,
        "delta_ci_at_blumen_alpha": delta_ci_at_blumen_alpha,
        "delta_alpha": rr.alpha - alpha_guess,
        "solver_workdir": str(rr.workdir),
    }


def build_targets(blumen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for curve_key, sub in blumen.groupby("curve_key", sort=False):
        sub = sub.sort_values(["Mach", "alpha"]).copy()
        target_ci = float(sub["target_ci"].median())
        curve_label = str(sub["curve_label"].iloc[0]) if "curve_label" in sub.columns else ""
        curve_id = str(sub["curve_id_key"].iloc[0]) if "curve_id_key" in sub.columns else ""

        # un point alpha_blumen par Mach : moyenne si plusieurs points au même Mach
        gg = (
            sub.groupby("Mach", as_index=False)
            .agg(
                alpha_blumen=("alpha", "mean"),
                n_raw_points=("alpha", "size"),
            )
            .sort_values("Mach")
        )

        for _, row in gg.iterrows():
            rows.append(
                {
                    "curve_key": curve_key,
                    "curve_id_key": curve_id,
                    "curve_label": curve_label,
                    "target_ci": target_ci,
                    "Mach": float(row["Mach"]),
                    "alpha_blumen": float(row["alpha_blumen"]),
                    "n_raw_points": int(row["n_raw_points"]),
                }
            )

    out = pd.DataFrame(rows).sort_values(["target_ci", "Mach"]).reset_index(drop=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--blumen-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--search-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-parquet", type=Path, default=None)
    parser.add_argument("--python-exe", type=str, default=sys.executable)

    parser.add_argument("--matching-y", type=float, default=1.0)
    parser.add_argument("--Ly", type=float, default=None)
    parser.add_argument("--max-step", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--method", type=str, default="DOP853")
    parser.add_argument("--optimizer-xtol", type=float, default=1e-11)
    parser.add_argument("--optimizer-ftol", type=float, default=1e-11)

    parser.add_argument("--alpha-scan-halfwidth", type=float, default=0.03)
    parser.add_argument("--alpha-scan-step", type=float, default=0.0025)
    parser.add_argument("--root-tol", type=float, default=5e-4)

    parser.add_argument("--include-neutral", action="store_true")
    parser.add_argument("--max-curves", type=int, default=None)

    args = parser.parse_args()

    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = output_dir / "_solver_runs"
    work_root.mkdir(parents=True, exist_ok=True)

    blumen = load_blumen_points(args.blumen_csv)
    ref = load_reference(args.reference_csv)

    if not args.include_neutral:
        blumen = blumen.loc[blumen["target_ci"] > 0].copy()

    targets = build_targets(blumen)
    if args.max_curves is not None:
        keep = targets["curve_key"].drop_duplicates().iloc[: args.max_curves]
        targets = targets.loc[targets["curve_key"].isin(set(keep))].copy()

    results = []
    summary_rows = []

    print("=== TRUE CLASSICAL ISOLINES AT BLUMEN LEVELS ===", flush=True)
    print(f"Repository       : {repo}", flush=True)
    print(f"Blumen CSV       : {args.blumen_csv}", flush=True)
    print(f"Reference CSV    : {args.reference_csv}", flush=True)
    print(f"Search script    : {args.search_script}", flush=True)
    print(f"Output           : {output_dir}", flush=True)
    print(f"Targets          : {len(targets)}", flush=True)
    print(
        f"Curves           : {targets['curve_key'].nunique()}",
        flush=True,
    )

    for i_curve, (curve_key, sub) in enumerate(
        targets.groupby("curve_key", sort=False), start=1
    ):
        sub = sub.sort_values("Mach").copy()
        target_ci = float(sub["target_ci"].iloc[0])
        curve_label = str(sub["curve_label"].iloc[0])
        print(
            f"\n[{i_curve}/{targets['curve_key'].nunique()}] "
            f"{curve_key}  target_ci={target_ci:.12g}  n_Mach={len(sub)}",
            flush=True,
        )

        previous_seed = None
        n_ok = 0

        for _, row in sub.iterrows():
            Mach = float(row["Mach"])
            alpha_blumen = float(row["alpha_blumen"])

            alpha_guess = alpha_blumen
            if previous_seed is not None and np.isfinite(alpha_blumen):
                # on reste proche de Blumen, mais on profite un peu de la continuation
                alpha_guess = 0.5 * alpha_blumen + 0.5 * alpha_guess

            rr = solve_curve_point(
                repo=repo,
                search_script=args.search_script.resolve(),
                work_root=work_root,
                ref=ref,
                Mach=Mach,
                alpha_guess=alpha_guess,
                target_ci=target_ci,
                python_exe=args.python_exe,
                reference_parquet=args.reference_parquet.resolve() if args.reference_parquet else None,
                previous_seed=previous_seed,
                Ly=args.Ly,
                matching_y=args.matching_y,
                max_step=args.max_step,
                rtol=args.rtol,
                atol=args.atol,
                method=args.method,
                optimizer_xtol=args.optimizer_xtol,
                optimizer_ftol=args.optimizer_ftol,
                alpha_scan_halfwidth=args.alpha_scan_halfwidth,
                alpha_scan_step=args.alpha_scan_step,
                root_tol=args.root_tol,
            )

            rr.update(
                {
                    "curve_key": curve_key,
                    "curve_label": curve_label,
                    "curve_id_key": row["curve_id_key"],
                    "n_raw_points": row["n_raw_points"],
                }
            )
            results.append(rr)

            if rr["status"].startswith("converged") and np.isfinite(rr["classical_cr"]) and np.isfinite(rr["classical_ci"]):
                previous_seed = (float(rr["classical_cr"]), float(rr["classical_ci"]))
                n_ok += 1

            print(
                f"  M={Mach:.6f}  alpha_B={alpha_blumen:.6f}  "
                f"status={rr['status']}  alpha_C={rr['alpha_classical']}",
                flush=True,
            )

        summary_rows.append(
            {
                "curve_key": curve_key,
                "curve_label": curve_label,
                "target_ci": target_ci,
                "n_targets": len(sub),
                "n_converged": n_ok,
            }
        )

    results_df = pd.DataFrame(results)
    summary_df = pd.DataFrame(summary_rows)

    out_csv = output_dir / "blumen_true_classical_isolines.csv"
    summary_csv = output_dir / "blumen_true_classical_isolines_summary.csv"

    results_df.to_csv(out_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    meta = {
        "status": "PASS",
        "n_rows": int(len(results_df)),
        "n_curves": int(targets["curve_key"].nunique()),
        "n_converged": int(results_df["status"].astype(str).str.startswith("converged").sum()),
        "output_csv": str(out_csv),
        "summary_csv": str(summary_csv),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("\n=== DONE ===", flush=True)
    print(f"Wrote: {out_csv}", flush=True)
    print(f"Wrote: {summary_csv}", flush=True)
    print(f"Wrote: {output_dir / 'metadata.json'}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
