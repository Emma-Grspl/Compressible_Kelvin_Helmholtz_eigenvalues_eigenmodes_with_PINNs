from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from src.scripts.gep.selection.solve_dense_gep_notebook_style import NotebookStyleDenseGEPSolver
from src.scripts.evaluation.evaluate_kh_subsonic_fixed_mach_modal_candidates import load_classic_full_mode


def split_gep_vector(vector: np.ndarray, n_points: int, mach: float):
    u = np.asarray(vector[0:n_points], dtype=np.complex128)
    v = np.asarray(vector[n_points:2*n_points], dtype=np.complex128)
    p = np.asarray(vector[2*n_points:3*n_points], dtype=np.complex128)
    rho = p * mach**2
    return {"u": u, "v": v, "p": p, "rho": rho}


def interp_complex(y_src: np.ndarray, f_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    real = np.interp(y_dst, y_src, np.real(f_src))
    imag = np.interp(y_dst, y_src, np.imag(f_src))
    return real + 1j * imag


def align_complex(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> complex:
    num = np.vdot(pred[mask], ref[mask])
    den = np.vdot(pred[mask], pred[mask])
    if abs(den) < 1e-30:
        return 1.0 + 0.0j
    return num / den


def rel_l2(pred: np.ndarray, ref: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    diff2 = np.abs(pred[mask] - ref[mask]) ** 2
    ref2 = np.abs(ref[mask]) ** 2
    num = np.trapz(diff2, y[mask])
    den = np.trapz(ref2, y[mask])
    return float(np.sqrt(num / max(den, 1e-30)))


def overlap_complex(pred: np.ndarray, ref: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    inner = np.trapz(np.conj(pred[mask]) * ref[mask], y[mask])
    npred = np.sqrt(np.trapz(np.abs(pred[mask]) ** 2, y[mask]))
    nref = np.sqrt(np.trapz(np.abs(ref[mask]) ** 2, y[mask]))
    return float(abs(inner) / max(float(npred * nref), 1e-30))


def load_atlas_points(manifest_path: Path, include_ci_only: bool) -> pd.DataFrame:
    man = pd.read_csv(manifest_path)

    rows = []
    for _, chart in man.iterrows():
        if (not include_ci_only) and str(chart["status"]) == "ci_only":
            continue

        diag_path = Path(chart["path"]) / "diagnostics_summary.csv"
        if not diag_path.exists():
            print(f"[WARN] missing diagnostics: {diag_path}")
            continue

        df = pd.read_csv(diag_path)
        df["chart_id"] = chart["chart_id"]
        df["chart_status"] = chart["status"]
        df["chart_priority"] = int(chart["priority"])
        rows.append(df)

    if not rows:
        raise SystemExit("No diagnostics_summary.csv found from manifest.")

    data = pd.concat(rows, ignore_index=True)

    data["Mach_key"] = data["Mach"].round(8)
    data["eta_key"] = data["eta"].round(8)

    data = data.sort_values(
        ["Mach_key", "eta_key", "chart_priority"],
        ascending=[True, True, False],
    )

    selected = data.drop_duplicates(["Mach_key", "eta_key"], keep="first")
    selected = selected.sort_values(["Mach", "eta"]).reset_index(drop=True)

    return selected


def run_one_seed(
    *,
    alpha: float,
    mach: float,
    eta: float,
    ci_seed: float,
    chart_id: str,
    chart_status: str,
    n_points: int,
    mapping_kind: str,
    mapping_scale: float,
    xi_max: float,
):
    classic_fields, ci_classic = load_classic_full_mode(alpha, mach)

    y_ref = np.asarray(classic_fields["y"], dtype=float)
    p_ref = np.asarray(classic_fields["p"], dtype=np.complex128)
    rho_ref = np.asarray(classic_fields["rho"], dtype=np.complex128)
    u_ref = np.asarray(classic_fields["u"], dtype=np.complex128)
    v_ref = np.asarray(classic_fields["v"], dtype=np.complex128)

    solver = NotebookStyleDenseGEPSolver(
        alpha=alpha,
        Mach=mach,
        n_points=n_points,
        mapping_kind=mapping_kind,
        mapping_scale=mapping_scale,
        xi_max=xi_max,
    )

    mode, selection_source, n_modes = solver.get_nearest_mode_to_target(
        target_guess=(0.0, float(ci_seed)),
        prefer_positive_cr=False,
        ci_weight=2.0,
    )

    base = {
        "alpha": alpha,
        "eta": eta,
        "Mach": mach,
        "chart_id": chart_id,
        "chart_status": chart_status,
        "N": n_points,
        "ci_classic": float(ci_classic),
        "ci_seed": float(ci_seed),
        "ci_seed_abs_err": abs(float(ci_seed) - float(ci_classic)),
        "ci_seed_rel_err": abs(float(ci_seed) - float(ci_classic)) / max(abs(float(ci_classic)), 1e-12),
        "selection_source": selection_source,
        "n_finite_modes": n_modes,
    }

    if mode is None:
        base.update({
            "gep_cr": np.nan,
            "gep_ci": np.nan,
            "ci_gep_abs_err": np.nan,
            "ci_gep_rel_err": np.nan,
            "p_rel": np.nan,
            "rho_rel": np.nan,
            "u_rel": np.nan,
            "v_rel": np.nan,
            "p_overlap": np.nan,
            "success": False,
        })
        return base

    gep = split_gep_vector(mode["vector"], solver.n_points, mach)

    p_gep = interp_complex(solver.y, gep["p"], y_ref)
    rho_gep = interp_complex(solver.y, gep["rho"], y_ref)
    u_gep = interp_complex(solver.y, gep["u"], y_ref)
    v_gep = interp_complex(solver.y, gep["v"], y_ref)

    y_min = max(float(np.min(y_ref)), float(np.min(solver.y)), -12.0)
    y_max = min(float(np.max(y_ref)), float(np.max(solver.y)), 12.0)
    mask = (y_ref >= y_min) & (y_ref <= y_max)

    scale = align_complex(p_gep, p_ref, mask)
    p_gep *= scale
    rho_gep *= scale
    u_gep *= scale
    v_gep *= scale

    base.update({
        "gep_cr": float(mode["cr"]),
        "gep_ci": float(mode["ci"]),
        "ci_gep_abs_err": abs(float(mode["ci"]) - float(ci_classic)),
        "ci_gep_rel_err": abs(float(mode["ci"]) - float(ci_classic)) / max(abs(float(ci_classic)), 1e-12),
        "p_rel": rel_l2(p_gep, p_ref, y_ref, mask),
        "rho_rel": rel_l2(rho_gep, rho_ref, y_ref, mask),
        "u_rel": rel_l2(u_gep, u_ref, y_ref, mask),
        "v_rel": rel_l2(v_gep, v_ref, y_ref, mask),
        "p_overlap": overlap_complex(p_gep, p_ref, y_ref, mask),
        "success": True,
    })

    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/pinn_subsonic/local_atlas_v1/atlas_manifest.csv")
    ap.add_argument("--output-dir", default="assets/pinn_subsonic/local_atlas_v1/gep_core_atlas_ci_seeded_v2")
    ap.add_argument("--N", type=int, default=301)
    ap.add_argument("--max-points", type=int, default=0)
    ap.add_argument("--include-ci-only", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    points = load_atlas_points(Path(args.manifest), include_ci_only=args.include_ci_only)

    if args.max_points and args.max_points > 0:
        points = points.head(args.max_points).copy()

    print(f"Selected atlas diagnostic points: {len(points)}")
    print(points[["Mach", "eta", "alpha", "chart_id", "chart_status", "ci_ref", "ci_pred"]].to_string(index=False))

    rows = []

    for k, r in points.iterrows():
        alpha = float(r["alpha"])
        mach = float(r["Mach"])
        eta = float(r["eta"])
        ci_seed = float(r["ci_pred"])
        chart_id = str(r["chart_id"])
        chart_status = str(r["chart_status"])

        print(
            f"\n[RUN {k+1}/{len(points)}] "
            f"M={mach:.4f} eta={eta:.4f} alpha={alpha:.6f} "
            f"chart={chart_id} ci_seed={ci_seed:.8e}"
        )

        try:
            row = run_one_seed(
                alpha=alpha,
                mach=mach,
                eta=eta,
                ci_seed=ci_seed,
                chart_id=chart_id,
                chart_status=chart_status,
                n_points=args.N,
                mapping_kind="pin",
                mapping_scale=15.0,
                xi_max=0.99,
            )
        except Exception as exc:
            row = {
                "alpha": alpha,
                "eta": eta,
                "Mach": mach,
                "chart_id": chart_id,
                "chart_status": chart_status,
                "N": args.N,
                "ci_classic": np.nan,
                "ci_seed": ci_seed,
                "ci_seed_abs_err": np.nan,
                "ci_seed_rel_err": np.nan,
                "gep_cr": np.nan,
                "gep_ci": np.nan,
                "ci_gep_abs_err": np.nan,
                "ci_gep_rel_err": np.nan,
                "p_rel": np.nan,
                "rho_rel": np.nan,
                "u_rel": np.nan,
                "v_rel": np.nan,
                "p_overlap": np.nan,
                "selection_source": f"ERROR: {type(exc).__name__}: {exc}",
                "n_finite_modes": 0,
                "success": False,
            }

        rows.append(row)

        print(
            f"  success={row.get('success')} "
            f"ci_seed_rel={row.get('ci_seed_rel_err')} "
            f"ci_gep_rel={row.get('ci_gep_rel_err')} "
            f"p_rel={row.get('p_rel')} "
            f"u_rel={row.get('u_rel')} "
            f"v_rel={row.get('v_rel')} "
            f"overlap={row.get('p_overlap')} "
            f"source={row.get('selection_source')}"
        )

    summary = pd.DataFrame(rows).sort_values(["Mach", "eta"]).reset_index(drop=True)

    path = outdir / "summary_atlas_core_ci_seeded_gep_modes.csv"
    summary.to_csv(path, index=False)

    print("\n===== global means =====")
    cols = ["ci_seed_rel_err", "ci_gep_rel_err", "p_rel", "rho_rel", "u_rel", "v_rel", "p_overlap"]
    print(summary[cols].mean(numeric_only=True).to_string())

    print("\n===== by chart =====")
    print(summary.groupby("chart_id")[cols].mean(numeric_only=True).to_string())

    print("\n===== worst p_rel =====")
    print(
        summary.sort_values("p_rel", ascending=False)[[
            "Mach", "eta", "alpha", "chart_id",
            "ci_seed_rel_err", "ci_gep_rel_err",
            "p_rel", "rho_rel", "u_rel", "v_rel", "p_overlap",
            "success", "selection_source",
        ]].head(20).to_string(index=False)
    )

    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()
