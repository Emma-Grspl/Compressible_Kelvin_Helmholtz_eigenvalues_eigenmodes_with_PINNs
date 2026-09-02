from __future__ import annotations

from pathlib import Path
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


def interp_complex(y_src, f_src, y_dst):
    return np.interp(y_dst, y_src, np.real(f_src)) + 1j * np.interp(y_dst, y_src, np.imag(f_src))


def align_complex(pred, ref, mask):
    den = np.vdot(pred[mask], pred[mask])
    if abs(den) < 1e-30:
        return 1.0 + 0.0j
    return np.vdot(pred[mask], ref[mask]) / den


def rel_l2(pred, ref, y, mask):
    num = np.trapz(np.abs(pred[mask] - ref[mask]) ** 2, y[mask])
    den = np.trapz(np.abs(ref[mask]) ** 2, y[mask])
    return float(np.sqrt(num / max(den, 1e-30)))


def overlap_complex(pred, ref, y, mask):
    inner = np.trapz(np.conj(pred[mask]) * ref[mask], y[mask])
    npred = np.sqrt(np.trapz(np.abs(pred[mask]) ** 2, y[mask]))
    nref = np.sqrt(np.trapz(np.abs(ref[mask]) ** 2, y[mask]))
    return float(abs(inner) / max(float(npred * nref), 1e-30))


def run_one(fields_csv: Path, n_points: int = 301):
    df = pd.read_csv(fields_csv)

    alpha = float(df["alpha"].iloc[0])
    mach = float(df["Mach"].iloc[0])
    ci_pred = float(df["ci_pred"].iloc[0])

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
        mapping_kind="pin",
        mapping_scale=5.0,
        xi_max=0.98,
    )

    mode, selection_source, n_modes = solver.get_nearest_mode_to_target(
        target_guess=(0.0, ci_pred),
        prefer_positive_cr=False,
        ci_weight=2.0,
    )

    if mode is None:
        return {
            "alpha": alpha,
            "Mach": mach,
            "N": n_points,
            "ci_classic": ci_classic,
            "ci_pinn": ci_pred,
            "gep_cr": np.nan,
            "gep_ci": np.nan,
            "ci_pinn_abs_err": abs(ci_pred - ci_classic),
            "ci_pinn_rel_err": abs(ci_pred - ci_classic) / max(abs(ci_classic), 1e-12),
            "ci_gep_abs_err": np.nan,
            "ci_gep_rel_err": np.nan,
            "p_rel": np.nan,
            "rho_rel": np.nan,
            "u_rel": np.nan,
            "v_rel": np.nan,
            "p_overlap": np.nan,
            "selection_source": selection_source,
            "n_finite_modes": n_modes,
            "success": False,
        }

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

    return {
        "alpha": alpha,
        "Mach": mach,
        "N": n_points,
        "ci_classic": float(ci_classic),
        "ci_pinn": float(ci_pred),
        "gep_cr": float(mode["cr"]),
        "gep_ci": float(mode["ci"]),
        "ci_pinn_abs_err": abs(ci_pred - ci_classic),
        "ci_pinn_rel_err": abs(ci_pred - ci_classic) / max(abs(ci_classic), 1e-12),
        "ci_gep_abs_err": abs(float(mode["ci"]) - ci_classic),
        "ci_gep_rel_err": abs(float(mode["ci"]) - ci_classic) / max(abs(ci_classic), 1e-12),
        "p_rel": rel_l2(p_gep, p_ref, y_ref, mask),
        "rho_rel": rel_l2(rho_gep, rho_ref, y_ref, mask),
        "u_rel": rel_l2(u_gep, u_ref, y_ref, mask),
        "v_rel": rel_l2(v_gep, v_ref, y_ref, mask),
        "p_overlap": overlap_complex(p_gep, p_ref, y_ref, mask),
        "selection_source": selection_source,
        "n_finite_modes": n_modes,
        "success": True,
    }


def main():
    run_dir = ROOT / "assets/pinn_subsonic/seedGEP_pq2d_M050_a030_a070_ci4_contalpha"
    outdir = ROOT / "assets/gep_subsonic/pinn_ci_seeded_modes"
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(run_dir.glob("fields_vs_classic_*.csv"))
    if not files:
        raise SystemExit(f"No fields_vs_classic_*.csv found in {run_dir}")

    rows = []
    for f in files:
        print(f"[RUN] {f.name}")
        row = run_one(f, n_points=301)
        rows.append(row)
        print(
            f"  alpha={row['alpha']} "
            f"ci_pinn_rel={row['ci_pinn_rel_err']:.3e} "
            f"ci_gep_rel={row['ci_gep_rel_err']:.3e} "
            f"p_rel={row['p_rel']:.3e} "
            f"u_rel={row['u_rel']:.3e} "
            f"v_rel={row['v_rel']:.3e} "
            f"overlap={row['p_overlap']:.9f} "
            f"gep_ci={row['gep_ci']:.6f}"
        )

    summary = pd.DataFrame(rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)
    path = outdir / "summary_pinn_ci_seeded_gep_modes.csv"
    summary.to_csv(path, index=False)

    print("\n===== summary =====")
    cols = [
        "alpha", "Mach", "N",
        "ci_classic", "ci_pinn", "gep_ci",
        "ci_pinn_rel_err", "ci_gep_rel_err",
        "p_rel", "rho_rel", "u_rel", "v_rel", "p_overlap",
        "selection_source", "n_finite_modes", "success",
    ]
    print(summary[cols].to_string(index=False))
    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()
