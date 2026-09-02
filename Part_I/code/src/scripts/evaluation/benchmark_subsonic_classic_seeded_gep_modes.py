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
    # L2 inner product with trapezoidal weights approximated by trapz.
    inner = np.trapz(np.conj(pred[mask]) * ref[mask], y[mask])
    npred = np.sqrt(np.trapz(np.abs(pred[mask]) ** 2, y[mask]))
    nref = np.sqrt(np.trapz(np.abs(ref[mask]) ** 2, y[mask]))
    return float(abs(inner) / max(float(npred * nref), 1e-30))


def run_one(alpha: float, mach: float, n_points: int, mapping_kind: str, mapping_scale: float, xi_max: float):
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

    target = (0.0, float(ci_classic))
    mode, selection_source, n_modes = solver.get_nearest_mode_to_target(
        target_guess=target,
        prefer_positive_cr=False,
        ci_weight=2.0,
    )

    if mode is None:
        return {
            "alpha": alpha,
            "Mach": mach,
            "N": n_points,
            "ci_classic": float(ci_classic),
            "gep_cr": np.nan,
            "gep_ci": np.nan,
            "ci_abs_err": np.nan,
            "ci_rel_err": np.nan,
            "p_rel": np.nan,
            "rho_rel": np.nan,
            "u_rel": np.nan,
            "v_rel": np.nan,
            "p_overlap": np.nan,
            "selection_source": selection_source,
            "n_finite_modes": n_modes,
            "success": False,
        }, None

    gep = split_gep_vector(mode["vector"], solver.n_points, mach)

    # Interpolate GEP fields onto the classic reference grid.
    p_gep = interp_complex(solver.y, gep["p"], y_ref)
    rho_gep = interp_complex(solver.y, gep["rho"], y_ref)
    u_gep = interp_complex(solver.y, gep["u"], y_ref)
    v_gep = interp_complex(solver.y, gep["v"], y_ref)

    # Compare only on the reliable common/core region.
    y_min = max(float(np.min(y_ref)), float(np.min(solver.y)), -12.0)
    y_max = min(float(np.max(y_ref)), float(np.max(solver.y)), 12.0)
    mask = (y_ref >= y_min) & (y_ref <= y_max)

    # Align the whole GEP eigenvector using pressure only.
    scale = align_complex(p_gep, p_ref, mask)
    p_gep *= scale
    rho_gep *= scale
    u_gep *= scale
    v_gep *= scale

    row = {
        "alpha": alpha,
        "Mach": mach,
        "N": n_points,
        "ci_classic": float(ci_classic),
        "gep_cr": float(mode["cr"]),
        "gep_ci": float(mode["ci"]),
        "gep_omega_i": float(mode["omega_i"]),
        "ci_abs_err": abs(float(mode["ci"]) - float(ci_classic)),
        "ci_rel_err": abs(float(mode["ci"]) - float(ci_classic)) / max(abs(float(ci_classic)), 1e-12),
        "p_rel": rel_l2(p_gep, p_ref, y_ref, mask),
        "rho_rel": rel_l2(rho_gep, rho_ref, y_ref, mask),
        "u_rel": rel_l2(u_gep, u_ref, y_ref, mask),
        "v_rel": rel_l2(v_gep, v_ref, y_ref, mask),
        "p_overlap": overlap_complex(p_gep, p_ref, y_ref, mask),
        "align_scale_real": float(np.real(scale)),
        "align_scale_imag": float(np.imag(scale)),
        "selection_source": selection_source,
        "n_finite_modes": n_modes,
        "y_min_compare": y_min,
        "y_max_compare": y_max,
        "success": True,
    }

    fields_df = pd.DataFrame({
        "y": y_ref,
        "p_ref_real": np.real(p_ref),
        "p_ref_imag": np.imag(p_ref),
        "p_gep_real": np.real(p_gep),
        "p_gep_imag": np.imag(p_gep),
        "rho_ref_real": np.real(rho_ref),
        "rho_ref_imag": np.imag(rho_ref),
        "rho_gep_real": np.real(rho_gep),
        "rho_gep_imag": np.imag(rho_gep),
        "u_ref_real": np.real(u_ref),
        "u_ref_imag": np.imag(u_ref),
        "u_gep_real": np.real(u_gep),
        "u_gep_imag": np.imag(u_gep),
        "v_ref_real": np.real(v_ref),
        "v_ref_imag": np.imag(v_ref),
        "v_gep_real": np.real(v_gep),
        "v_gep_imag": np.imag(v_gep),
    })

    return row, fields_df


def main():
    outdir = ROOT / "assets" / "gep_subsonic" / "classic_seeded_modes"
    outdir.mkdir(parents=True, exist_ok=True)

    mach = 0.5
    alphas = [0.3, 0.5, 0.7]
    n_values = [241, 301]

    rows = []

    for n_points in n_values:
        for alpha in alphas:
            print(f"[RUN] M={mach:g} alpha={alpha:g} N={n_points}")
            try:
                row, fields_df = run_one(
                    alpha=alpha,
                    mach=mach,
                    n_points=n_points,
                    mapping_kind="pin",
                    mapping_scale=5.0,
                    xi_max=0.98,
                )
            except Exception as exc:
                row = {
                    "alpha": alpha,
                    "Mach": mach,
                    "N": n_points,
                    "ci_classic": np.nan,
                    "gep_cr": np.nan,
                    "gep_ci": np.nan,
                    "ci_abs_err": np.nan,
                    "ci_rel_err": np.nan,
                    "p_rel": np.nan,
                    "rho_rel": np.nan,
                    "u_rel": np.nan,
                    "v_rel": np.nan,
                    "p_overlap": np.nan,
                    "selection_source": f"ERROR: {type(exc).__name__}: {exc}",
                    "n_finite_modes": 0,
                    "success": False,
                }
                fields_df = None

            rows.append(row)

            tag = f"M{mach:.3f}_a{alpha:.3f}_N{n_points}"
            if fields_df is not None:
                fields_df.to_csv(outdir / f"fields_classic_vs_gep_{tag}.csv", index=False)

            print(
                f"  ci_rel={row.get('ci_rel_err')} "
                f"p_rel={row.get('p_rel')} "
                f"u_rel={row.get('u_rel')} "
                f"v_rel={row.get('v_rel')} "
                f"overlap={row.get('p_overlap')} "
                f"source={row.get('selection_source')}"
            )

    summary = pd.DataFrame(rows)
    summary_path = outdir / "summary_classic_seeded_gep_modes.csv"
    summary.to_csv(summary_path, index=False)

    print("\n===== summary =====")
    print(summary.to_string(index=False))
    print(f"\nCSV: {summary_path}")


if __name__ == "__main__":
    main()
