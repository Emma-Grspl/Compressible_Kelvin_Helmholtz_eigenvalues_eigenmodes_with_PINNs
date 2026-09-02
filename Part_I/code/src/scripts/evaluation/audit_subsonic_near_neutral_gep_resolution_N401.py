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


def run_one(row, n_points=401, xi_max=0.98):
    mach = float(row["Mach"])
    eta = float(row["eta"])
    alpha = float(row["alpha"])
    ci_seed = float(row["ci_seed"])

    classic_fields, ci_classic = load_classic_full_mode(alpha, mach)

    y_ref = np.asarray(classic_fields["y"], dtype=float)
    p_ref = np.asarray(classic_fields["p"], dtype=np.complex128)
    u_ref = np.asarray(classic_fields["u"], dtype=np.complex128)
    v_ref = np.asarray(classic_fields["v"], dtype=np.complex128)

    solver = NotebookStyleDenseGEPSolver(
        alpha=alpha,
        Mach=mach,
        n_points=n_points,
        mapping_kind="pin",
        mapping_scale=5.0,
        xi_max=xi_max,
    )

    mode, selection_source, n_modes = solver.get_nearest_mode_to_target(
        target_guess=(0.0, ci_seed),
        prefer_positive_cr=False,
        ci_weight=2.0,
    )

    if mode is None:
        return {
            "Mach": mach, "eta": eta, "alpha": alpha,
            "N": n_points, "xi_max": xi_max,
            "ci_classic": float(ci_classic),
            "ci_seed": ci_seed,
            "gep_ci": np.nan,
            "ci_seed_rel_err": abs(ci_seed - float(ci_classic)) / max(abs(float(ci_classic)), 1e-12),
            "ci_gep_abs_err": np.nan,
            "ci_gep_rel_err": np.nan,
            "p_rel": np.nan,
            "u_rel": np.nan,
            "v_rel": np.nan,
            "selection_source": selection_source,
            "n_modes": n_modes,
            "success": False,
        }

    gep = split_gep_vector(mode["vector"], solver.n_points, mach)

    p_gep = interp_complex(solver.y, gep["p"], y_ref)
    u_gep = interp_complex(solver.y, gep["u"], y_ref)
    v_gep = interp_complex(solver.y, gep["v"], y_ref)

    y_min = max(float(np.min(y_ref)), float(np.min(solver.y)), -12.0)
    y_max = min(float(np.max(y_ref)), float(np.max(solver.y)), 12.0)
    mask = (y_ref >= y_min) & (y_ref <= y_max)

    scale = align_complex(p_gep, p_ref, mask)
    p_gep *= scale
    u_gep *= scale
    v_gep *= scale

    gep_ci = float(mode["ci"])

    return {
        "Mach": mach,
        "eta": eta,
        "alpha": alpha,
        "N": n_points,
        "xi_max": xi_max,
        "ci_classic": float(ci_classic),
        "ci_seed": ci_seed,
        "gep_ci": gep_ci,
        "ci_seed_rel_err": abs(ci_seed - float(ci_classic)) / max(abs(float(ci_classic)), 1e-12),
        "ci_gep_abs_err": abs(gep_ci - float(ci_classic)),
        "ci_gep_rel_err": abs(gep_ci - float(ci_classic)) / max(abs(float(ci_classic)), 1e-12),
        "p_rel": rel_l2(p_gep, p_ref, y_ref, mask),
        "u_rel": rel_l2(u_gep, u_ref, y_ref, mask),
        "v_rel": rel_l2(v_gep, v_ref, y_ref, mask),
        "selection_source": selection_source,
        "n_modes": n_modes,
        "success": True,
    }


def main():
    summary_path = ROOT / "assets/pinn_subsonic/local_atlas_v1/gep_core_atlas_ci_seeded_v2/summary_atlas_core_ci_seeded_gep_modes.csv"
    df = pd.read_csv(summary_path)

    targets = df[
        (df["eta"] >= 0.9575) &
        (df["Mach"].isin([0.10, 0.20, 0.40, 0.60, 0.70]))
    ].copy()

    rows = []
    for _, row in targets.iterrows():
        print(f"[RUN] M={row['Mach']} eta={row['eta']} N=401 xi=0.98")
        out = run_one(row, n_points=401, xi_max=0.98)
        rows.append(out)
        print(
            f"  success={out['success']} "
            f"ci_seed_rel={out['ci_seed_rel_err']} "
            f"ci_gep_rel={out['ci_gep_rel_err']} "
            f"p={out['p_rel']} u={out['u_rel']} v={out['v_rel']} "
            f"source={out['selection_source']}"
        )

    res = pd.DataFrame(rows)

    outdir = ROOT / "assets/pinn_subsonic/local_atlas_v1/gep_near_neutral_resolution_audit_N401"
    outdir.mkdir(parents=True, exist_ok=True)

    path = outdir / "summary_gep_near_neutral_N401.csv"
    res.to_csv(path, index=False)

    print("\n===== means =====")
    print(res[[
        "ci_seed_rel_err", "ci_gep_rel_err",
        "p_rel", "u_rel", "v_rel"
    ]].mean(numeric_only=True).to_string())

    print("\n===== worst ci_gep_rel =====")
    print(res.sort_values("ci_gep_rel_err", ascending=False).to_string(index=False))

    print("\nCSV:", path)


if __name__ == "__main__":
    main()
