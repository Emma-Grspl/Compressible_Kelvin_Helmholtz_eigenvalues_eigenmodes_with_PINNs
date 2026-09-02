from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from src.scripts.classical.solve_robust_subsonic_shooting import RobustSubsonicShootingSolver
from src.scripts.gep.selection.solve_dense_gep_notebook_style import NotebookStyleDenseGEPSolver


def run_one(alpha: float, mach: float, n_points: int, mapping_kind: str, mapping_scale: float, xi_max: float):
    classic = RobustSubsonicShootingSolver(alpha=alpha, Mach=mach).solve(force_cross_check=True)

    target = (0.0, float(classic.ci))

    solver = NotebookStyleDenseGEPSolver(
        alpha=alpha,
        Mach=mach,
        n_points=n_points,
        mapping_kind=mapping_kind,
        mapping_scale=mapping_scale,
        xi_max=xi_max,
    )

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
            "classic_ci": float(classic.ci),
            "classic_omega_i": float(classic.omega_i),
            "classic_source": classic.source,
            "classic_success": bool(classic.success),
            "gep_cr": np.nan,
            "gep_ci": np.nan,
            "gep_omega_i": np.nan,
            "ci_abs_err": np.nan,
            "ci_rel_err": np.nan,
            "distance_to_classic": np.nan,
            "selection_source": selection_source,
            "n_finite_modes": n_modes,
            "success": False,
        }

    distance = solver.spectral_distance(mode, target, ci_weight=2.0)

    return {
        "alpha": alpha,
        "Mach": mach,
        "N": n_points,
        "classic_ci": float(classic.ci),
        "classic_omega_i": float(classic.omega_i),
        "classic_source": classic.source,
        "classic_success": bool(classic.success),
        "gep_cr": float(mode["cr"]),
        "gep_ci": float(mode["ci"]),
        "gep_omega_i": float(mode["omega_i"]),
        "ci_abs_err": abs(float(mode["ci"]) - float(classic.ci)),
        "ci_rel_err": abs(float(mode["ci"]) - float(classic.ci)) / max(abs(float(classic.ci)), 1e-12),
        "distance_to_classic": float(distance),
        "selection_source": selection_source,
        "n_finite_modes": n_modes,
        "success": True,
    }


def main():
    outdir = ROOT / "assets" / "gep_subsonic"
    outdir.mkdir(parents=True, exist_ok=True)

    mach = 0.5
    alphas = [0.3, 0.5, 0.7]

    rows = []
    for n_points in [181, 241, 301, 361]:
        for alpha in alphas:
            print(f"[RUN] M={mach:g} alpha={alpha:g} N={n_points}")
            try:
                row = run_one(
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
                    "classic_ci": np.nan,
                    "classic_omega_i": np.nan,
                    "classic_source": "error",
                    "classic_success": False,
                    "gep_cr": np.nan,
                    "gep_ci": np.nan,
                    "gep_omega_i": np.nan,
                    "ci_abs_err": np.nan,
                    "ci_rel_err": np.nan,
                    "distance_to_classic": np.nan,
                    "selection_source": f"ERROR: {type(exc).__name__}: {exc}",
                    "n_finite_modes": 0,
                    "success": False,
                }
            rows.append(row)

            print(
                f"  classic_ci={row['classic_ci']} "
                f"gep=({row['gep_cr']}, {row['gep_ci']}) "
                f"ci_rel_err={row['ci_rel_err']} "
                f"source={row['selection_source']} "
                f"n_modes={row['n_finite_modes']}"
            )

    df = pd.DataFrame(rows)
    csv_path = outdir / "benchmark_subsonic_classic_seeded_gep_ci.csv"
    df.to_csv(csv_path, index=False)

    print("\n===== summary =====")
    print(df.to_string(index=False))
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
