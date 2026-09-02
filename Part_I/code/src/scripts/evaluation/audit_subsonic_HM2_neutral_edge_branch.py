from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from src.scripts.gep.selection.solve_dense_gep_notebook_style import (
    NotebookStyleDenseGEPSolver,
)
from src.scripts.evaluation.evaluate_kh_subsonic_fixed_mach_modal_candidates import (
    load_classic_full_mode,
)


MACH = 0.97
ETAS = np.array(
    [0.935, 0.940, 0.945, 0.950, 0.955,
     0.960, 0.965, 0.970, 0.975, 0.980],
    dtype=float,
)

N_POINTS = 401
MAPPING_SCALE = 5.0
XI_MAX = 0.98

CHART = Path(
    "assets/pinn_subsonic/local_atlas_ext_M005_098_eta005_098/"
    "seedGEP_pq2d_ATLAS_HM2_NEUTRAL3_"
    "M0.85_0.98_eta0.92_0.98_ciIDW_EVALFIX"
)

OUTDIR = Path(
    "assets/pinn_subsonic/local_atlas_v1/"
    "audit_HM2_neutral_edge_branch_M097"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


def build_seed_interpolant() -> tuple[np.ndarray, np.ndarray]:
    diag_path = CHART / "diagnostics_summary.csv"
    if not diag_path.exists():
        raise FileNotFoundError(diag_path)

    df = pd.read_csv(diag_path)
    df = df[np.isclose(df["Mach"], MACH, atol=1e-8)].copy()

    if df.empty:
        raise RuntimeError(f"No diagnostics at Mach={MACH} in {diag_path}")

    df = df.sort_values("eta")
    return (
        df["eta"].to_numpy(dtype=float),
        df["ci_pred"].to_numpy(dtype=float),
    )


def interpolate_seed(
    eta: float,
    eta_known: np.ndarray,
    ci_known: np.ndarray,
) -> float:
    return float(np.interp(eta, eta_known, ci_known))


def nearest_mode(
    modes: list[dict],
    target: tuple[float, float],
    ci_weight: float = 2.0,
) -> dict:
    if not modes:
        raise RuntimeError("No finite unstable GEP modes")

    return min(
        modes,
        key=lambda mode: np.sqrt(
            (mode["cr"] - target[0]) ** 2
            + (ci_weight * (mode["ci"] - target[1])) ** 2
        ),
    )


def main() -> None:
    eta_known, ci_known = build_seed_interpolant()

    rows: list[dict] = []
    candidate_rows: list[dict] = []

    previous_guess: tuple[float, float] | None = None
    previous_signature: np.ndarray | None = None

    alpha_cut = np.sqrt(1.0 - MACH**2)

    for eta in ETAS:
        alpha = float(eta * alpha_cut)
        _, ci_classic = load_classic_full_mode(alpha, MACH)
        ci_seed = interpolate_seed(eta, eta_known, ci_known)

        print(
            f"[RUN] M={MACH:.3f} eta={eta:.3f} "
            f"alpha={alpha:.8f} "
            f"ci_classic={ci_classic:.8f} "
            f"ci_seed={ci_seed:.8f}"
        )

        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=MACH,
            n_points=N_POINTS,
            mapping_kind="pin",
            mapping_scale=MAPPING_SCALE,
            xi_max=XI_MAX,
        )

        modes = solver.finite_modes()
        if not modes:
            raise RuntimeError(f"No modes at eta={eta}")

        # Diagnostic oracle: closest available GEP eigenvalue to classic ci.
        mode_classic = nearest_mode(modes, (0.0, float(ci_classic)))

        # Production-like selection: closest to PINN seed.
        mode_seed = nearest_mode(modes, (0.0, ci_seed))

        # Branch continuation selection.
        if previous_guess is None:
            mode_cont = mode_seed
            cont_source = "initial_seed"
        else:
            mode_cont, cont_source, _ = solver.get_branch_mode(
                target_guess=(0.0, ci_seed),
                previous_guess=previous_guess,
                previous_signature=previous_signature,
                prefer_positive_cr=False,
                ci_weight=2.0,
                spectral_window_factor=2.5,
                spectral_window_floor=0.015,
                overlap_top_k=12,
                overlap_weight=0.50,
                jump_cr_weight=0.25,
                jump_ci_weight=0.50,
            )
            if mode_cont is None:
                raise RuntimeError(
                    f"Continuation selection failed at eta={eta}"
                )

        previous_guess = (
            float(mode_cont["cr"]),
            float(mode_cont["ci"]),
        )
        previous_signature = np.asarray(
            mode_cont["signature"],
            dtype=float,
        )

        rows.append(
            {
                "Mach": MACH,
                "eta": eta,
                "alpha": alpha,
                "N": N_POINTS,
                "mapping_scale": MAPPING_SCALE,
                "xi_max": XI_MAX,
                "ci_classic": float(ci_classic),
                "ci_seed": ci_seed,
                "ci_nearest_seed": float(mode_seed["ci"]),
                "cr_nearest_seed": float(mode_seed["cr"]),
                "ci_continuation": float(mode_cont["ci"]),
                "cr_continuation": float(mode_cont["cr"]),
                "continuation_source": cont_source,
                "ci_nearest_classic": float(mode_classic["ci"]),
                "cr_nearest_classic": float(mode_classic["cr"]),
                "seed_abs_err": abs(ci_seed - ci_classic),
                "nearest_seed_abs_err": abs(
                    mode_seed["ci"] - ci_classic
                ),
                "continuation_abs_err": abs(
                    mode_cont["ci"] - ci_classic
                ),
                "nearest_classic_abs_err": abs(
                    mode_classic["ci"] - ci_classic
                ),
                "n_finite_modes": len(modes),
            }
        )

        # Export the modes closest to the imaginary axis.
        near_axis = sorted(
            modes,
            key=lambda mode: (
                abs(mode["cr"]),
                abs(mode["ci"] - ci_seed),
            ),
        )[:20]

        for rank, mode in enumerate(near_axis, start=1):
            candidate_rows.append(
                {
                    "Mach": MACH,
                    "eta": eta,
                    "alpha": alpha,
                    "rank": rank,
                    "cr": float(mode["cr"]),
                    "ci": float(mode["ci"]),
                    "omega_i": float(mode["omega_i"]),
                    "abs_cr": abs(float(mode["cr"])),
                    "distance_to_seed": solver.spectral_distance(
                        mode,
                        (0.0, ci_seed),
                        ci_weight=2.0,
                    ),
                    "distance_to_classic": solver.spectral_distance(
                        mode,
                        (0.0, float(ci_classic)),
                        ci_weight=2.0,
                    ),
                }
            )

    summary = pd.DataFrame(rows)
    candidates = pd.DataFrame(candidate_rows)

    summary_path = OUTDIR / "summary_branch_audit.csv"
    candidates_path = OUTDIR / "candidate_modes.csv"

    summary.to_csv(summary_path, index=False)
    candidates.to_csv(candidates_path, index=False)

    print("\n===== BRANCH AUDIT =====")
    print(summary.to_string(index=False))
    print()
    print("Summary:", summary_path)
    print("Candidates:", candidates_path)


if __name__ == "__main__":
    main()
