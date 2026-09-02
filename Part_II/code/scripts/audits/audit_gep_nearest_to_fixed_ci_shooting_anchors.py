#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from classical_solver.gep.dense_gep_notebook_style import NotebookStyleDenseGEPSolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ci-weight", type=float, default=25.0)
    parser.add_argument("--n-points", type=int, default=301)
    args = parser.parse_args()

    anchors = pd.read_csv(args.anchors)
    rows = []

    for _, a in anchors.iterrows():
        mach = float(a["Mach"])
        alpha = float(a["alpha"])
        cr0 = float(a["shooting_cr"])
        ci0 = float(a["shooting_ci"])

        print(f"[gep-nearest] M={mach} alpha={alpha} target=({cr0}, {ci0})")

        solver = NotebookStyleDenseGEPSolver(
            alpha=alpha,
            Mach=mach,
            n_points=args.n_points,
            mapping_kind="pin",
            mapping_scale=5.0,
            xi_max=0.98,
        )
        modes = solver.finite_modes()

        best = None
        best_d = np.inf
        for mode in modes:
            cr = float(mode["cr"])
            ci = float(mode["ci"])
            d = ((cr - cr0) ** 2 + args.ci_weight * (ci - ci0) ** 2) ** 0.5
            if d < best_d:
                best = mode
                best_d = d

        if best is None:
            rows.append({
                **a.to_dict(),
                "n_finite_gep_modes": 0,
                "nearest_gep_cr": np.nan,
                "nearest_gep_ci": np.nan,
                "nearest_gep_distance": np.nan,
                "nearest_gep_abs_dcr": np.nan,
                "nearest_gep_abs_dci": np.nan,
                "gep_anchor_match_status": "no_gep_modes",
            })
            continue

        dcr = abs(float(best["cr"]) - cr0)
        dci = abs(float(best["ci"]) - ci0)

        if dcr <= 0.03 and dci <= max(5e-4, 0.25 * abs(ci0)):
            match_status = "gep_matches_fixed_ci_shooting_anchor"
        elif dci <= max(5e-4, 0.25 * abs(ci0)):
            match_status = "gep_ci_matches_but_cr_differs"
        elif dcr <= 0.03:
            match_status = "gep_cr_matches_but_ci_differs"
        else:
            match_status = "gep_does_not_match_anchor"

        rows.append({
            **a.to_dict(),
            "n_finite_gep_modes": len(modes),
            "nearest_gep_cr": float(best["cr"]),
            "nearest_gep_ci": float(best["ci"]),
            "nearest_gep_omega_i": float(best["omega_i"]),
            "nearest_gep_distance": float(best_d),
            "nearest_gep_abs_dcr": float(dcr),
            "nearest_gep_abs_dci": float(dci),
            "gep_anchor_match_status": match_status,
        })

        print(
            f"[gep-nearest-result] nearest=({best['cr']}, {best['ci']}), "
            f"dcr={dcr}, dci={dci}, status={match_status}"
        )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[gep-nearest] wrote {args.output}")


if __name__ == "__main__":
    main()
