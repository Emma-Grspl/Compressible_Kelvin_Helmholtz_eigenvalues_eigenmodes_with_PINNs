#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from classical_solver.supersonic.mstab17_supersonic_solver import Mstab17SupersonicSolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--cr-min", type=float, required=True)
    parser.add_argument("--cr-max", type=float, required=True)
    parser.add_argument("--ci-min", type=float, required=True)
    parser.add_argument("--ci-max", type=float, required=True)
    parser.add_argument("--n-cr", type=int, default=61)
    parser.add_argument("--n-ci", type=int, default=61)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    solver = Mstab17SupersonicSolver(
        alpha=args.alpha,
        Mach=args.mach,
        match_y=1.0,
        use_mapping=True,
        mapping_scale=5.0,
        min_y_limit=10.0,
        max_y_limit=1200.0,
        y_limit_factor=10.0,
        ln_p_right_min=-30.0,
        ln_p_right_max=5.0,
    )

    cr_vals = np.linspace(args.cr_min, args.cr_max, args.n_cr)
    ci_vals = np.linspace(args.ci_min, args.ci_max, args.n_ci)

    rows = []
    total = len(cr_vals) * len(ci_vals)
    k = 0

    for cr in cr_vals:
        for ci in ci_vals:
            k += 1
            if k % 100 == 0:
                print(f"[map] {k}/{total}")

            try:
                mismatch = float(solver.stage1_mismatch(float(cr), float(ci)))
                ok = np.isfinite(mismatch)
                exc = ""
            except Exception as e:
                mismatch = np.nan
                ok = False
                exc = repr(e)

            rows.append(
                {
                    "Mach": args.mach,
                    "alpha": args.alpha,
                    "cr": float(cr),
                    "ci": float(ci),
                    "stage1_mismatch": mismatch,
                    "finite": bool(ok),
                    "exception": exc,
                }
            )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    finite = df[np.isfinite(df["stage1_mismatch"])].copy()
    finite = finite.sort_values("stage1_mismatch")

    print("[map] wrote", args.output)
    print("[map] best points:")
    print(finite.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
