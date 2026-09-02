#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


TRAINING_S = np.array(
    [
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.65,
        0.78,
        0.87,
        0.93,
        0.97,
        0.985,
    ],
    dtype=float,
)

HOLDOUT_S = np.array(
    [
        0.075,
        0.15,
        0.275,
        0.425,
        0.575,
        0.715,
        0.825,
        0.90,
        0.95,
        0.978,
    ],
    dtype=float,
)


def chart_flags(
    Mach: float,
    s: float,
) -> dict[str, bool]:
    return {
        "chart_s_low": bool(
            s <= 0.18
        ),
        "chart_s_bulk": bool(
            0.12 <= s <= 0.88
        ),
        "chart_s_neutral": bool(
            s >= 0.82
        ),
        "chart_M_sonic": bool(
            Mach <= 1.25
        ),
        "chart_M_mid": bool(
            1.15 <= Mach <= 1.65
        ),
        "chart_M_high": bool(
            Mach >= 1.55
        ),
    }


def make_plan(
    mach_values: np.ndarray,
    s_values: np.ndarray,
    alpha_interpolator: PchipInterpolator,
    split: str,
) -> pd.DataFrame:
    rows = []

    for Mach in mach_values:
        alpha_neutral = float(
            alpha_interpolator(Mach)
        )

        for s in s_values:
            rows.append(
                {
                    "split": split,
                    "Mach": float(Mach),
                    "s": float(s),
                    "alpha_neutral": (
                        alpha_neutral
                    ),
                    "alpha": float(
                        s * alpha_neutral
                    ),
                    "distance_to_neutral": float(
                        1.0 - s
                    ),
                    **chart_flags(
                        float(Mach),
                        float(s),
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--neutral-refit",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dense-points",
        type=int,
        default=1001,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.read_csv(
        args.neutral_refit
    )

    required_columns = [
        "Mach",
        "alpha_neutral_selected",
        "cr_neutral_estimate",
        "delta_alpha_from_last",
        "status",
    ]

    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"Missing neutral columns: {missing}"
        )

    valid = frame[
        np.isfinite(
            pd.to_numeric(
                frame[
                    "alpha_neutral_selected"
                ],
                errors="coerce",
            )
        )
        & (
            frame["status"].astype(str)
            == "ok"
        )
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "No valid neutral-boundary points."
        )

    if "fit_spread" not in valid:
        valid["fit_spread"] = np.nan

    fit_spread = pd.to_numeric(
        valid["fit_spread"],
        errors="coerce",
    )

    delta = pd.to_numeric(
        valid["delta_alpha_from_last"],
        errors="coerce",
    ).abs()

    valid["_selection_quality"] = (
        fit_spread.fillna(
            delta.fillna(np.inf)
        )
    )

    valid = (
        valid
        .sort_values(
            [
                "Mach",
                "_selection_quality",
                "delta_alpha_from_last",
            ]
        )
        .drop_duplicates(
            subset=["Mach"],
            keep="first",
        )
        .drop(
            columns=["_selection_quality"]
        )
        .sort_values("Mach")
        .reset_index(drop=True)
    )

    Mach_nodes = valid[
        "Mach"
    ].to_numpy(dtype=float)

    alpha_nodes = valid[
        "alpha_neutral_selected"
    ].to_numpy(dtype=float)

    cr_nodes = valid[
        "cr_neutral_estimate"
    ].to_numpy(dtype=float)

    if not np.all(
        np.isfinite(cr_nodes)
    ):
        raise RuntimeError(
            "Some neutral cr estimates are missing."
        )

    if not np.isclose(
        Mach_nodes[0],
        1.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            f"Boundary starts at M={Mach_nodes[0]}, not 1."
        )

    if not np.isclose(
        Mach_nodes[-1],
        2.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            f"Boundary ends at M={Mach_nodes[-1]}, not 2."
        )

    if np.any(
        np.diff(Mach_nodes) <= 0.0
    ):
        raise RuntimeError(
            "Mach knots are not strictly increasing."
        )

    if np.any(
        np.diff(alpha_nodes) > 1.0e-8
    ):
        raise RuntimeError(
            "Neutral-boundary knots are not decreasing."
        )

    alpha_interpolator = PchipInterpolator(
        Mach_nodes,
        alpha_nodes,
        extrapolate=False,
    )

    cr_interpolator = PchipInterpolator(
        Mach_nodes,
        cr_nodes,
        extrapolate=False,
    )

    Mach_dense = np.linspace(
        1.0,
        2.0,
        args.dense_points,
    )

    alpha_dense = np.asarray(
        alpha_interpolator(Mach_dense),
        dtype=float,
    )

    cr_dense = np.asarray(
        cr_interpolator(Mach_dense),
        dtype=float,
    )

    derivative_dense = np.asarray(
        alpha_interpolator
        .derivative()(Mach_dense),
        dtype=float,
    )

    monotone = bool(
        np.all(
            np.diff(alpha_dense)
            <= 1.0e-10
        )
    )

    if not monotone:
        raise RuntimeError(
            "Dense PCHIP boundary is not decreasing."
        )

    pd.DataFrame(
        {
            "Mach": Mach_dense,
            "alpha_neutral": alpha_dense,
            "cr_neutral": cr_dense,
            "dalpha_neutral_dMach": (
                derivative_dense
            ),
        }
    ).to_csv(
        args.output_dir
        / "neutral_boundary_pchip_dense.csv",
        index=False,
    )

    valid.to_csv(
        args.output_dir
        / "neutral_boundary_knots.csv",
        index=False,
    )

    training_Mach = Mach_nodes

    holdout_Mach = (
        0.5
        * (
            Mach_nodes[:-1]
            + Mach_nodes[1:]
        )
    )

    training_plan = make_plan(
        training_Mach,
        TRAINING_S,
        alpha_interpolator,
        split="train",
    )

    holdout_plan = make_plan(
        holdout_Mach,
        HOLDOUT_S,
        alpha_interpolator,
        split="holdout",
    )

    training_plan.to_csv(
        args.output_dir
        / "atlas_training_plan.csv",
        index=False,
    )

    holdout_plan.to_csv(
        args.output_dir
        / "atlas_holdout_plan.csv",
        index=False,
    )

    np.savez_compressed(
        args.output_dir
        / "neutral_boundary_pchip.npz",
        Mach_nodes=Mach_nodes,
        alpha_neutral_nodes=alpha_nodes,
        cr_neutral_nodes=cr_nodes,
        Mach_dense=Mach_dense,
        alpha_neutral_dense=alpha_dense,
        cr_neutral_dense=cr_dense,
    )

    refinement = valid[
        pd.to_numeric(
            valid[
                "delta_alpha_from_last"
            ],
            errors="coerce",
        ).abs()
        > 5.0e-4
    ]

    report = {
        "n_boundary_knots": int(
            len(valid)
        ),
        "Mach_min": float(
            Mach_nodes[0]
        ),
        "Mach_max": float(
            Mach_nodes[-1]
        ),
        "alpha_neutral_min": float(
            np.min(alpha_dense)
        ),
        "alpha_neutral_max": float(
            np.max(alpha_dense)
        ),
        "boundary_monotone_decreasing": (
            monotone
        ),
        "n_training_parameter_points": int(
            len(training_plan)
        ),
        "n_holdout_parameter_points": int(
            len(holdout_plan)
        ),
        "n_boundary_points_requiring_refinement": int(
            len(refinement)
        ),
        "boundary_refinement_Mach": (
            refinement[
                "Mach"
            ].astype(float).tolist()
        ),
        "training_s": (
            TRAINING_S.tolist()
        ),
        "holdout_s": (
            HOLDOUT_S.tolist()
        ),
    }

    (
        args.output_dir
        / "atlas_plan_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
