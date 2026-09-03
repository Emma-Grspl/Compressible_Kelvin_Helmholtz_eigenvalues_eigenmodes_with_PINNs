#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def linear_root(alpha: np.ndarray, ci: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(alpha, ci, deg=1)

    if slope >= 0.0:
        return np.nan, slope

    return float(-intercept / slope), float(slope)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-extrapolation", type=float, default=0.01)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.alpha_traces)

    valid = data[
        data["accepted"].fillna(False)
        & np.isfinite(data["alpha"])
        & np.isfinite(data["ci"])
        & (data["ci"] > 0.0)
    ].copy()

    rows = []

    for Mach, group in valid.groupby("Mach"):
        group = group.sort_values("alpha").drop_duplicates(
            subset=["alpha"],
            keep="last",
        )

        if len(group) < 2:
            rows.append(
                {
                    "Mach": float(Mach),
                    "status": "insufficient_points",
                }
            )
            continue

        last = group.iloc[-1]
        previous = group.iloc[-2]

        alpha_last = float(last["alpha"])
        ci_last = float(last["ci"])

        alpha_two = group.tail(2)["alpha"].to_numpy(float)
        ci_two = group.tail(2)["ci"].to_numpy(float)

        alpha_secant, slope_secant = linear_root(
            alpha_two,
            ci_two,
        )

        if len(group) >= 3:
            alpha_three, slope_three = linear_root(
                group.tail(3)["alpha"].to_numpy(float),
                group.tail(3)["ci"].to_numpy(float),
            )
        else:
            alpha_three = np.nan
            slope_three = np.nan

        secant_valid = bool(
            np.isfinite(alpha_secant)
            and alpha_secant > alpha_last
            and alpha_secant
            <= alpha_last + args.max_extrapolation
        )

        three_valid = bool(
            np.isfinite(alpha_three)
            and alpha_three > alpha_last
            and alpha_three
            <= alpha_last + args.max_extrapolation
        )

        if secant_valid:
            alpha_selected = alpha_secant
            method = "last_two_secant"
        elif three_valid:
            alpha_selected = alpha_three
            method = "last_three_linear"
        else:
            alpha_selected = np.nan
            method = "no_valid_extrapolation"

        if secant_valid and three_valid:
            fit_spread = abs(alpha_secant - alpha_three)
        else:
            fit_spread = np.nan

        delta_alpha = (
            alpha_selected - alpha_last
            if np.isfinite(alpha_selected)
            else np.nan
        )

        cr_last = float(last["cr"])
        cr_previous = float(previous["cr"])

        if (
            np.isfinite(alpha_selected)
            and alpha_last != float(previous["alpha"])
        ):
            dcr_dalpha = (
                cr_last - cr_previous
            ) / (
                alpha_last - float(previous["alpha"])
            )

            cr_neutral = (
                cr_last
                + dcr_dalpha
                * (
                    alpha_selected - alpha_last
                )
            )
        else:
            dcr_dalpha = np.nan
            cr_neutral = np.nan

        rows.append(
            {
                "Mach": float(Mach),
                "n_points": int(len(group)),
                "alpha_previous": float(previous["alpha"]),
                "ci_previous": float(previous["ci"]),
                "alpha_last": alpha_last,
                "ci_last": ci_last,
                "alpha_neutral_secant": alpha_secant,
                "secant_slope_dci_dalpha": slope_secant,
                "alpha_neutral_last3": alpha_three,
                "last3_slope_dci_dalpha": slope_three,
                "alpha_neutral_selected": alpha_selected,
                "delta_alpha_from_last": delta_alpha,
                "fit_spread": fit_spread,
                "cr_neutral_estimate": cr_neutral,
                "dcr_dalpha_last": dcr_dalpha,
                "selected_method": method,
                "status": (
                    "ok"
                    if np.isfinite(alpha_selected)
                    else "failed"
                ),
            }
        )

    output = pd.DataFrame(rows).sort_values("Mach")

    csv_path = (
        args.output_dir
        / "neutral_boundary_secant_refit.csv"
    )
    output.to_csv(csv_path, index=False)

    report = {
        "n_mach": int(len(output)),
        "n_valid": int(
            np.isfinite(
                output["alpha_neutral_selected"]
            ).sum()
        ),
        "method": (
            "Linear zero crossing based primarily on the last "
            "two positive-ci shooting points."
        ),
    }

    (
        args.output_dir
        / "neutral_boundary_secant_refit_report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(output.to_string(index=False))
    print("\n", json.dumps(report, indent=2))
    print("\nwrote:", csv_path)


if __name__ == "__main__":
    main()
