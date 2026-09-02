#!/usr/bin/env python3
"""Merge and audit the full joint-PINN dense-GEP atlas campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def stats(frame: pd.DataFrame, column: str):
    if column not in frame:
        return {"mean": None, "max": None, "min": None}
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return {"mean": None, "max": None, "min": None}
    return {
        "mean": float(values.mean()),
        "max": float(values.max()),
        "min": float(values.min()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default=(
            "archive/csv/assets/pinn_subsonic/joint_ci_mode_atlas_v2/"
            "gep_chart_plan.tsv"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=(
            "assets/pinn_subsonic/"
            "joint_ci_mode_full_gep_atlas_v2"
        ),
    )
    args = parser.parse_args()

    plan = pd.read_csv(args.plan, sep="\t")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    chart_rows = []
    point_frames = []

    for _, item in plan.iterrows():
        chart_id = str(item["chart_id"])
        chart_dir = Path(item["output_dir"])
        summary_path = chart_dir / "summary.csv"
        metrics_path = chart_dir / "summary_metrics.json"

        row = {
            "chart_id": chart_id,
            "output_dir": str(chart_dir),
            "has_summary": summary_path.is_file(),
            "has_metrics": metrics_path.is_file(),
            "complete": False,
        }

        if summary_path.is_file():
            frame = pd.read_csv(summary_path)
            point_frames.append(frame)

            technical = frame["technical_success"].fillna(False).astype(bool)
            reference = frame["reference_success"].fillna(False).astype(bool)
            matched = (
                frame.get(
                    "pinn_matched_is_most_unstable",
                    pd.Series(False, index=frame.index),
                )
                .fillna(False)
                .astype(bool)
            )

            row.update(
                {
                    "n_points": int(len(frame)),
                    "n_technical_success": int(technical.sum()),
                    "n_reference_success": int(reference.sum()),
                    "n_matches_most_unstable": int(
                        (technical & matched).sum()
                    ),
                    "all_technical_success": bool(technical.all()),
                    "all_matches_most_unstable": bool(
                        technical.all() and matched.all()
                    ),
                    "complete": bool(
                        technical.all() and metrics_path.is_file()
                    ),
                }
            )

            for metric in [
                "ci_pinn_rel_err_classic",
                "pinn_matched_ci_rel_err_classic",
                "pinn_matched_p_overlap",
                "pinn_matched_q_overlap",
                "pinn_matched_p_rel_classic",
                "pinn_matched_u_rel_classic",
                "pinn_matched_v_rel_classic",
            ]:
                for key, value in stats(frame, metric).items():
                    row[f"{metric}_{key}"] = value

        chart_rows.append(row)

    charts = pd.DataFrame(chart_rows).sort_values("chart_id")
    charts.to_csv(output_root / "atlas_chart_summary.csv", index=False)

    if point_frames:
        points = pd.concat(
            point_frames,
            ignore_index=True,
            sort=False,
        ).sort_values(["chart_id", "Mach", "eta"])
    else:
        points = pd.DataFrame()

    points.to_csv(output_root / "atlas_pointwise_summary.csv", index=False)

    technical = (
        points.get("technical_success", pd.Series(dtype=bool))
        .fillna(False)
        .astype(bool)
    )
    matched = (
        points.get(
            "pinn_matched_is_most_unstable",
            pd.Series(False, index=points.index),
        )
        .fillna(False)
        .astype(bool)
    )
    continuation = (
        points.get(
            "continuation_required",
            pd.Series(False, index=points.index),
        )
        .fillna(False)
        .astype(bool)
    )

    technical_failures = points.loc[~technical].copy()
    matching_failures = points.loc[technical & ~matched].copy()
    near_neutral = points.loc[continuation].copy()

    technical_failures.to_csv(
        output_root / "technical_failures.csv",
        index=False,
    )
    matching_failures.to_csv(
        output_root / "matching_failures.csv",
        index=False,
    )
    near_neutral.to_csv(
        output_root / "near_neutral_points.csv",
        index=False,
    )

    metrics = [
        "ci_pinn_rel_err_classic",
        "pinn_matched_ci_rel_err_classic",
        "pinn_matched_p_overlap",
        "pinn_matched_q_overlap",
        "pinn_matched_p_rel_classic",
        "pinn_matched_u_rel_classic",
        "pinn_matched_v_rel_classic",
    ]

    report = {
        "n_planned_charts": int(len(plan)),
        "n_complete_charts": int(
            charts["complete"].fillna(False).sum()
        ),
        "n_points": int(len(points)),
        "n_technical_success": int(technical.sum()),
        "n_technical_failures": int((~technical).sum()),
        "n_pinn_matches_most_unstable": int(
            (technical & matched).sum()
        ),
        "n_matching_failures": int(
            (technical & ~matched).sum()
        ),
        "n_near_neutral_points": int(continuation.sum()),
        "metrics": {
            metric: stats(points, metric)
            for metric in metrics
        },
    }

    (output_root / "atlas_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("ATLAS GEP CAMPAIGN")
    print("=" * 100)
    print(json.dumps(report, indent=2, sort_keys=True))

    print("\nIncomplete charts:")
    incomplete_columns = [
        column
        for column in [
            "chart_id",
            "has_summary",
            "has_metrics",
            "n_points",
            "n_technical_success",
        ]
        if column in charts
    ]
    incomplete = charts.loc[
        ~charts["complete"].fillna(False),
        incomplete_columns,
    ]
    print(
        incomplete.to_string(index=False)
        if not incomplete.empty
        else "None"
    )

    print("\nMatching failures:")
    failure_columns = [
        column
        for column in [
            "chart_id",
            "point_id",
            "Mach",
            "eta",
            "alpha",
            "gep_regime",
            "ci_pinn",
            "pinn_matched_ci",
            "most_unstable_ci",
            "pinn_matched_p_overlap",
            "pinn_matched_q_overlap",
        ]
        if column in matching_failures
    ]
    print(
        matching_failures[failure_columns].to_string(index=False)
        if not matching_failures.empty
        else "None"
    )


if __name__ == "__main__":
    main()
