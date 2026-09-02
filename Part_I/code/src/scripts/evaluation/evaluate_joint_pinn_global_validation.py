#!/usr/bin/env python3
"""
Global validation campaign for the jointly trained subsonic KH PINN atlas.

Subcommands
-----------
aggregate
    Aggregate the 49 existing diagnostics_summary.csv files, add robust spectral
    metrics, identify anchors, deduplicate physical points, and audit existing
    duplicated evaluations.

build-plans
    Build:
      - the historical 384-point off-grid plan;
      - dense chart-overlap/seam points;
      - targeted neutral / long-wave / ultralow sweeps.

validate-points
    Evaluate one shard of an off-grid or targeted plan against the classical
    shooting/modal reference. No GEP is used.

validate-seams
    Evaluate one shard of chart-pair overlap points. No classical solve is used;
    this measures atlas-to-atlas continuity directly.

merge
    Merge array shards, write grouped metrics, and build a unique outlier plan.

recheck-outliers
    Re-run robust classical shooting with forced primary/secondary cross-check
    and denser scans for one shard of the outlier plan, including two nearby
    eta points when they stay inside the operational rectangle.

finalize
    Build the global report, publication-ready diagnostic figures, a frozen
    release tree, provenance, and SHA-256 manifests.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.scripts.classical.solve_robust_subsonic_shooting import (
    RobustSubsonicShootingSolver,
)
from src.scripts.evaluation.evaluate_kh_subsonic_fixed_mach_modal_candidates import (
    load_classic_full_mode,
)
from src.scripts.gep.selection.audit_mid_joint_pinn_full_gep import (
    alpha_from_eta,
    call_pinn_profiles,
    evaluate_pinn,
    overlap_complex,
    phase_alignment,
    rel_l2,
)

DEFAULT_TRAIN_PLAN = (
    "archive/csv/archive/csv/assets/pinn_subsonic/joint_ci_mode_atlas_v2/training_plan.tsv"
)
DEFAULT_VALIDATION_ROOT = (
    "assets/pinn_subsonic/joint_ci_mode_global_validation_v1"
)
DEFAULT_HISTORICAL_POINTS = (
    "assets/pinn_subsonic/csv/curated/pinn_subsonic/configs/manifests/Table_offgrid_validation_points_384.csv"
)

CI_SCALE_FLOOR = 5.0e-2
OMEGA_SCALE_FLOOR = 1.0e-3
DEFAULT_YMAX = 12.0
DEFAULT_NY = 801
DEFAULT_AMP_FLOOR = 0.02

SPECTRAL_COLUMNS = [
    "ci_abs_err",
    "ci_rel_err",
    "ci_rel_err_reg",
    "omega_abs_err",
    "omega_rel_err",
    "omega_rel_err_reg",
]
MODAL_COLUMNS = [
    "p_rel",
    "q_rel",
    "rho_rel",
    "u_rel",
    "v_rel",
    "gamma_rel",
    "p_overlap",
    "q_overlap",
    "rho_overlap",
    "u_overlap",
    "v_overlap",
]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    return value


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values[np.isfinite(values)]


def metric_stats(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    values = finite_series(frame, column)
    if values.empty:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def grouped_metric_report(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    *,
    group_columns: Iterable[str] = (),
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "metrics": {
            metric: metric_stats(frame, metric)
            for metric in metrics
        },
    }
    for group_column in group_columns:
        if group_column not in frame:
            continue
        grouped: dict[str, Any] = {}
        for group_name, group in frame.groupby(group_column, dropna=False):
            grouped[str(group_name)] = {
                "n_rows": int(len(group)),
                "metrics": {
                    metric: metric_stats(group, metric)
                    for metric in metrics
                },
            }
        report[f"by_{group_column}"] = grouped
    return report


def normalize_plan(training_plan_path: Path) -> pd.DataFrame:
    plan = pd.read_csv(training_plan_path, sep="\t").copy()
    required = {
        "chart_id",
        "output_dir",
        "mach_min",
        "mach_max",
        "eta_min",
        "eta_max",
    }
    missing = sorted(required.difference(plan.columns))
    if missing:
        raise KeyError(
            f"{training_plan_path} is missing required columns {missing}"
        )

    plan["chart_id"] = plan["chart_id"].astype(str)
    for column in ("mach_min", "mach_max", "eta_min", "eta_max"):
        plan[column] = pd.to_numeric(plan[column], errors="raise")

    plan["checkpoint"] = plan["output_dir"].map(
        lambda value: str(Path(str(value)) / "model_state.pt")
    )
    plan["diagnostics_csv"] = plan["output_dir"].map(
        lambda value: str(Path(str(value)) / "diagnostics_summary.csv")
    )
    plan["anchor_csv"] = plan["output_dir"].map(
        lambda value: str(Path(str(value)) / "ci_anchor_points.csv")
    )
    plan["chart_area"] = (
        (plan["mach_max"] - plan["mach_min"])
        * (plan["eta_max"] - plan["eta_min"])
    )
    plan = plan.sort_values("chart_id").reset_index(drop=True)

    for _, row in plan.iterrows():
        for column in ("checkpoint", "diagnostics_csv"):
            path = Path(str(row[column]))
            if not path.is_file():
                raise FileNotFoundError(
                    f"{row['chart_id']}: missing {column}: {path}"
                )
    return plan


def route_chart(
    plan: pd.DataFrame,
    mach: float,
    eta: float,
    preferred: str | None = None,
) -> pd.Series:
    tolerance = 5.0e-10
    covering = plan.loc[
        (plan["mach_min"] - tolerance <= mach)
        & (mach <= plan["mach_max"] + tolerance)
        & (plan["eta_min"] - tolerance <= eta)
        & (eta <= plan["eta_max"] + tolerance)
    ].copy()

    if covering.empty:
        raise RuntimeError(
            f"No chart covers M={mach:.9g}, eta={eta:.9g}."
        )

    if preferred is not None:
        preferred_rows = covering.loc[
            covering["chart_id"].astype(str).eq(str(preferred))
        ]
        if not preferred_rows.empty:
            return preferred_rows.iloc[0]

    m_half = 0.5 * (
        covering["mach_max"] - covering["mach_min"]
    ).clip(lower=1.0e-12)
    e_half = 0.5 * (
        covering["eta_max"] - covering["eta_min"]
    ).clip(lower=1.0e-12)
    m_center = 0.5 * (
        covering["mach_max"] + covering["mach_min"]
    )
    e_center = 0.5 * (
        covering["eta_max"] + covering["eta_min"]
    )
    covering["route_center_distance"] = (
        ((mach - m_center) / m_half) ** 2
        + ((eta - e_center) / e_half) ** 2
    )
    covering = covering.sort_values(
        ["chart_area", "route_center_distance", "chart_id"],
        kind="mergesort",
    )
    return covering.iloc[0]


def point_key(mach: float, eta: float) -> str:
    return f"M{float(mach):.10f}_eta{float(eta):.10f}"


def add_spectral_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    result["ci_ref"] = pd.to_numeric(result["ci_ref"], errors="coerce")
    result["ci_pred"] = pd.to_numeric(result["ci_pred"], errors="coerce")
    result["alpha"] = pd.to_numeric(result["alpha"], errors="coerce")

    result["ci_abs_err"] = (result["ci_pred"] - result["ci_ref"]).abs()
    result["ci_rel_err"] = (
        result["ci_abs_err"]
        / result["ci_ref"].abs().clip(lower=1.0e-12)
    )
    result["ci_rel_err_reg"] = (
        result["ci_abs_err"]
        / result["ci_ref"].abs().clip(lower=CI_SCALE_FLOOR)
    )

    result["omega_ref"] = result["alpha"] * result["ci_ref"]
    result["omega_pred"] = result["alpha"] * result["ci_pred"]
    result["omega_abs_err"] = (
        result["omega_pred"] - result["omega_ref"]
    ).abs()
    result["omega_rel_err"] = (
        result["omega_abs_err"]
        / result["omega_ref"].abs().clip(lower=1.0e-12)
    )
    result["omega_rel_err_reg"] = (
        result["omega_abs_err"]
        / result["omega_ref"].abs().clip(lower=OMEGA_SCALE_FLOOR)
    )
    result["near_neutral"] = result["eta"] >= 0.92
    result["point_key"] = [
        point_key(mach, eta)
        for mach, eta in zip(result["Mach"], result["eta"])
    ]
    return result


def find_historical_points(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path(DEFAULT_HISTORICAL_POINTS),
            Path(
                "assets/pinn_subsonic/csv/curated/pinn_subsonic/configs/"
                "manifests/Table_offgrid_validation_points_384.csv"
            ),
            Path(
                "assets/pinn_subsonic/csv/curated/pinn_subsonic/data/"
                "scientific_outputs/release_v1/validation/"
                "Table_offgrid_validation_results_384_release.csv"
            ),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the historical 384-point validation file. "
        f"Tried: {[str(path) for path in candidates]}"
    )


def mark_anchors(
    diagnostics: pd.DataFrame,
    anchor_path: Path,
) -> pd.Series:
    if not anchor_path.is_file():
        return pd.Series(False, index=diagnostics.index)

    anchors = pd.read_csv(anchor_path)
    if anchors.empty:
        return pd.Series(False, index=diagnostics.index)

    anchor_keys = set()
    for _, row in anchors.iterrows():
        if "eta" in anchors.columns:
            eta = float(row["eta"])
        elif "alpha" in anchors.columns:
            mach = float(row["Mach"])
            alpha_cut = math.sqrt(max(1.0 - mach * mach, 1.0e-14))
            eta = float(row["alpha"]) / alpha_cut
        else:
            continue
        anchor_keys.add(
            (
                round(float(row["Mach"]), 10),
                round(eta, 10),
            )
        )

    return pd.Series(
        [
            (
                round(float(mach), 10),
                round(float(eta), 10),
            )
            in anchor_keys
            for mach, eta in zip(
                diagnostics["Mach"],
                diagnostics["eta"],
            )
        ],
        index=diagnostics.index,
    )


def command_aggregate(args: argparse.Namespace) -> None:
    validation_root = Path(args.validation_root)
    output_dir = validation_root / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = normalize_plan(Path(args.training_plan))
    frames = []

    for _, chart in plan.iterrows():
        diagnostics_path = Path(chart["diagnostics_csv"])
        frame = pd.read_csv(diagnostics_path).copy()
        frame.insert(0, "chart_id", str(chart["chart_id"]))
        frame["field_family"] = str(chart.get("family", ""))
        for column in ("mach_min", "mach_max", "eta_min", "eta_max"):
            frame[column] = float(chart[column])

        frame["is_ci_anchor"] = mark_anchors(
            frame,
            Path(chart["anchor_csv"]),
        )

        if "q_rel" not in frame and "p_y_rel" in frame:
            frame["q_rel"] = frame["p_y_rel"]
        elif "q_rel" in frame and "p_y_rel" in frame:
            frame["q_rel"] = frame["q_rel"].fillna(frame["p_y_rel"])

        frames.append(frame)

    all_evaluations = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )
    all_evaluations = add_spectral_metrics(all_evaluations)
    all_evaluations = all_evaluations.sort_values(
        ["Mach", "eta", "chart_id"]
    ).reset_index(drop=True)
    all_evaluations.to_csv(
        output_dir / "direct_diagnostics_all_chart_evaluations.csv",
        index=False,
    )

    consensus_rows = []
    canonical_rows = []

    for key, group in all_evaluations.groupby("point_key", sort=True):
        first = group.iloc[0]
        routed = route_chart(
            plan,
            float(first["Mach"]),
            float(first["eta"]),
        )
        canonical = group.loc[
            group["chart_id"].astype(str).eq(
                str(routed["chart_id"])
            )
        ]
        if canonical.empty:
            canonical = group.sort_values("chart_id").iloc[[0]]

        canonical_rows.append(canonical.iloc[0].to_dict())

        row: dict[str, Any] = {
            "point_key": key,
            "Mach": float(first["Mach"]),
            "eta": float(first["eta"]),
            "alpha": float(first["alpha"]),
            "n_charts": int(len(group)),
            "charts": "|".join(sorted(group["chart_id"].astype(str))),
            "canonical_chart": str(canonical.iloc[0]["chart_id"]),
            "any_anchor": bool(as_bool(group["is_ci_anchor"]).any()),
            "ci_ref_median": float(group["ci_ref"].median()),
            "ci_ref_spread": float(
                group["ci_ref"].max() - group["ci_ref"].min()
            ),
            "ci_pred_median": float(group["ci_pred"].median()),
            "ci_pred_spread": float(
                group["ci_pred"].max() - group["ci_pred"].min()
            ),
        }
        for metric in SPECTRAL_COLUMNS + [
            "p_rel",
            "q_rel",
            "rho_rel",
            "u_rel",
            "v_rel",
            "gamma_rel",
        ]:
            values = finite_series(group, metric)
            if values.empty:
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        consensus_rows.append(row)

    consensus = pd.DataFrame(consensus_rows).sort_values(
        ["Mach", "eta"]
    )
    canonical = pd.DataFrame(canonical_rows).sort_values(
        ["Mach", "eta"]
    )

    consensus.to_csv(
        output_dir / "direct_diagnostics_unique_consensus.csv",
        index=False,
    )
    canonical.to_csv(
        output_dir / "direct_diagnostics_unique_canonical.csv",
        index=False,
    )

    overlaps = consensus.loc[consensus["n_charts"] > 1].copy()
    overlaps.to_csv(
        output_dir / "existing_duplicate_point_audit.csv",
        index=False,
    )

    report = {
        "n_charts": int(plan["chart_id"].nunique()),
        "n_chart_evaluations": int(len(all_evaluations)),
        "n_unique_physical_points": int(len(consensus)),
        "n_unique_points_with_multiple_charts": int(len(overlaps)),
        "n_anchor_evaluations": int(
            as_bool(all_evaluations["is_ci_anchor"]).sum()
        ),
        "all_evaluations": grouped_metric_report(
            all_evaluations,
            SPECTRAL_COLUMNS
            + [
                "p_rel",
                "q_rel",
                "rho_rel",
                "u_rel",
                "v_rel",
                "gamma_rel",
            ],
            group_columns=("near_neutral", "field_family", "chart_id"),
        ),
        "off_anchor_evaluations": grouped_metric_report(
            all_evaluations.loc[
                ~as_bool(all_evaluations["is_ci_anchor"])
            ],
            SPECTRAL_COLUMNS
            + [
                "p_rel",
                "q_rel",
                "rho_rel",
                "u_rel",
                "v_rel",
                "gamma_rel",
            ],
            group_columns=("near_neutral", "field_family"),
        ),
        "canonical_unique_points": grouped_metric_report(
            canonical,
            SPECTRAL_COLUMNS
            + [
                "p_rel",
                "q_rel",
                "rho_rel",
                "u_rel",
                "v_rel",
                "gamma_rel",
            ],
            group_columns=("near_neutral", "field_family"),
        ),
    }
    write_json(
        output_dir / "direct_diagnostics_metrics.json",
        json_safe(report),
    )

    print(json.dumps(json_safe(report), indent=2, sort_keys=True))
    print("Wrote:", output_dir)


def normalize_points_source(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()

    rename_candidates = {
        "M": "Mach",
        "mach": "Mach",
        "Eta": "eta",
    }
    for old, new in rename_candidates.items():
        if old in frame and new not in frame:
            frame = frame.rename(columns={old: new})

    required = {"Mach", "eta"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Historical point file missing {missing}")

    frame["Mach"] = pd.to_numeric(frame["Mach"], errors="raise")
    frame["eta"] = pd.to_numeric(frame["eta"], errors="raise")

    if "alpha" not in frame:
        frame["alpha"] = [
            alpha_from_eta(float(eta), float(mach))
            for mach, eta in zip(frame["Mach"], frame["eta"])
        ]

    if "point_id" not in frame:
        frame["point_id"] = [
            f"OFFGRID_{index:04d}"
            for index in range(len(frame))
        ]

    if "sample_group" not in frame:
        frame["sample_group"] = np.where(
            frame["eta"] >= 0.92,
            "near_neutral",
            "non_neutral",
        )

    return (
        frame.sort_values(["point_id", "Mach", "eta"])
        .drop_duplicates(["point_id"])
        .reset_index(drop=True)
    )


def attach_route(
    points: pd.DataFrame,
    plan: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, point in points.iterrows():
        preferred = (
            None
            if "chart_id" not in points.columns
            or pd.isna(point.get("chart_id"))
            else str(point.get("chart_id"))
        )
        chart = route_chart(
            plan,
            float(point["Mach"]),
            float(point["eta"]),
            preferred=preferred,
        )
        row = point.to_dict()
        row.update(
            {
                "chart_id": str(chart["chart_id"]),
                "checkpoint": str(chart["checkpoint"]),
                "chart_mach_min": float(chart["mach_min"]),
                "chart_mach_max": float(chart["mach_max"]),
                "chart_eta_min": float(chart["eta_min"]),
                "chart_eta_max": float(chart["eta_max"]),
                "field_family_plan": str(chart.get("family", "")),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_targeted_points() -> pd.DataFrame:
    rows = []
    index = 0

    neutral_machs = [0.70, 0.85, 0.915, 0.95, 0.98]
    neutral_etas = np.linspace(0.90, 0.98, 17)
    for mach in neutral_machs:
        for eta in neutral_etas:
            rows.append(
                {
                    "point_id": f"TARGET_NEUTRAL_{index:04d}",
                    "sample_group": "target_neutral",
                    "Mach": mach,
                    "eta": float(eta),
                }
            )
            index += 1

    longwave_machs = [
        0.02,
        0.05,
        0.10,
        0.30,
        0.50,
        0.70,
        0.85,
        0.95,
        0.98,
    ]
    longwave_etas = np.linspace(0.02, 0.12, 16)
    for mach in longwave_machs:
        for eta in longwave_etas:
            rows.append(
                {
                    "point_id": f"TARGET_LONGWAVE_{index:04d}",
                    "sample_group": "target_longwave",
                    "Mach": mach,
                    "eta": float(eta),
                }
            )
            index += 1

    ultralow_values = np.linspace(0.02, 0.08, 10)
    for mach in ultralow_values:
        for eta in ultralow_values:
            rows.append(
                {
                    "point_id": f"TARGET_ULTRALOW_{index:04d}",
                    "sample_group": "target_ultralow",
                    "Mach": float(mach),
                    "eta": float(eta),
                }
            )
            index += 1

    frame = pd.DataFrame(rows)
    frame["alpha"] = [
        alpha_from_eta(float(eta), float(mach))
        for mach, eta in zip(frame["Mach"], frame["eta"])
    ]
    return frame


def build_seam_points(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seam_index = 0

    for first_index in range(len(plan)):
        first = plan.iloc[first_index]
        for second_index in range(first_index + 1, len(plan)):
            second = plan.iloc[second_index]

            m_low = max(first["mach_min"], second["mach_min"])
            m_high = min(first["mach_max"], second["mach_max"])
            e_low = max(first["eta_min"], second["eta_min"])
            e_high = min(first["eta_max"], second["eta_max"])

            if m_high - m_low <= 1.0e-8:
                continue
            if e_high - e_low <= 1.0e-8:
                continue

            seam_id = (
                f"SEAM_{seam_index:04d}_"
                f"{first['chart_id']}__{second['chart_id']}"
            )
            m_values = [
                0.5 * (m_low + m_high),
                m_low + 0.2 * (m_high - m_low),
                m_low + 0.8 * (m_high - m_low),
            ]
            e_values = [
                0.5 * (e_low + e_high),
                e_low + 0.2 * (e_high - e_low),
                e_low + 0.8 * (e_high - e_low),
            ]
            samples = [
                (m_values[0], e_values[0]),
                (m_values[1], e_values[1]),
                (m_values[1], e_values[2]),
                (m_values[2], e_values[1]),
                (m_values[2], e_values[2]),
            ]

            for point_index, (mach, eta) in enumerate(samples):
                rows.append(
                    {
                        "point_id": f"{seam_id}_P{point_index}",
                        "seam_id": seam_id,
                        "chart_a": str(first["chart_id"]),
                        "checkpoint_a": str(first["checkpoint"]),
                        "chart_b": str(second["chart_id"]),
                        "checkpoint_b": str(second["checkpoint"]),
                        "Mach": float(mach),
                        "eta": float(eta),
                        "alpha": alpha_from_eta(float(eta), float(mach)),
                        "overlap_mach_min": float(m_low),
                        "overlap_mach_max": float(m_high),
                        "overlap_eta_min": float(e_low),
                        "overlap_eta_max": float(e_high),
                    }
                )
            seam_index += 1

    return pd.DataFrame(rows)


def command_build_plans(args: argparse.Namespace) -> None:
    validation_root = Path(args.validation_root)
    output_dir = validation_root / "plans"
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = normalize_plan(Path(args.training_plan))
    plan.to_csv(output_dir / "chart_catalog.tsv", sep="\t", index=False)

    historical_path = find_historical_points(args.historical_points)
    historical = normalize_points_source(pd.read_csv(historical_path))
    if len(historical) != 384:
        raise RuntimeError(
            f"Expected 384 historical points, found {len(historical)} "
            f"in {historical_path}."
        )
    historical = attach_route(historical, plan)
    historical.to_csv(
        output_dir / "offgrid_384_plan.csv",
        index=False,
    )

    targeted = attach_route(build_targeted_points(), plan)
    targeted.to_csv(
        output_dir / "targeted_scans_plan.csv",
        index=False,
    )

    seams = build_seam_points(plan)
    seams.to_csv(
        output_dir / "seam_dense_plan.csv",
        index=False,
    )

    report = {
        "historical_source": str(historical_path),
        "n_offgrid_points": int(len(historical)),
        "n_targeted_points": int(len(targeted)),
        "targeted_counts": {
            str(key): int(value)
            for key, value
            in targeted["sample_group"].value_counts().items()
        },
        "n_seam_pairs": int(seams["seam_id"].nunique()),
        "n_seam_points": int(len(seams)),
    }
    write_json(output_dir / "plan_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Wrote:", output_dir)


def fields_from_pq(
    y: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    alpha: float,
    mach: float,
    ci: float,
) -> dict[str, np.ndarray]:
    c = 1j * float(ci)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=np.complex128)
    q = np.asarray(q, dtype=np.complex128)
    u_bar = np.tanh(y)
    du_bar = 1.0 - u_bar**2
    denom = u_bar - c

    rho = float(mach) ** 2 * p
    v = -q / (1j * float(alpha) * denom)
    u = -(du_bar * v + 1j * float(alpha) * p) / (
        1j * float(alpha) * denom
    )
    gamma = q / np.where(
        np.abs(p) > 1.0e-14,
        p,
        np.nan + 1j * np.nan,
    )
    return {
        "p": p,
        "q": q,
        "rho": rho,
        "u": u,
        "v": v,
        "gamma": gamma,
    }


def reference_q_from_fields(
    y: np.ndarray,
    v: np.ndarray,
    alpha: float,
    mach: float,
    ci: float,
) -> np.ndarray:
    del mach
    u_bar = np.tanh(np.asarray(y, dtype=float))
    return (
        -1j
        * float(alpha)
        * (u_bar - 1j * float(ci))
        * np.asarray(v, dtype=np.complex128)
    )


def modal_mask(
    y: np.ndarray,
    p_reference: np.ndarray,
    *,
    y_max: float,
    amplitude_floor: float,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p_reference = np.asarray(p_reference, dtype=np.complex128)
    mask = np.abs(y) <= float(y_max)

    finite = np.abs(p_reference[np.isfinite(p_reference)])
    if finite.size:
        mask &= (
            np.abs(p_reference)
            >= float(amplitude_floor) * float(np.max(finite))
        )

    if int(np.count_nonzero(mask)) < 20:
        mask = np.abs(y) <= float(y_max)
    if int(np.count_nonzero(mask)) < 20:
        mask = np.isfinite(y)
    return mask


def evaluate_direct_point(
    point: pd.Series,
    *,
    model_cache: dict[str, tuple[Any, ...]],
    device: torch.device,
    y_max: float,
    amplitude_floor: float,
) -> dict[str, Any]:
    chart_id = str(point["chart_id"])
    checkpoint_path = Path(str(point["checkpoint"]))

    if chart_id not in model_cache:
        model_cache[chart_id] = evaluate_pinn(
            checkpoint_path=checkpoint_path,
            device=device,
        )

    field, ci_net, module, checkpoint_args, family = model_cache[chart_id]
    mach = float(point["Mach"])
    eta = float(point["eta"])
    alpha = float(point.get("alpha", alpha_from_eta(eta, mach)))

    classic_fields, ci_ref = load_classic_full_mode(alpha, mach)
    y = np.asarray(classic_fields["y"], dtype=float)
    p_ref = np.asarray(classic_fields["p"], dtype=np.complex128)
    rho_ref = np.asarray(classic_fields["rho"], dtype=np.complex128)
    u_ref = np.asarray(classic_fields["u"], dtype=np.complex128)
    v_ref = np.asarray(classic_fields["v"], dtype=np.complex128)
    q_ref = reference_q_from_fields(
        y,
        v_ref,
        alpha,
        mach,
        float(ci_ref),
    )
    gamma_ref = q_ref / np.where(
        np.abs(p_ref) > 1.0e-14,
        p_ref,
        np.nan + 1j * np.nan,
    )

    p_pred, q_pred, ci_pred = call_pinn_profiles(
        field=field,
        ci_net=ci_net,
        module=module,
        family=family,
        y=y,
        alpha=alpha,
        mach=mach,
        device=device,
    )

    mask = modal_mask(
        y,
        p_ref,
        y_max=y_max,
        amplitude_floor=amplitude_floor,
    )
    scale = phase_alignment(
        p_pred,
        p_ref,
        y,
        mask,
    )
    p_pred = scale * p_pred
    q_pred = scale * q_pred

    predicted = fields_from_pq(
        y,
        p_pred,
        q_pred,
        alpha,
        mach,
        ci_pred,
    )
    reference = {
        "p": p_ref,
        "q": q_ref,
        "rho": rho_ref,
        "u": u_ref,
        "v": v_ref,
        "gamma": gamma_ref,
    }

    result = point.to_dict()
    result.update(
        {
            "field_family": family,
            "checkpoint_ymax": float(
                checkpoint_args.get("ymax", float("nan"))
            ),
            "ci_ref": float(ci_ref),
            "ci_pred": float(ci_pred),
            "ci_abs_err": abs(float(ci_pred) - float(ci_ref)),
            "ci_rel_err": abs(float(ci_pred) - float(ci_ref))
            / max(abs(float(ci_ref)), 1.0e-12),
            "ci_rel_err_reg": abs(float(ci_pred) - float(ci_ref))
            / max(abs(float(ci_ref)), CI_SCALE_FLOOR),
            "omega_ref": alpha * float(ci_ref),
            "omega_pred": alpha * float(ci_pred),
            "omega_abs_err": abs(
                alpha * float(ci_pred) - alpha * float(ci_ref)
            ),
            "omega_rel_err": abs(
                alpha * float(ci_pred) - alpha * float(ci_ref)
            )
            / max(abs(alpha * float(ci_ref)), 1.0e-12),
            "omega_rel_err_reg": abs(
                alpha * float(ci_pred) - alpha * float(ci_ref)
            )
            / max(abs(alpha * float(ci_ref)), OMEGA_SCALE_FLOOR),
            "near_neutral": eta >= 0.92,
            "n_modal_mask": int(np.count_nonzero(mask)),
            "alignment_real": float(np.real(scale)),
            "alignment_imag": float(np.imag(scale)),
            "success": True,
            "error": "",
        }
    )

    for field_name in ("p", "q", "rho", "u", "v", "gamma"):
        result[f"{field_name}_rel"] = rel_l2(
            predicted[field_name],
            reference[field_name],
            y,
            mask,
        )
        if field_name != "gamma":
            result[f"{field_name}_overlap"] = overlap_complex(
                predicted[field_name],
                reference[field_name],
                y,
                mask,
            )

    return result


def shard_slice(
    frame: pd.DataFrame,
    task_index: int,
    chunk_size: int,
) -> pd.DataFrame:
    start = int(task_index) * int(chunk_size)
    stop = min(start + int(chunk_size), len(frame))
    if start >= len(frame):
        return frame.iloc[0:0].copy()
    return frame.iloc[start:stop].copy()


def command_validate_points(args: argparse.Namespace) -> None:
    plan = pd.read_csv(args.plan)
    shard = shard_slice(
        plan,
        args.task_index,
        args.chunk_size,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"shard_{args.task_index:05d}.csv"

    if shard.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        print("Empty shard:", output_path)
        return

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model_cache: dict[str, tuple[Any, ...]] = {}
    rows = []

    for _, point in shard.iterrows():
        print(
            f"[{point['point_id']}] chart={point['chart_id']} "
            f"M={float(point['Mach']):.7f} "
            f"eta={float(point['eta']):.7f}"
        )
        try:
            rows.append(
                evaluate_direct_point(
                    point,
                    model_cache=model_cache,
                    device=device,
                    y_max=args.y_max,
                    amplitude_floor=args.amplitude_floor,
                )
            )
        except Exception as error:
            row = point.to_dict()
            row.update(
                {
                    "success": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
            rows.append(row)
            print(row["error"])

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print("Wrote:", output_path)


def seam_metrics(
    first_fields: dict[str, np.ndarray],
    second_fields: dict[str, np.ndarray],
    y: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    result = {}
    for field_name in ("p", "q", "rho", "u", "v"):
        result[f"{field_name}_rel_ab"] = rel_l2(
            second_fields[field_name],
            first_fields[field_name],
            y,
            mask,
        )
        result[f"{field_name}_overlap_ab"] = overlap_complex(
            second_fields[field_name],
            first_fields[field_name],
            y,
            mask,
        )
    return result


def command_validate_seams(args: argparse.Namespace) -> None:
    plan = pd.read_csv(args.plan)
    shard = shard_slice(
        plan,
        args.task_index,
        args.chunk_size,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"shard_{args.task_index:05d}.csv"

    if shard.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        print("Empty shard:", output_path)
        return

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model_cache: dict[str, tuple[Any, ...]] = {}
    y = np.linspace(-float(args.y_max), float(args.y_max), int(args.n_y))
    rows = []

    for _, point in shard.iterrows():
        base = point.to_dict()
        try:
            evaluated = {}
            for label in ("a", "b"):
                chart_id = str(point[f"chart_{label}"])
                checkpoint = Path(str(point[f"checkpoint_{label}"]))
                if chart_id not in model_cache:
                    model_cache[chart_id] = evaluate_pinn(
                        checkpoint_path=checkpoint,
                        device=device,
                    )
                field, ci_net, module, _, family = model_cache[chart_id]
                p, q, ci = call_pinn_profiles(
                    field=field,
                    ci_net=ci_net,
                    module=module,
                    family=family,
                    y=y,
                    alpha=float(point["alpha"]),
                    mach=float(point["Mach"]),
                    device=device,
                )
                fields = fields_from_pq(
                    y,
                    p,
                    q,
                    float(point["alpha"]),
                    float(point["Mach"]),
                    ci,
                )
                evaluated[label] = {
                    "fields": fields,
                    "ci": float(ci),
                    "family": family,
                }

            mask = modal_mask(
                y,
                evaluated["a"]["fields"]["p"],
                y_max=args.y_max,
                amplitude_floor=args.amplitude_floor,
            )
            scale = phase_alignment(
                evaluated["b"]["fields"]["p"],
                evaluated["a"]["fields"]["p"],
                y,
                mask,
            )
            for field_name in evaluated["b"]["fields"]:
                evaluated["b"]["fields"][field_name] = (
                    scale * evaluated["b"]["fields"][field_name]
                )

            base.update(
                {
                    "family_a": evaluated["a"]["family"],
                    "family_b": evaluated["b"]["family"],
                    "ci_a": evaluated["a"]["ci"],
                    "ci_b": evaluated["b"]["ci"],
                    "ci_abs_diff_ab": abs(
                        evaluated["a"]["ci"] - evaluated["b"]["ci"]
                    ),
                    "ci_rel_diff_reg_ab": abs(
                        evaluated["a"]["ci"] - evaluated["b"]["ci"]
                    )
                    / max(
                        abs(evaluated["a"]["ci"]),
                        abs(evaluated["b"]["ci"]),
                        CI_SCALE_FLOOR,
                    ),
                    "alignment_real": float(np.real(scale)),
                    "alignment_imag": float(np.imag(scale)),
                    "success": True,
                    "error": "",
                }
            )
            base.update(
                seam_metrics(
                    evaluated["a"]["fields"],
                    evaluated["b"]["fields"],
                    y,
                    mask,
                )
            )
        except Exception as error:
            base.update(
                {
                    "success": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
        rows.append(base)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print("Wrote:", output_path)


def merge_shards(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("shard_*.csv"))
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frame["shard_file"] = str(path)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def outlier_flags(
    frame: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    result = frame.copy()

    def numeric(column: str) -> pd.Series:
        if column not in result:
            return pd.Series(np.nan, index=result.index)
        return pd.to_numeric(result[column], errors="coerce")

    result["outlier_spectral"] = (
        (numeric("ci_abs_err") > args.ci_abs_threshold)
        | (numeric("ci_rel_err_reg") > args.ci_rel_reg_threshold)
        | (numeric("omega_abs_err") > args.omega_abs_threshold)
    )
    result["outlier_modal"] = (
        (numeric("p_overlap") < args.p_overlap_threshold)
        | (numeric("q_overlap") < args.q_overlap_threshold)
        | (numeric("p_rel") > args.p_rel_threshold)
    )
    result["outlier_any"] = (
        result["outlier_spectral"]
        | result["outlier_modal"]
        | ~as_bool(result.get("success", pd.Series(True, index=result.index)))
    )
    return result


def command_merge(args: argparse.Namespace) -> None:
    validation_root = Path(args.validation_root)
    merged_dir = validation_root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    offgrid = merge_shards(validation_root / "shards" / "offgrid")
    targeted = merge_shards(validation_root / "shards" / "targeted")
    seams = merge_shards(validation_root / "shards" / "seams")

    if not offgrid.empty:
        offgrid = outlier_flags(offgrid, args)
        offgrid.to_csv(
            merged_dir / "offgrid_384_direct_PINN_results.csv",
            index=False,
        )
        write_json(
            merged_dir / "offgrid_384_metrics.json",
            json_safe(
                grouped_metric_report(
                    offgrid.loc[as_bool(offgrid["success"])],
                    SPECTRAL_COLUMNS + MODAL_COLUMNS,
                    group_columns=("near_neutral", "sample_group", "chart_id"),
                )
            ),
        )

    if not targeted.empty:
        targeted = outlier_flags(targeted, args)
        targeted.to_csv(
            merged_dir / "targeted_scans_direct_PINN_results.csv",
            index=False,
        )
        write_json(
            merged_dir / "targeted_scans_metrics.json",
            json_safe(
                grouped_metric_report(
                    targeted.loc[as_bool(targeted["success"])],
                    SPECTRAL_COLUMNS + MODAL_COLUMNS,
                    group_columns=("sample_group", "chart_id"),
                )
            ),
        )

    if not seams.empty:
        seams["seam_outlier"] = (
            pd.to_numeric(
                seams.get("p_overlap_ab"),
                errors="coerce",
            )
            < args.seam_p_overlap_threshold
        ) | (
            pd.to_numeric(
                seams.get("ci_abs_diff_ab"),
                errors="coerce",
            )
            > args.seam_ci_abs_threshold
        )
        seams.to_csv(
            merged_dir / "seam_dense_results.csv",
            index=False,
        )
        write_json(
            merged_dir / "seam_dense_metrics.json",
            json_safe(
                grouped_metric_report(
                    seams.loc[as_bool(seams["success"])],
                    [
                        "ci_abs_diff_ab",
                        "ci_rel_diff_reg_ab",
                        "p_rel_ab",
                        "q_rel_ab",
                        "p_overlap_ab",
                        "q_overlap_ab",
                    ],
                    group_columns=("seam_id",),
                )
            ),
        )

    candidates = []
    for origin, frame in (
        ("offgrid", offgrid),
        ("targeted", targeted),
    ):
        if frame.empty:
            continue
        flagged = frame.loc[as_bool(frame["outlier_any"])].copy()
        flagged["outlier_origin"] = origin
        candidates.append(flagged)

    aggregate_path = (
        validation_root
        / "aggregate/direct_diagnostics_all_chart_evaluations.csv"
    )
    if aggregate_path.is_file():
        aggregate = pd.read_csv(aggregate_path)
        aggregate = outlier_flags(aggregate, args)
        flagged = aggregate.loc[as_bool(aggregate["outlier_any"])].copy()
        flagged["point_id"] = [
            f"DIAG_{chart}_{index:04d}"
            for index, chart in enumerate(flagged["chart_id"].astype(str))
        ]
        flagged["checkpoint"] = flagged["output_dir"].map(
            lambda value: str(Path(str(value)) / "model_state.pt")
        ) if "output_dir" in flagged else ""
        flagged["sample_group"] = "existing_diagnostic"
        flagged["outlier_origin"] = "existing_diagnostic"
        candidates.append(flagged)

    if candidates:
        outliers = pd.concat(candidates, ignore_index=True, sort=False)
        outliers["outlier_severity"] = np.nanmax(
            np.column_stack(
                [
                    pd.to_numeric(
                        outliers.get("ci_rel_err_reg"),
                        errors="coerce",
                    ).fillna(0.0),
                    pd.to_numeric(
                        outliers.get("omega_abs_err"),
                        errors="coerce",
                    ).fillna(0.0)
                    / max(args.omega_abs_threshold, 1.0e-12),
                    pd.to_numeric(
                        outliers.get("p_rel"),
                        errors="coerce",
                    ).fillna(0.0),
                    (
                        1.0
                        - pd.to_numeric(
                            outliers.get("p_overlap"),
                            errors="coerce",
                        ).fillna(1.0)
                    ),
                ]
            ),
            axis=1,
        )
        outliers = (
            outliers.sort_values(
                "outlier_severity",
                ascending=False,
            )
            .drop_duplicates(["Mach", "eta"])
            .reset_index(drop=True)
        )

        training_plan = normalize_plan(Path(args.training_plan))
        routed_rows = []
        for _, row in outliers.iterrows():
            chart = route_chart(
                training_plan,
                float(row["Mach"]),
                float(row["eta"]),
                preferred=str(row.get("chart_id", "")),
            )
            payload = row.to_dict()
            payload.update(
                {
                    "chart_id": str(chart["chart_id"]),
                    "checkpoint": str(chart["checkpoint"]),
                    "alpha": alpha_from_eta(
                        float(row["eta"]),
                        float(row["Mach"]),
                    ),
                }
            )
            routed_rows.append(payload)
        outlier_plan = pd.DataFrame(routed_rows)
    else:
        outlier_plan = pd.DataFrame()

    outlier_plan.to_csv(
        merged_dir / "outlier_recheck_plan.csv",
        index=False,
    )

    report = {
        "n_offgrid_rows": int(len(offgrid)),
        "n_targeted_rows": int(len(targeted)),
        "n_seam_rows": int(len(seams)),
        "n_offgrid_outliers": (
            int(as_bool(offgrid["outlier_any"]).sum())
            if not offgrid.empty
            else 0
        ),
        "n_targeted_outliers": (
            int(as_bool(targeted["outlier_any"]).sum())
            if not targeted.empty
            else 0
        ),
        "n_seam_outliers": (
            int(as_bool(seams["seam_outlier"]).sum())
            if not seams.empty
            else 0
        ),
        "n_unique_outlier_rechecks": int(len(outlier_plan)),
        "thresholds": {
            "ci_abs": args.ci_abs_threshold,
            "ci_rel_reg": args.ci_rel_reg_threshold,
            "omega_abs": args.omega_abs_threshold,
            "p_overlap": args.p_overlap_threshold,
            "q_overlap": args.q_overlap_threshold,
            "p_rel": args.p_rel_threshold,
            "seam_p_overlap": args.seam_p_overlap_threshold,
            "seam_ci_abs": args.seam_ci_abs_threshold,
        },
    }
    write_json(merged_dir / "merge_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def robust_recheck(
    mach: float,
    eta: float,
    *,
    force_cross_check: bool,
    primary_n_scan: int,
    secondary_n_scan: int,
) -> dict[str, Any]:
    alpha = alpha_from_eta(eta, mach)
    solver = RobustSubsonicShootingSolver(
        alpha=alpha,
        Mach=mach,
    )
    result = solver.solve(
        force_cross_check=force_cross_check,
        primary_n_scan=primary_n_scan,
        secondary_n_scan=secondary_n_scan,
    )
    return {
        "Mach": mach,
        "eta": eta,
        "alpha": alpha,
        "shoot_ci_recheck": float(result.ci),
        "shoot_omega_recheck": float(result.omega_i),
        "shoot_success_recheck": bool(result.success),
        "shoot_source_recheck": str(result.source),
        "primary_ci": float(result.primary_ci),
        "primary_omega_i": float(result.primary_omega_i),
        "primary_success": bool(result.primary_success),
        "primary_mismatch": float(result.primary_mismatch),
        "secondary_ci": (
            None if result.secondary_ci is None
            else float(result.secondary_ci)
        ),
        "secondary_omega_i": (
            None if result.secondary_omega_i is None
            else float(result.secondary_omega_i)
        ),
        "secondary_success": (
            None if result.secondary_success is None
            else bool(result.secondary_success)
        ),
        "secondary_stage1_mismatch": (
            None
            if result.secondary_stage1_mismatch is None
            else float(result.secondary_stage1_mismatch)
        ),
        "secondary_stage2_mismatch": (
            None
            if result.secondary_stage2_mismatch is None
            else float(result.secondary_stage2_mismatch)
        ),
        "primary_secondary_ci_abs_diff": (
            None if result.ci_abs_diff is None
            else float(result.ci_abs_diff)
        ),
        "primary_secondary_omega_abs_diff": (
            None if result.omega_abs_diff is None
            else float(result.omega_abs_diff)
        ),
    }


def command_recheck_outliers(args: argparse.Namespace) -> None:
    plan = pd.read_csv(args.plan)
    shard = shard_slice(
        plan,
        args.task_index,
        args.chunk_size,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"shard_{args.task_index:05d}.csv"

    if shard.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        print("Empty shard:", output_path)
        return

    training_plan = normalize_plan(Path(args.training_plan))
    device = torch.device("cpu")
    model_cache: dict[str, tuple[Any, ...]] = {}
    rows = []

    for _, target in shard.iterrows():
        target_mach = float(target["Mach"])
        target_eta = float(target["eta"])
        neighborhood = [
            ("eta_minus", max(0.02, target_eta - args.eta_delta)),
            ("target", target_eta),
            ("eta_plus", min(0.98, target_eta + args.eta_delta)),
        ]

        for neighborhood_label, eta in neighborhood:
            try:
                chart = route_chart(
                    training_plan,
                    target_mach,
                    eta,
                    preferred=str(target.get("chart_id", "")),
                )
                point = pd.Series(
                    {
                        **target.to_dict(),
                        "point_id": (
                            f"{target['point_id']}__{neighborhood_label}"
                        ),
                        "neighborhood_label": neighborhood_label,
                        "is_target": neighborhood_label == "target",
                        "Mach": target_mach,
                        "eta": eta,
                        "alpha": alpha_from_eta(eta, target_mach),
                        "chart_id": str(chart["chart_id"]),
                        "checkpoint": str(chart["checkpoint"]),
                    }
                )

                direct = evaluate_direct_point(
                    point,
                    model_cache=model_cache,
                    device=device,
                    y_max=args.y_max,
                    amplitude_floor=args.amplitude_floor,
                )
                recheck = robust_recheck(
                    target_mach,
                    eta,
                    force_cross_check=True,
                    primary_n_scan=args.primary_n_scan,
                    secondary_n_scan=args.secondary_n_scan,
                )
                direct.update(recheck)
                direct["ci_pred_abs_err_recheck"] = abs(
                    float(direct["ci_pred"])
                    - float(recheck["shoot_ci_recheck"])
                )
                direct["ci_pred_rel_err_reg_recheck"] = (
                    direct["ci_pred_abs_err_recheck"]
                    / max(
                        abs(float(recheck["shoot_ci_recheck"])),
                        CI_SCALE_FLOOR,
                    )
                )
                rows.append(direct)
            except Exception as error:
                row = target.to_dict()
                row.update(
                    {
                        "neighborhood_label": neighborhood_label,
                        "Mach": target_mach,
                        "eta": eta,
                        "success": False,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
                rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print("Wrote:", output_path)


def plot_parity(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    data = frame[[x_column, y_column]].apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()
    if data.empty:
        return
    lower = float(data.min().min())
    upper = float(data.max().max())
    fig, axis = plt.subplots(figsize=(6.3, 5.7))
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axis.scatter(data[x_column], data[y_column], s=18, alpha=0.7)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_map(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    path: Path,
    *,
    log: bool = True,
) -> None:
    columns = ["Mach", "eta", metric]
    data = frame[columns].copy()
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()
    if data.empty:
        return

    values = data[metric].to_numpy()
    if log:
        values = np.log10(np.clip(values, 1.0e-12, None))
        label = f"log10({metric})"
    else:
        label = metric

    fig, axis = plt.subplots(figsize=(7.2, 6.1))
    scatter = axis.scatter(
        data["Mach"],
        data["eta"],
        c=values,
        s=26,
        edgecolors="black",
        linewidths=0.2,
    )
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label(label)
    axis.set_xlabel("Mach M")
    axis.set_ylabel(r"$\eta$")
    axis.set_title(title)
    axis.set_xlim(0.015, 0.985)
    axis.set_ylim(0.015, 0.985)
    axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def git_text(arguments: list[str]) -> str:
    try:
        return subprocess.check_output(
            arguments,
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as error:
        return f"UNAVAILABLE: {type(error).__name__}: {error}"


def copy_validation_file(
    source: Path,
    destination_root: Path,
    relative: Path,
) -> None:
    if not source.is_file():
        return
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_finalize(args: argparse.Namespace) -> None:
    validation_root = Path(args.validation_root)
    final_dir = validation_root / "final"
    figures_dir = final_dir / "figures"
    tables_dir = final_dir / "tables"
    manifests_dir = final_dir / "manifests"
    for directory in (figures_dir, tables_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)

    aggregate_all_path = (
        validation_root
        / "aggregate/direct_diagnostics_all_chart_evaluations.csv"
    )
    aggregate_unique_path = (
        validation_root
        / "aggregate/direct_diagnostics_unique_canonical.csv"
    )
    offgrid_path = (
        validation_root
        / "merged/offgrid_384_direct_PINN_results.csv"
    )
    targeted_path = (
        validation_root
        / "merged/targeted_scans_direct_PINN_results.csv"
    )
    seam_path = validation_root / "merged/seam_dense_results.csv"
    recheck = merge_shards(validation_root / "shards" / "outlier_recheck")

    required = [
        aggregate_all_path,
        aggregate_unique_path,
        offgrid_path,
        targeted_path,
        seam_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot finalize; missing required files: {missing}"
        )

    aggregate_all = pd.read_csv(aggregate_all_path)
    aggregate_unique = pd.read_csv(aggregate_unique_path)
    offgrid = pd.read_csv(offgrid_path)
    targeted = pd.read_csv(targeted_path)
    seams = pd.read_csv(seam_path)

    offgrid_valid = offgrid.loc[as_bool(offgrid["success"])].copy()
    targeted_valid = targeted.loc[as_bool(targeted["success"])].copy()
    seam_valid = seams.loc[as_bool(seams["success"])].copy()

    direct_combined = pd.concat(
        [
            aggregate_unique.assign(validation_origin="existing_unique"),
            offgrid_valid.assign(validation_origin="offgrid_384"),
            targeted_valid.assign(validation_origin="targeted"),
        ],
        ignore_index=True,
        sort=False,
    )
    direct_combined.to_csv(
        tables_dir / "global_direct_PINN_validation_pointwise.csv",
        index=False,
    )
    seams.to_csv(
        tables_dir / "global_atlas_seam_validation.csv",
        index=False,
    )
    if not recheck.empty:
        recheck.to_csv(
            tables_dir / "global_outlier_classical_rechecks.csv",
            index=False,
        )

    report = {
        "release_name": "joint_subsonic_PINN_global_validation_v1",
        "n_charts": int(aggregate_all["chart_id"].nunique()),
        "existing_diagnostics": grouped_metric_report(
            aggregate_unique,
            SPECTRAL_COLUMNS
            + [
                "p_rel",
                "q_rel",
                "rho_rel",
                "u_rel",
                "v_rel",
                "gamma_rel",
            ],
            group_columns=("near_neutral", "field_family"),
        ),
        "offgrid_384": grouped_metric_report(
            offgrid_valid,
            SPECTRAL_COLUMNS + MODAL_COLUMNS,
            group_columns=("near_neutral", "sample_group", "chart_id"),
        ),
        "targeted_scans": grouped_metric_report(
            targeted_valid,
            SPECTRAL_COLUMNS + MODAL_COLUMNS,
            group_columns=("sample_group",),
        ),
        "seams": grouped_metric_report(
            seam_valid,
            [
                "ci_abs_diff_ab",
                "ci_rel_diff_reg_ab",
                "p_rel_ab",
                "q_rel_ab",
                "p_overlap_ab",
                "q_overlap_ab",
            ],
            group_columns=(),
        ),
        "outlier_rechecks": {
            "n_rows": int(len(recheck)),
            "n_target_points": (
                int(as_bool(recheck["is_target"]).sum())
                if not recheck.empty and "is_target" in recheck
                else 0
            ),
            "metrics": (
                {
                    metric: metric_stats(recheck, metric)
                    for metric in [
                        "ci_pred_abs_err_recheck",
                        "ci_pred_rel_err_reg_recheck",
                        "primary_secondary_ci_abs_diff",
                        "primary_secondary_omega_abs_diff",
                    ]
                }
                if not recheck.empty
                else {}
            ),
        },
    }
    write_json(
        final_dir / "global_validation_report.json",
        json_safe(report),
    )

    summary_rows = []
    for origin, frame in (
        ("existing_unique", aggregate_unique),
        ("offgrid_384", offgrid_valid),
        ("targeted", targeted_valid),
        ("seams", seam_valid),
    ):
        metrics = (
            SPECTRAL_COLUMNS + MODAL_COLUMNS
            if origin != "seams"
            else [
                "ci_abs_diff_ab",
                "ci_rel_diff_reg_ab",
                "p_overlap_ab",
                "q_overlap_ab",
            ]
        )
        for metric in metrics:
            stats = metric_stats(frame, metric)
            summary_rows.append(
                {
                    "origin": origin,
                    "metric": metric,
                    **stats,
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        tables_dir / "global_validation_summary.csv",
        index=False,
    )

    plot_parity(
        direct_combined,
        "ci_ref",
        "ci_pred",
        r"Shooting $c_i$",
        r"PINN $c_i$",
        "Direct spectral PINN validation",
        figures_dir / "ci_shooting_vs_PINN_parity",
    )
    plot_parity(
        direct_combined,
        "omega_ref",
        "omega_pred",
        r"Shooting $\omega_i$",
        r"PINN $\omega_i$",
        "Direct growth-rate PINN validation",
        figures_dir / "omega_shooting_vs_PINN_parity",
    )
    plot_metric_map(
        direct_combined,
        "ci_abs_err",
        "Direct PINN absolute spectral error",
        figures_dir / "map_ci_abs_err",
        log=True,
    )
    plot_metric_map(
        direct_combined,
        "p_overlap",
        "Direct PINN pressure overlap",
        figures_dir / "map_p_overlap",
        log=False,
    )

    if "p_overlap_ab" in seam_valid:
        fig, axis = plt.subplots(figsize=(6.5, 4.8))
        values = finite_series(seam_valid, "p_overlap_ab")
        if not values.empty:
            axis.hist(values, bins=40)
            axis.set_xlabel(r"Chart-to-chart pressure overlap $\mathcal O_p$")
            axis.set_ylabel("Count")
            axis.set_title("Atlas seam continuity")
            axis.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(
                figures_dir / "seam_pressure_overlap_hist.pdf",
                bbox_inches="tight",
            )
            fig.savefig(
                figures_dir / "seam_pressure_overlap_hist.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)

    training_plan = normalize_plan(Path(args.training_plan))
    release_dir = validation_root / "release_v1"
    if release_dir.exists():
        if args.clean_release:
            shutil.rmtree(release_dir)
        else:
            raise FileExistsError(
                f"{release_dir} already exists. "
                "Pass --clean-release to replace it."
            )

    for relative in (
        Path("models"),
        Path("tables"),
        Path("figures"),
        Path("validation"),
        Path("manifests"),
        Path("scripts"),
    ):
        (release_dir / relative).mkdir(parents=True, exist_ok=True)

    for _, chart in training_plan.iterrows():
        chart_id = str(chart["chart_id"])
        source_dir = Path(str(chart["output_dir"]))
        destination = release_dir / "models" / chart_id
        destination.mkdir(parents=True, exist_ok=True)
        for filename in (
            "model_state.pt",
            "joint_training_metadata.json",
            "ci_anchor_points.csv",
            "diagnostics_summary.csv",
        ):
            source = source_dir / filename
            if source.is_file():
                shutil.copy2(source, destination / filename)

    copy_validation_file(
        final_dir / "global_validation_report.json",
        release_dir,
        Path("validation/global_validation_report.json"),
    )
    for source in sorted(tables_dir.glob("*")):
        copy_validation_file(
            source,
            release_dir,
            Path("tables") / source.name,
        )
    for source in sorted(figures_dir.glob("*")):
        copy_validation_file(
            source,
            release_dir,
            Path("figures") / source.name,
        )

    for source in (
        Path(__file__),
        ROOT / "slurm/joint_pinn_validation_array.slurm",
        ROOT / "code/src/scripts/gep/selection/audit_mid_joint_pinn_full_gep.py",
    ):
        if source.is_file():
            shutil.copy2(
                source,
                release_dir / "scripts" / source.name,
            )

    provenance = {
        "date_utc": subprocess.check_output(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
            text=True,
        ).strip(),
        "git_commit": git_text(["git", "rev-parse", "HEAD"]),
        "git_status": git_text(["git", "status", "--short"]),
        "training_plan": str(args.training_plan),
        "validation_root": str(validation_root),
        "model_count": int(training_plan["chart_id"].nunique()),
    }
    write_json(
        release_dir / "manifests/provenance.json",
        provenance,
    )
    training_plan.to_csv(
        release_dir / "manifests/chart_catalog.tsv",
        sep="\t",
        index=False,
    )

    checksum_lines = []
    for path in sorted(release_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "sha256_all_files.txt":
            continue
        checksum_lines.append(
            f"{sha256_file(path)}  {path.relative_to(release_dir)}"
        )
    (
        release_dir / "manifests/sha256_all_files.txt"
    ).write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print(json.dumps(json_safe(report), indent=2, sort_keys=True))
    print("Final report:", final_dir / "global_validation_report.json")
    print("Frozen release:", release_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--training-plan",
        default=DEFAULT_TRAIN_PLAN,
    )
    common.add_argument(
        "--validation-root",
        default=DEFAULT_VALIDATION_ROOT,
    )

    aggregate = subparsers.add_parser(
        "aggregate",
        parents=[common],
    )
    aggregate.set_defaults(function=command_aggregate)

    plans = subparsers.add_parser(
        "build-plans",
        parents=[common],
    )
    plans.add_argument("--historical-points", default=None)
    plans.set_defaults(function=command_build_plans)

    points = subparsers.add_parser("validate-points")
    points.add_argument("--plan", required=True)
    points.add_argument("--output-dir", required=True)
    points.add_argument("--task-index", type=int, required=True)
    points.add_argument("--chunk-size", type=int, default=8)
    points.add_argument("--device", default="cpu")
    points.add_argument("--y-max", type=float, default=DEFAULT_YMAX)
    points.add_argument(
        "--amplitude-floor",
        type=float,
        default=DEFAULT_AMP_FLOOR,
    )
    points.set_defaults(function=command_validate_points)

    seams = subparsers.add_parser("validate-seams")
    seams.add_argument("--plan", required=True)
    seams.add_argument("--output-dir", required=True)
    seams.add_argument("--task-index", type=int, required=True)
    seams.add_argument("--chunk-size", type=int, default=40)
    seams.add_argument("--device", default="cpu")
    seams.add_argument("--y-max", type=float, default=DEFAULT_YMAX)
    seams.add_argument("--n-y", type=int, default=DEFAULT_NY)
    seams.add_argument(
        "--amplitude-floor",
        type=float,
        default=DEFAULT_AMP_FLOOR,
    )
    seams.set_defaults(function=command_validate_seams)

    merge = subparsers.add_parser(
        "merge",
        parents=[common],
    )
    merge.add_argument("--ci-abs-threshold", type=float, default=1.0e-2)
    merge.add_argument(
        "--ci-rel-reg-threshold",
        type=float,
        default=1.0e-1,
    )
    merge.add_argument(
        "--omega-abs-threshold",
        type=float,
        default=2.0e-3,
    )
    merge.add_argument("--p-overlap-threshold", type=float, default=0.90)
    merge.add_argument("--q-overlap-threshold", type=float, default=0.80)
    merge.add_argument("--p-rel-threshold", type=float, default=0.35)
    merge.add_argument(
        "--seam-p-overlap-threshold",
        type=float,
        default=0.98,
    )
    merge.add_argument(
        "--seam-ci-abs-threshold",
        type=float,
        default=5.0e-3,
    )
    merge.set_defaults(function=command_merge)

    recheck = subparsers.add_parser(
        "recheck-outliers",
        parents=[common],
    )
    recheck.add_argument("--plan", required=True)
    recheck.add_argument("--output-dir", required=True)
    recheck.add_argument("--task-index", type=int, required=True)
    recheck.add_argument("--chunk-size", type=int, default=1)
    recheck.add_argument("--eta-delta", type=float, default=2.5e-3)
    recheck.add_argument("--primary-n-scan", type=int, default=241)
    recheck.add_argument("--secondary-n-scan", type=int, default=121)
    recheck.add_argument("--y-max", type=float, default=DEFAULT_YMAX)
    recheck.add_argument(
        "--amplitude-floor",
        type=float,
        default=DEFAULT_AMP_FLOOR,
    )
    recheck.set_defaults(function=command_recheck_outliers)

    finalize = subparsers.add_parser(
        "finalize",
        parents=[common],
    )
    finalize.add_argument("--clean-release", action="store_true")
    finalize.set_defaults(function=command_finalize)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
