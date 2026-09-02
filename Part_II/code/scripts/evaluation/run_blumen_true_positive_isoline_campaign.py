#!/usr/bin/env python3
"""Compute and render true classical isolines at positive Blumen levels.

Each digitized Blumen point keeps its Mach number.  For every curve, points
are processed sequentially in the original digitization order.  The worker
solves the production Riccati matching problem repeatedly while searching in
alpha for ``ci_classical(Mach, alpha) = ci_target``.  Independent Blumen
curves may run concurrently, but a curve is never split across workers so
that the previous accepted eigenvalue remains available as a continuation
seed.

The command is restartable: existing point results are reused unless
``--force`` is supplied.  Once all requested points have been processed, the
publication CSV files and 1x2 PDF/PNG figure are generated automatically.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

from scripts.evaluation.blumen_isoline_common import (
    POSITIVE_LEVELS,
    attach_original_digitization_order,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_BLUMEN = REPO_ROOT / "assets/classic_supersonic/csv/blumen_validation/supersonic/table_ci_datasets.csv"
DEFAULT_POINTWISE = (
    REPO_ROOT / 'article/tables/table_supersonic_blumen_positive_pointwise_values.csv'
)
DEFAULT_REFERENCE = (
    REPO_ROOT / 'assets/classic_supersonic/csv/modal_reconstruction/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/table_classical_supersonic_final_reference.csv'
)
DEFAULT_CONFIG = (
    REPO_ROOT / 'code/configs/legacy/dense_supersonic_campaign_config.json'
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "classic_supersonic/reproducibility/results"
    / "blumen_true_positive_isolines_local_v2"
)
DEFAULT_ARTICLE_ROOT = REPO_ROOT / "assets/classic_supersonic/article"
POINT_WORKER = (
    REPO_ROOT / 'code/scripts/evaluation/compute_blumen_true_positive_isoline_point.py'
)
ASSET_BUILDER = (
    REPO_ROOT / 'code/scripts/data_preparation/prepare_build_blumen_true_positive_isoline_assets.py'
)
OK_STATUSES = {"converged_center", "converged_scan_point", "converged_root"}


def load_targets(path: Path, raw_blumen_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "blumen_row_id",
        "source_row_id",
        "curve_id",
        "curve_key",
        "curve_label",
        "Mach",
        "alpha",
        "blumen_ci",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing target columns: {missing}")
    for column in (
        "blumen_row_id",
        "source_row_id",
        "curve_id",
        "Mach",
        "alpha",
        "blumen_ci",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        np.isfinite(frame["Mach"])
        & np.isfinite(frame["alpha"])
        & np.isfinite(frame["blumen_ci"])
        & (frame["blumen_ci"] > 0.0)
    ].copy()
    level_mask = np.zeros(len(frame), dtype=bool)
    for level in POSITIVE_LEVELS:
        level_mask |= np.isclose(frame["blumen_ci"], level, rtol=0.0, atol=1e-12)
    frame = frame.loc[level_mask].copy()
    frame = frame.sort_values("source_row_id", kind="stable").reset_index(drop=True)
    frame["true_isoline_task_index"] = np.arange(len(frame), dtype=int)
    if len(frame) != 117:
        raise ValueError(f"Expected 117 positive targets, found {len(frame)}.")
    return attach_original_digitization_order(frame, raw_blumen_csv)


def read_point_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    frame = pd.read_csv(path)
    if len(frame) != 1:
        return None
    return frame.iloc[0].to_dict()


def accepted_root(result: dict[str, Any] | None) -> dict[str, float] | None:
    if result is None or str(result.get("status")) not in OK_STATUSES:
        return None
    cr = pd.to_numeric(pd.Series([result.get("classical_cr")]), errors="coerce").iloc[0]
    ci = pd.to_numeric(pd.Series([result.get("classical_ci")]), errors="coerce").iloc[0]
    if not (np.isfinite(cr) and np.isfinite(ci) and ci > 0.0):
        return None
    return {"cr": float(cr), "ci": float(ci)}


def write_crash_result(
    *,
    target: pd.Series,
    result_path: Path,
    message: str,
) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    row = target.to_dict()
    row.update(
        {
            "status": "worker_process_failure",
            "alpha_blumen": float(target["alpha"]),
            "target_ci": float(target["blumen_ci"]),
            "alpha_classical": np.nan,
            "delta_alpha": np.nan,
            "abs_delta_alpha": np.nan,
            "classical_cr": np.nan,
            "classical_ci": np.nan,
            "delta_ci": np.nan,
            "residual_norm": np.nan,
            "n_solver_calls": 0,
            "bracket_alpha_left": np.nan,
            "bracket_alpha_right": np.nan,
            "message": message,
        }
    )
    pd.DataFrame([row]).to_csv(result_path, index=False)


def process_curve(
    *,
    curve: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    curve_key = str(curve["curve_key"].iloc[0])
    previous: dict[str, float] | None = None
    n_converged = 0
    logs_root = args.output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    for _, target in curve.sort_values(
        "digitization_order", kind="stable"
    ).iterrows():
        task_index = int(target["true_isoline_task_index"])
        point_root = args.output_root / f"point_{task_index:03d}"
        result_path = point_root / "result.csv"

        existing = None if args.force else read_point_result(result_path)
        if existing is not None:
            continuation = accepted_root(existing)
            if continuation is not None:
                previous = continuation
                n_converged += 1
            continue

        command = [
            sys.executable,
            str(POINT_WORKER),
            "--pointwise-csv",
            str(args.pointwise_csv),
            "--reference-csv",
            str(args.reference_csv),
            "--config",
            str(args.config),
            "--task-index",
            str(task_index),
            "--output-root",
            str(args.output_root),
            "--alpha-min",
            str(args.alpha_min),
            "--alpha-max",
            str(args.alpha_max),
            "--scan-step",
            str(args.scan_step),
            "--max-halfwidth",
            str(args.max_halfwidth),
            "--alpha-tolerance",
            str(args.alpha_tolerance),
            "--ci-tolerance",
            str(args.ci_tolerance),
            "--max-refine-iterations",
            str(args.max_refine_iterations),
        ]
        if previous is not None:
            command.extend(
                [
                    f"--preferred-cr={previous['cr']:.17g}",
                    f"--preferred-ci={previous['ci']:.17g}",
                ]
            )

        environment = os.environ.copy()
        environment.setdefault("MPLCONFIGDIR", "/tmp/kh_mpl")
        environment.setdefault("XDG_CACHE_HOME", "/tmp/kh_cache")
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (logs_root / f"point_{task_index:03d}.out").write_text(
            process.stdout, encoding="utf-8"
        )
        (logs_root / f"point_{task_index:03d}.err").write_text(
            process.stderr, encoding="utf-8"
        )
        if process.returncode != 0 or not result_path.is_file():
            write_crash_result(
                target=target,
                result_path=result_path,
                message=(
                    f"Worker return code {process.returncode}. "
                    f"See {logs_root / f'point_{task_index:03d}.err'}."
                ),
            )

        result = read_point_result(result_path)
        continuation = accepted_root(result)
        if continuation is not None:
            previous = continuation
            n_converged += 1

    return {
        "curve_key": curve_key,
        "n_points": int(len(curve)),
        "n_converged": n_converged,
    }


def build_assets(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(ASSET_BUILDER),
        "--pointwise-csv",
        str(args.pointwise_csv),
        "--raw-blumen-csv",
        str(args.raw_blumen_csv),
        "--result-root",
        str(args.output_root),
        "--article-root",
        str(args.article_root),
        "--ci-tolerance",
        str(args.ci_tolerance),
        "--residual-tolerance",
        str(args.residual_tolerance),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-blumen-csv", type=Path, default=DEFAULT_RAW_BLUMEN)
    parser.add_argument("--pointwise-csv", type=Path, default=DEFAULT_POINTWISE)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--article-root", type=Path, default=DEFAULT_ARTICLE_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--alpha-min", type=float, default=1e-6)
    parser.add_argument("--alpha-max", type=float, default=0.55)
    parser.add_argument("--scan-step", type=float, default=0.0025)
    parser.add_argument("--max-halfwidth", type=float, default=0.20)
    parser.add_argument("--alpha-tolerance", type=float, default=1e-6)
    parser.add_argument("--ci-tolerance", type=float, default=1e-6)
    parser.add_argument("--residual-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-refine-iterations", type=int, default=32)
    parser.add_argument(
        "--task-indices",
        default=None,
        help="Optional comma-separated internal indices for smoke tests.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-build-assets", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.raw_blumen_csv = args.raw_blumen_csv.resolve()
    args.pointwise_csv = args.pointwise_csv.resolve()
    args.reference_csv = args.reference_csv.resolve()
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()
    args.article_root = args.article_root.resolve()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("--workers must be between 1 and 4.")

    targets = load_targets(args.pointwise_csv, args.raw_blumen_csv)
    full_campaign = args.task_indices is None
    if args.task_indices:
        selected = {
            int(value.strip())
            for value in args.task_indices.split(",")
            if value.strip()
        }
        targets = targets.loc[
            targets["true_isoline_task_index"].isin(selected)
        ].copy()
        if len(targets) != len(selected):
            raise ValueError("At least one requested task index is invalid.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    curves = [
        group.copy()
        for _, group in targets.groupby("curve_key", sort=False)
    ]
    print("=== TRUE POSITIVE BLUMEN ISOLINE CAMPAIGN ===", flush=True)
    print(f"Targets : {len(targets)}", flush=True)
    print(f"Curves  : {len(curves)}", flush=True)
    print(f"Workers : {min(args.workers, len(curves))}", flush=True)

    reports: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(curves))) as executor:
        futures = [
            executor.submit(process_curve, curve=curve, args=args)
            for curve in curves
        ]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(
                f"[curve] {report['curve_key']}: "
                f"{report['n_converged']}/{report['n_points']}",
                flush=True,
            )

    if full_campaign and not args.no_build_assets:
        build_assets(args)
    elif not args.no_build_assets:
        print("[skip assets] A task subset cannot produce the final 117-point figure.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
