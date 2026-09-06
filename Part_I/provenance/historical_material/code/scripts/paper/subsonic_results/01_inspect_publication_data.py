#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


FILES = [
    # Single-case supervision budget
    Path(
        "model_saved/kh_subsonic_fixed_mach_M05_ci_supervision_budget/"
        "ci_supervision_budget_summary.csv"
    ),

    # Atlas-wide metrics
    Path(
        "assets/pinn_subsonic/joint_ci_mode_final_assets_v1/tables/"
        "Table_global_PINN_GEP_metrics.csv"
    ),
    Path(
        "assets/pinn_subsonic/joint_ci_mode_global_validation_v1/merged/"
        "seam_dense_results.csv"
    ),
    Path(
        "assets/pinn_subsonic/joint_ci_mode_global_validation_v1/merged/"
        "seam_dense_metrics.json"
    ),
    Path(
        "assets/pinn_subsonic/joint_ci_mode_global_validation_v1/merged/"
        "offgrid_384_metrics.json"
    ),

    # Modal metrics
    Path(
        "assets/pinn_subsonic/modes/"
        "mode_reconstruction_metrics.csv"
    ),

    # Near-neutral data
    Path(
        "assets/pinn_subsonic/joint_ci_mode_full_gep_atlas_v2/"
        "near_neutral_points.csv"
    ),
    Path(
        "assets/pinn_subsonic/joint_ci_mode_full_gep_atlas_v2/"
        "near_neutral_refinement/near_neutral_all_54.csv"
    ),
    Path(
        "assets/pinn_subsonic/local_atlas_v1/"
        "gep_core_atlas_ci_seeded_v2/near_neutral_ci_omega_audit.csv"
    ),
    Path(
        "assets/pinn_subsonic/local_atlas_v1/"
        "audit_HM2_neutral_edge_branch_M097/summary_branch_audit.csv"
    ),
]

OUT = Path(
    "assets/pinn_subsonic/paper_results_v1/data/"
    "publication_data_inspection.txt"
)
OUT.parent.mkdir(parents=True, exist_ok=True)


def describe_csv(path: Path) -> list[str]:
    lines: list[str] = []

    df = pd.read_csv(path)

    lines.append(f"shape: {df.shape}")
    lines.append(f"columns: {list(df.columns)}")
    lines.append("")
    lines.append("head:")
    lines.append(df.head(8).to_string(index=False))

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        lines.append("")
        lines.append("numeric summary:")
        lines.append(
            numeric.describe(
                percentiles=[0.05, 0.5, 0.9, 0.95, 0.99]
            ).transpose().to_string()
        )

    return lines


def describe_json(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    return [
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    ]


report: list[str] = []

for path in FILES:
    report.append("=" * 120)
    report.append(str(path))
    report.append("=" * 120)

    if not path.exists():
        report.append("ABSENT")
        report.append("")
        continue

    try:
        if path.suffix.lower() == ".csv":
            report.extend(describe_csv(path))
        elif path.suffix.lower() == ".json":
            report.extend(describe_json(path))
        else:
            report.append("Unsupported format")
    except Exception as exc:
        report.append(f"ERROR: {type(exc).__name__}: {exc}")

    report.append("")

OUT.write_text("\n".join(report), encoding="utf-8")
print(f"Written: {OUT}")
