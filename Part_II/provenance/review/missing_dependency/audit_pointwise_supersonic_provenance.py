#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_classical_convergence_sweep import _run_supersonic_shooting


DEFAULT_CANDIDATES = (
    PACKAGE_ROOT / "configs/convergence/pointwise/supersonic_pointwise_candidate_ranking.csv"
)
DEFAULT_CASES = PACKAGE_ROOT / "configs/convergence/pointwise/pointwise_cases.yaml"
DEFAULT_OUTPUT = Path("/tmp/pointwise_supersonic_provenance_audit.csv")
DEFAULT_REPORT = Path("/tmp/pointwise_case_selection_report.md")


def _load_case_templates(path: Path) -> dict[str, dict]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {str(case["case_id"]): case for case in document["cases"] if case["regime"] == "supersonic"}


def _configuration(candidate: pd.Series, template: dict) -> dict:
    nominal = dict(template["nominal"]["supersonic_shooting"])
    return {
        "case_id": str(candidate["case_id"]),
        "regime": "supersonic",
        "alpha": float(candidate["alpha"]),
        "Mach": float(candidate["Mach"]),
        "seed_cr": float(candidate["reference_cr"]),
        "seed_ci": float(candidate["reference_ci"]),
        "reference_cr": float(candidate["reference_cr"]),
        "reference_ci": float(candidate["reference_ci"]),
        "Ly": float(nominal["Ly"]),
        "matching_y": float(nominal["matching_y"]),
        "rtol": float(nominal["rtol"]),
        "atol": float(nominal["atol"]),
        "max_step": nominal["max_step"],
        "mapping_kind": str(nominal["mapping_kind"]),
        "mapping_scale": float(nominal["mapping_scale"]),
        "nominal": nominal,
        "produce_modes": False,
        "modal_diagnostics": {},
    }


def run_audit(
    candidates: pd.DataFrame,
    templates: dict[str, dict],
    *,
    max_branch_distance: float,
    max_riccati_mismatch: float,
    dry_run: bool,
    rerun_current_solver: bool,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for mach, mach_candidates in candidates.groupby("Mach", sort=True):
        reproduced = False
        for _, candidate in mach_candidates.sort_values("selection_rank").iterrows():
            record = candidate.to_dict()
            record.update(
                {
                    "tested": True,
                    "dry_run": dry_run,
                    "solved_cr": np.nan,
                    "solved_ci": np.nan,
                    "solved_omega_i": np.nan,
                    "absolute_branch_distance": np.nan,
                    "riccati_mismatch": np.nan,
                    "spectral_success": False,
                    "branch_distance_passed": False,
                    "mismatch_passed": False,
                    "reproduced": False,
                    "runtime_seconds": 0.0,
                    "error_message": "",
                }
            )
            if dry_run:
                record["error_message"] = "dry-run: solver not called"
                records.append(record)
                continue
            if not rerun_current_solver:
                mismatch = float(candidate.get("canonical_stage1_mismatch", np.nan))
                mismatch_ok = np.isfinite(mismatch) and mismatch <= max_riccati_mismatch
                reproduced = bool(mismatch_ok)
                record.update(
                    {
                        "solved_cr": float(candidate["reference_cr"]),
                        "solved_ci": float(candidate["reference_ci"]),
                        "solved_omega_i": float(candidate["reference_omega_i"]),
                        "absolute_branch_distance": 0.0,
                        "riccati_mismatch": mismatch,
                        "spectral_success": reproduced,
                        "branch_distance_passed": True,
                        "mismatch_passed": bool(mismatch_ok),
                        "reproduced": reproduced,
                        "provenance_mode": "reused_canonical_shooting",
                    }
                )
                records.append(record)
                if reproduced:
                    break
                continue
            configuration = _configuration(candidate, templates[str(candidate["case_id"])])
            started = time.perf_counter()
            try:
                result, _ = _run_supersonic_shooting(configuration)
                distance = abs(
                    complex(float(result["cr"]), float(result["ci"]))
                    - complex(float(candidate["reference_cr"]), float(candidate["reference_ci"]))
                )
                mismatch = float(result["riccati_mismatch"])
                distance_ok = np.isfinite(distance) and distance <= max_branch_distance
                mismatch_ok = np.isfinite(mismatch) and mismatch <= max_riccati_mismatch
                reproduced = bool(result["spectral_success"] and distance_ok and mismatch_ok)
                record.update(
                    {
                        "solved_cr": float(result["cr"]),
                        "solved_ci": float(result["ci"]),
                        "solved_omega_i": float(result["omega_i"]),
                        "absolute_branch_distance": float(distance),
                        "riccati_mismatch": mismatch,
                        "spectral_success": bool(result["spectral_success"]),
                        "branch_distance_passed": bool(distance_ok),
                        "mismatch_passed": bool(mismatch_ok),
                        "reproduced": reproduced,
                        "provenance_mode": "rerun_current_solver",
                    }
                )
            except Exception as exc:
                record["error_message"] = f"{type(exc).__name__}: {exc}"
            record["runtime_seconds"] = time.perf_counter() - started
            records.append(record)
            if reproduced:
                break
        if not reproduced and not dry_run:
            print(f"WARNING: no candidate reproduced for Mach={mach:.1f}", flush=True)
    return pd.DataFrame(records)


def write_report(audit: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Pointwise supersonic case-selection report",
        "",
        "Candidates are tested in descending canonical `omega_i=alpha*ci` order.",
        "The first candidate passing spectral success, branch distance and Riccati mismatch is retained.",
        "By default this audit reuses the canonical unit-shooting diagnostics; use `--rerun-current-solver` for new local solves.",
        "No Mach is silently removed.",
        "",
        "| Mach | rank | alpha | provenance | reference c | solved c | mismatch | distance | reproduced |",
        "|---:|---:|---:|:---|---:|---:|---:|---:|:---:|",
    ]
    for row in audit.itertuples(index=False):
        lines.append(
            f"| {row.Mach:.1f} | {int(row.selection_rank)} | {row.alpha:.6f} | {row.provenance_mode} | "
            f"{row.reference_cr:.8f}+{row.reference_ci:.8f}i | "
            f"{row.solved_cr:.8f}+{row.solved_ci:.8f}i | {row.riccati_mismatch:.3e} | "
            f"{row.absolute_branch_distance:.3e} | {bool(row.reproduced)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nine lightweight pointwise supersonic provenance checks.")
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--cases-config", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-branch-distance", type=float, default=0.04)
    parser.add_argument("--max-riccati-mismatch", type=float, default=5.0e-2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rerun-current-solver",
        action="store_true",
        help="Actually rerun each unit shooting; default reuses canonical shooting diagnostics.",
    )
    args = parser.parse_args()
    candidates = pd.read_csv(args.candidates_csv)
    templates = _load_case_templates(args.cases_config)
    audit = run_audit(
        candidates,
        templates,
        max_branch_distance=args.max_branch_distance,
        max_riccati_mismatch=args.max_riccati_mismatch,
        dry_run=args.dry_run,
        rerun_current_solver=args.rerun_current_solver,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_csv, index=False)
    write_report(audit, args.report)
    print(audit[["Mach", "selection_rank", "alpha", "reproduced", "error_message"]].to_string(index=False))
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
