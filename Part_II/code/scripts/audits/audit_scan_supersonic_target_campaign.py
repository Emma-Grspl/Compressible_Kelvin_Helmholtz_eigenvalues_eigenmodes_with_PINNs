#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.audits.audit_scan_supersonic_M18_M19_strict_modal_validation import (
    solve_candidates_for_point,
    validate_candidate,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-csv", type=Path, required=True)
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--grid-size", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=7)
    ap.add_argument("--max-validate-per-point", type=int, default=8)
    args_cli = ap.parse_args()

    targets = pd.read_csv(args_cli.target_csv)
    targets["Mach"] = targets["Mach"].astype(float)
    targets["alpha"] = targets["alpha"].astype(float)
    targets = targets.sort_values(["Mach", "alpha"]).reset_index(drop=True)

    if args_cli.task_id is not None:
        if args_cli.task_id < 0 or args_cli.task_id >= len(targets):
            raise SystemExit(f"task-id outside range 0..{len(targets)-1}")
        targets = targets.iloc[[args_cli.task_id]].copy()

    outdir = args_cli.output_dir
    (outdir / "point_results").mkdir(parents=True, exist_ok=True)
    (outdir / "accepted_fields").mkdir(parents=True, exist_ok=True)

    scan_args = SimpleNamespace(
        search_y_limit=600.0,
        grid_size=args_cli.grid_size,
        max_iter=args_cli.max_iter,
        max_validate_per_point=args_cli.max_validate_per_point,

        stage1_tol=1e-3,
        stage2_tol=1e-8,
        max_edge_frac=2e-2,
        max_center_jump=0.25,
        max_adjacent_jump=0.35,
        max_ylimit_rel_l2=0.20,
    )

    all_rows = []

    for _, t in targets.iterrows():
        M = float(t["Mach"])
        a = float(t["alpha"])
        print(f"[target] M={M:.2f} alpha={a:.5f}", flush=True)

        candidates = solve_candidates_for_point(a, M, scan_args)

        rows = []
        accepted_fields = []

        for cand in candidates:
            if "cr" not in cand:
                cand["campaign_target_status"] = "raw_solve_failed"
                rows.append(cand)
                continue

            result, fields = validate_candidate(cand, scan_args)

            reasons = str(result.get("reject_reasons", ""))
            near_valid = (
                result.get("status") == "rejected"
                and reasons == "adjacent_jump_too_large"
                and float(result.get("stage1_mismatch", np.inf)) <= 1e-3
                and float(result.get("max_edge_frac", np.inf)) <= 2e-2
                and float(result.get("max_center_jump", np.inf)) <= 0.25
                and float(result.get("max_ylimit_rel_l2", np.inf)) <= 0.20
            )

            if result.get("status") == "strict_auto_validated":
                result["campaign_target_status"] = "strict_auto_validated"
            elif near_valid:
                result["campaign_target_status"] = "near_valid_except_adjacent_jump"
            else:
                result["campaign_target_status"] = "rejected"

            rows.append(result)

            print(
                f"  cand cr={result.get('cr')} ci={result.get('ci')} "
                f"stage1={result.get('stage1_mismatch')} "
                f"status={result.get('campaign_target_status')} "
                f"reasons={result.get('reject_reasons')}",
                flush=True,
            )

            if fields is not None:
                fields["campaign_target_status"] = result["campaign_target_status"]
                accepted_fields.append(fields)

        df = pd.DataFrame(rows)
        point_path = outdir / "point_results" / f"M{M:.2f}_alpha{a:.5f}_candidates.csv"
        df.to_csv(point_path, index=False)

        if accepted_fields:
            f = pd.concat(accepted_fields, ignore_index=True)
            fpath = outdir / "accepted_fields" / f"M{M:.2f}_alpha{a:.5f}_accepted_fields.csv"
            f.to_csv(fpath, index=False)

        all_rows.extend(rows)

    if args_cli.task_id is None:
        df = pd.DataFrame(all_rows)
        df.to_csv(outdir / "all_candidates.csv", index=False)

        good = df[df["campaign_target_status"].isin([
            "strict_auto_validated",
            "near_valid_except_adjacent_jump",
        ])].copy()
        good.to_csv(outdir / "campaign_near_valid_candidates.csv", index=False)

        summary = {
            "n_targets": int(len(targets)),
            "n_candidate_rows": int(len(df)),
            "status_counts": df["campaign_target_status"].value_counts(dropna=False).to_dict(),
            "n_near_valid_or_strict": int(len(good)),
            "by_Mach": good.groupby("Mach").size().to_dict() if len(good) else {},
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
