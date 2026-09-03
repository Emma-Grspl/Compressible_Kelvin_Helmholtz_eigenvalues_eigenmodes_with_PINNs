#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    mapping = {str(c).lower(): str(c) for c in frame.columns}
    for name in names:
        found = mapping.get(name.lower())
        if found is not None:
            return found
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    extension_root = (
        repo
        / "classic_supersonic/reproducibility/results/"
          "dense_kappa_q_lowM_lowalpha_extension_v1"
    )
    base_config_path = (
        repo / "code/configs/legacy/dense_supersonic_campaign_config.json"
    )
    anchor_path = (
        repo / "classic_supersonic/data/spectral/"
               "lowM_full_branch_anchor_alpha_0p05.csv"
    )
    config_path = (
        repo / "classic_supersonic/configs/"
               "dense_supersonic_lowM_full_branches_config.json"
    )

    if not extension_root.is_dir():
        raise FileNotFoundError(extension_root)
    if not base_config_path.is_file():
        raise FileNotFoundError(base_config_path)

    preferred = [
        extension_root / "lowM_lowalpha_dense_reference.csv",
        extension_root / "lowM_lowalpha_spectral_retained.csv",
        extension_root / "dense_spectral_retained.csv",
        extension_root / "all_spectral_rows.csv",
    ]
    csv_paths = [p for p in preferred if p.is_file()]
    if not csv_paths:
        csv_paths = sorted(extension_root.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV found under {extension_root}")

    rows: list[pd.DataFrame] = []
    for path in csv_paths:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        m_col = first_column(frame, ("Mach", "M", "mach"))
        a_col = first_column(frame, ("alpha", "Alpha"))
        cr_col = first_column(frame, ("cr", "reference_cr", "best_cr"))
        ci_col = first_column(frame, ("ci", "reference_ci", "best_ci"))
        if None in (m_col, a_col, cr_col, ci_col):
            continue
        work = pd.DataFrame(
            {
                "Mach": numeric(frame[m_col]),
                "alpha": numeric(frame[a_col]),
                "cr": numeric(frame[cr_col]),
                "ci": numeric(frame[ci_col]),
            }
        )
        if "residual_norm" in frame.columns:
            work["residual_norm"] = numeric(frame["residual_norm"])
        else:
            work["residual_norm"] = np.nan
        if "status" in frame.columns:
            work["status"] = frame["status"].astype(str)
        else:
            work["status"] = "converged"
        work["source_path"] = str(path)
        work = work[
            np.isfinite(work["Mach"])
            & np.isfinite(work["alpha"])
            & np.isfinite(work["cr"])
            & np.isfinite(work["ci"])
            & (work["ci"] > 0.0)
            & np.isclose(work["alpha"], 0.05, rtol=0.0, atol=5e-12)
        ].copy()
        if not work.empty:
            rows.append(work)

    if not rows:
        raise RuntimeError(
            "No converged alpha=0.05 anchors found in the validated extension."
        )

    anchors = pd.concat(rows, ignore_index=True, sort=False)
    anchors["residual_sort"] = anchors["residual_norm"].fillna(np.inf)
    anchors = (
        anchors.sort_values(["Mach", "residual_sort"])
        .drop_duplicates(["Mach", "alpha"], keep="first")
        .drop(columns="residual_sort")
        .sort_values("Mach")
        .reset_index(drop=True)
    )

    # For M=1.00, the nearest validated source should be M=1.01.
    # For M=1.05, an exact source must exist.
    if not np.any(np.isclose(anchors["Mach"], 1.05, atol=5e-10)):
        raise RuntimeError(
            "The validated extension has no M=1.05, alpha=0.05 anchor."
        )
    if float(np.min(np.abs(anchors["Mach"] - 1.00))) > 0.051:
        raise RuntimeError(
            "No validated alpha=0.05 source is close enough to M=1.00."
        )

    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(anchor_path, index=False)

    config = json.loads(base_config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "mach_values": [1.0, 1.05],
            "alpha_min": 0.05,
            "alpha_max": 0.45,
            "alpha_step": 0.005,
            "output_root": (
                "classic_supersonic/reproducibility/results/"
                "dense_kappa_q_lowM_full_branches_v1"
            ),
            "anchor_table": str(anchor_path.relative_to(repo)),
            "anchor_mach_max_distance": 0.06,
            "anchor_candidates": 20,
            "max_nfev": max(int(config.get("max_nfev", 120)), 180),
            "minimum_bridge_step": min(
                float(config.get("minimum_bridge_step", 1e-5)), 1e-6
            ),
            "max_bridge_attempts_per_target": max(
                int(config.get("max_bridge_attempts_per_target", 120)), 180
            ),
        }
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== LOW-M FULL-BRANCH PREPARATION ===")
    print(f"Anchor table : {anchor_path}")
    print(anchors.to_string(index=False))
    print(f"Config       : {config_path}")
    print("Mach values  : [1.0, 1.05]")
    print("Alpha grid   : 0.05 .. 0.45, step 0.005")
    print("PREP STATUS  : PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
