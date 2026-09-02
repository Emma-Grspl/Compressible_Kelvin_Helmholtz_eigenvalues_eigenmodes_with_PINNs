#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONVERGED = {"converged", "anchor_converged"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--original-reference",
        type=Path,
        default=Path(
            "classic_supersonic/reproducibility/results/"
            "dense_kappa_q_campaign_v1/dense_spectral_retained.csv"
        ),
    )
    parser.add_argument("--overlap-tolerance", type=float, default=1.0e-6)
    return parser.parse_args()


def resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def decimal_grid(start: float, stop: float, step: float) -> np.ndarray:
    count = int(math.floor((stop - start) / step + 1.0e-10))
    grid = start + step * np.arange(count + 1, dtype=float)
    if grid[-1] < stop - 1.0e-10:
        grid = np.append(grid, stop)
    return np.unique(np.round(grid, 12))


def keys(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_Mach": pd.to_numeric(frame["Mach"], errors="coerce").round(10),
            "_alpha": pd.to_numeric(frame["alpha"], errors="coerce").round(12),
        },
        index=frame.index,
    )


def latest_targets(frame: pd.DataFrame, requested: pd.DataFrame) -> pd.DataFrame:
    work = pd.concat([frame.copy(), keys(frame)], axis=1)
    work = work.merge(requested[["_Mach", "_alpha"]], on=["_Mach", "_alpha"], how="inner")
    work = work.loc[work["is_target"].astype(str).str.lower().isin(("true", "1"))].copy()
    if "timestamp" in work.columns:
        work = work.sort_values("timestamp", kind="stable")
    return work.drop_duplicates(["_Mach", "_alpha"], keep="last").sort_values(["_Mach", "_alpha"])


def latest_modes(frame: pd.DataFrame, requested: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = pd.concat([frame.copy(), keys(frame)], axis=1)
    work = work.merge(requested[["_Mach", "_alpha"]], on=["_Mach", "_alpha"], how="inner")
    work = work.loc[work["status"].astype(str).eq("converged")].copy()
    if "updated" in work.columns:
        work = work.sort_values("updated", kind="stable")
    return work.drop_duplicates(["_Mach", "_alpha"], keep="last").sort_values(["_Mach", "_alpha"])


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    config_path = resolve(repo, args.config.expanduser())
    config = json.loads(config_path.read_text())
    root = resolve(repo, Path(str(config["output_root"])))

    alphas = decimal_grid(float(config["alpha_min"]), float(config["alpha_max"]), float(config["alpha_step"]))
    requested = pd.DataFrame(
        [
            {"Mach": float(Mach), "alpha": float(alpha)}
            for Mach in config["mach_values"]
            for alpha in alphas
        ]
    )
    requested = pd.concat([requested, keys(requested)], axis=1)

    spectral_parts = []
    mode_parts = []
    for Mach in [float(value) for value in config["mach_values"]]:
        directory = root / f"M{Mach:.6f}".replace(".", "p")
        spectral = pd.read_csv(directory / "spectral_points.csv")
        spectral["mach_directory"] = str(directory)
        spectral_parts.append(spectral)
        mode_path = directory / "modes" / "mode_summary.csv"
        if mode_path.is_file():
            modes = pd.read_csv(mode_path)
            modes["mach_directory"] = str(directory)
            mode_parts.append(modes)

    all_spectral = pd.concat(spectral_parts, ignore_index=True, sort=False)
    all_modes = pd.concat(mode_parts, ignore_index=True, sort=False) if mode_parts else pd.DataFrame()
    targets = latest_targets(all_spectral, requested)

    cr = pd.to_numeric(targets["cr"], errors="coerce")
    ci = pd.to_numeric(targets["ci"], errors="coerce")
    retained = targets.loc[
        targets["status"].astype(str).isin(CONVERGED)
        & np.isfinite(cr)
        & np.isfinite(ci)
        & (ci > 0.0)
    ].copy()
    retained["cr"] = pd.to_numeric(retained["cr"], errors="coerce")
    retained["ci"] = pd.to_numeric(retained["ci"], errors="coerce")
    retained["omega_i"] = retained["alpha"].astype(float) * retained["ci"]
    modes = latest_modes(all_modes, requested)

    expected_keys = set(map(tuple, requested[["_Mach", "_alpha"]].to_numpy()))
    retained_keys = set(map(tuple, retained[["_Mach", "_alpha"]].to_numpy()))
    mode_keys = set(map(tuple, modes[["_Mach", "_alpha"]].to_numpy())) if not modes.empty else set()
    missing_roots = sorted(expected_keys - retained_keys)
    missing_modes = sorted(expected_keys - mode_keys)

    residual = pd.to_numeric(retained.get("residual_norm", pd.Series(dtype=float)), errors="coerce")
    max_residual = float(residual.max()) if len(residual) else None
    residual_limit = float(config.get("root_tolerance", 1.0e-8))

    overlap_summary: dict[str, Any] = {"available": False}
    overlap_path = resolve(repo, args.original_reference.expanduser())
    overlap = pd.DataFrame()
    if overlap_path.is_file():
        original = pd.read_csv(overlap_path)
        original = pd.concat([original, keys(original)], axis=1)
        overlap = retained.merge(original, on=["_Mach", "_alpha"], suffixes=("_extension", "_original"))
        if not overlap.empty:
            overlap["delta_cr"] = pd.to_numeric(overlap["cr_extension"], errors="coerce") - pd.to_numeric(overlap["cr_original"], errors="coerce")
            overlap["delta_ci"] = pd.to_numeric(overlap["ci_extension"], errors="coerce") - pd.to_numeric(overlap["ci_original"], errors="coerce")
            overlap["delta_c_abs"] = np.hypot(overlap["delta_cr"], overlap["delta_ci"])
            max_delta = float(overlap["delta_c_abs"].max())
            overlap_summary = {
                "available": True,
                "n_overlap": int(len(overlap)),
                "max_delta_c_abs": max_delta,
                "tolerance": float(args.overlap_tolerance),
                "passed": bool(np.isfinite(max_delta) and max_delta <= args.overlap_tolerance),
            }
        else:
            overlap_summary = {
                "available": True,
                "n_overlap": 0,
                "passed": None,
                "note": "No common retained target; overlap is reported but is not a completion criterion.",
            }

    complete = (
        len(targets) == len(requested)
        and len(retained) == len(requested)
        and len(modes) == len(requested)
        and not missing_roots
        and not missing_modes
        and max_residual is not None
        and np.isfinite(max_residual)
        and max_residual <= residual_limit
        and (overlap_summary.get("passed") is not False)
    )

    clean = retained[[c for c in retained.columns if not c.startswith("_")]].copy()
    clean["reference_block"] = "lowM_lowalpha_extension_v1_mach_continuation"
    atomic_csv(root / "all_spectral_rows.csv", all_spectral)
    atomic_csv(root / "lowM_lowalpha_spectral_targets.csv", targets)
    atomic_csv(root / "lowM_lowalpha_spectral_retained.csv", retained)
    atomic_csv(root / "all_mode_summaries.csv", modes)
    atomic_csv(root / "lowM_lowalpha_dense_reference.csv", clean)
    if not overlap.empty:
        atomic_csv(root / "overlap_consistency.csv", overlap)

    counts = retained.groupby("Mach").size().to_dict() if not retained.empty else {}
    summary = {
        "expected_grid_points": int(len(requested)),
        "target_rows": int(len(targets)),
        "retained_rows": int(len(retained)),
        "mode_rows": int(len(modes)),
        "retained_per_mach": {str(k): int(v) for k, v in counts.items()},
        "missing_roots": missing_roots,
        "missing_modes": missing_modes,
        "max_spectral_residual": max_residual,
        "residual_limit": residual_limit,
        "overlap_consistency": overlap_summary,
        "complete": bool(complete),
    }
    atomic_json(root / "extension_summary.json", summary)

    print("=== LOW-M / LOW-ALPHA REPAIRED EXTENSION ===")
    print(f"Expected grid points : {len(requested)}")
    print(f"Target rows          : {len(targets)}")
    print(f"Retained roots       : {len(retained)}")
    print(f"Reconstructed modes  : {len(modes)}")
    print(f"Missing roots        : {len(missing_roots)}")
    print(f"Missing modes        : {len(missing_modes)}")
    print(f"Max spectral residual: {max_residual} (limit={residual_limit})")
    print(f"Points per Mach      : {counts}")
    print(f"Overlap              : {overlap_summary}")
    print(f"Written to           : {root}")
    print(f"EXTENSION STATUS     : {'PASS' if complete else 'FAIL'}")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
