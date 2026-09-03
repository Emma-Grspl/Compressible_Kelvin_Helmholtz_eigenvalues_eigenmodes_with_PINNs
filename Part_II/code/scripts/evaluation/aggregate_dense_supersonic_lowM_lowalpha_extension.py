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


CONVERGED_STATUSES = {"converged", "anchor_converged"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and strictly validate the low-Mach/low-alpha extension "
            "of the dense classical supersonic campaign."
        )
    )
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


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def decimal_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0 or stop < start:
        raise ValueError("Invalid alpha grid.")
    count = int(math.floor((stop - start) / step + 1.0e-10))
    grid = start + step * np.arange(count + 1, dtype=float)
    if grid[-1] < stop - 1.0e-10:
        grid = np.append(grid, stop)
    grid[-1] = min(grid[-1], stop)
    return np.unique(np.round(grid, 12))


def key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_Mach": pd.to_numeric(frame["Mach"], errors="coerce").round(10),
            "_alpha": pd.to_numeric(frame["alpha"], errors="coerce").round(12),
        },
        index=frame.index,
    )


def requested_pairs(config: dict[str, Any]) -> pd.DataFrame:
    alphas = decimal_grid(
        float(config["alpha_min"]),
        float(config["alpha_max"]),
        float(config["alpha_step"]),
    )
    rows = [
        {"Mach": float(Mach), "alpha": float(alpha)}
        for Mach in config["mach_values"]
        for alpha in alphas
    ]
    frame = pd.DataFrame(rows)
    keys = key_frame(frame)
    return pd.concat([frame, keys], axis=1)


def resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def choose_target_rows(all_rows: pd.DataFrame, requested: pd.DataFrame) -> pd.DataFrame:
    work = all_rows.copy()
    keys = key_frame(work)
    work = pd.concat([work, keys], axis=1)
    work = work.merge(
        requested[["_Mach", "_alpha"]],
        on=["_Mach", "_alpha"],
        how="inner",
    )
    is_target = work["is_target"].astype(str).str.lower().isin(("true", "1"))
    work = work.loc[is_target].copy()
    if "timestamp" in work.columns:
        work = work.sort_values("timestamp", kind="stable")
    work = work.drop_duplicates(["_Mach", "_alpha"], keep="last")
    return work.sort_values(["_Mach", "_alpha"]).reset_index(drop=True)


def select_retained(targets: pd.DataFrame) -> pd.DataFrame:
    cr = pd.to_numeric(targets["cr"], errors="coerce")
    ci = pd.to_numeric(targets["ci"], errors="coerce")
    status = targets["status"].astype(str)
    retained = targets.loc[
        status.isin(CONVERGED_STATUSES)
        & np.isfinite(cr)
        & np.isfinite(ci)
        & (ci > 0.0)
    ].copy()
    retained["cr"] = pd.to_numeric(retained["cr"], errors="coerce")
    retained["ci"] = pd.to_numeric(retained["ci"], errors="coerce")
    retained["omega_i"] = retained["alpha"].astype(float) * retained["ci"]
    return retained.sort_values(["_Mach", "_alpha"]).reset_index(drop=True)


def select_modes(all_modes: pd.DataFrame, requested: pd.DataFrame) -> pd.DataFrame:
    if all_modes.empty:
        return all_modes
    work = all_modes.copy()
    keys = key_frame(work)
    work = pd.concat([work, keys], axis=1)
    work = work.merge(
        requested[["_Mach", "_alpha"]],
        on=["_Mach", "_alpha"],
        how="inner",
    )
    if "status" in work.columns:
        work = work.loc[work["status"].astype(str).eq("converged")].copy()
    if "updated" in work.columns:
        work = work.sort_values("updated", kind="stable")
    work = work.drop_duplicates(["_Mach", "_alpha"], keep="last")
    return work.sort_values(["_Mach", "_alpha"]).reset_index(drop=True)


def compare_overlap(
    retained: pd.DataFrame,
    original_path: Path,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not original_path.is_file():
        return pd.DataFrame(), {
            "available": False,
            "path": str(original_path),
            "reason": "original reference not found",
        }

    original = pd.read_csv(original_path)
    original_keys = key_frame(original)
    original = pd.concat([original, original_keys], axis=1)

    extension = retained.copy()
    overlap = extension.merge(
        original,
        on=["_Mach", "_alpha"],
        how="inner",
        suffixes=("_extension", "_original"),
    )
    if overlap.empty:
        return overlap, {
            "available": True,
            "path": str(original_path),
            "n_overlap": 0,
            "passed": False,
            "reason": "no overlap point found",
        }

    overlap["delta_cr"] = (
        pd.to_numeric(overlap["cr_extension"], errors="coerce")
        - pd.to_numeric(overlap["cr_original"], errors="coerce")
    )
    overlap["delta_ci"] = (
        pd.to_numeric(overlap["ci_extension"], errors="coerce")
        - pd.to_numeric(overlap["ci_original"], errors="coerce")
    )
    overlap["delta_c_abs"] = np.hypot(overlap["delta_cr"], overlap["delta_ci"])
    max_delta = float(overlap["delta_c_abs"].max())
    passed = bool(np.isfinite(max_delta) and max_delta <= tolerance)
    return overlap, {
        "available": True,
        "path": str(original_path),
        "n_overlap": int(len(overlap)),
        "max_delta_c_abs": max_delta,
        "tolerance": float(tolerance),
        "passed": passed,
    }


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    config_path = resolve(repo, args.config.expanduser())
    config = json.loads(config_path.read_text(encoding="utf-8"))

    output_root = resolve(repo, Path(str(config["output_root"])))
    output_root.mkdir(parents=True, exist_ok=True)
    requested = requested_pairs(config)

    spectral_frames: list[pd.DataFrame] = []
    mode_frames: list[pd.DataFrame] = []
    for Mach in [float(value) for value in config["mach_values"]]:
        mach_dir = output_root / f"M{Mach:.6f}".replace(".", "p")
        spectral = load_csv(mach_dir / "spectral_points.csv")
        spectral["mach_directory"] = str(mach_dir)
        spectral_frames.append(spectral)

        mode_path = mach_dir / "modes" / "mode_summary.csv"
        if mode_path.is_file():
            modes = pd.read_csv(mode_path)
            modes["mach_directory"] = str(mach_dir)
            mode_frames.append(modes)

    all_spectral = pd.concat(spectral_frames, ignore_index=True, sort=False)
    all_modes = (
        pd.concat(mode_frames, ignore_index=True, sort=False)
        if mode_frames
        else pd.DataFrame()
    )

    targets = choose_target_rows(all_spectral, requested)
    retained = select_retained(targets)
    modes = select_modes(all_modes, requested)

    requested_keys = set(map(tuple, requested[["_Mach", "_alpha"]].to_numpy()))
    target_keys = set(map(tuple, targets[["_Mach", "_alpha"]].to_numpy()))
    retained_keys = set(map(tuple, retained[["_Mach", "_alpha"]].to_numpy()))
    mode_keys = set(map(tuple, modes[["_Mach", "_alpha"]].to_numpy()))

    missing_targets = sorted(requested_keys - target_keys)
    nonconverged = sorted(requested_keys - retained_keys)
    missing_modes = sorted(retained_keys - mode_keys)
    extra_modes = sorted(mode_keys - retained_keys)

    overlap_path = resolve(repo, args.original_reference.expanduser())
    overlap, overlap_summary = compare_overlap(
        retained,
        overlap_path,
        args.overlap_tolerance,
    )

    clean_columns = [column for column in retained.columns if not column.startswith("_")]
    dense_reference = retained[clean_columns].copy()
    dense_reference["reference_block"] = "lowM_lowalpha_extension_v1"

    atomic_write_csv(output_root / "all_spectral_rows.csv", all_spectral)
    atomic_write_csv(output_root / "lowM_lowalpha_spectral_targets.csv", targets)
    atomic_write_csv(output_root / "lowM_lowalpha_spectral_retained.csv", retained)
    atomic_write_csv(output_root / "all_mode_summaries.csv", modes)
    atomic_write_csv(output_root / "lowM_lowalpha_dense_reference.csv", dense_reference)
    if not overlap.empty:
        atomic_write_csv(output_root / "overlap_consistency.csv", overlap)

    expected = int(len(requested))
    complete = (
        len(targets) == expected
        and len(retained) == expected
        and len(modes) == expected
        and not missing_targets
        and not nonconverged
        and not missing_modes
        and not extra_modes
        and bool(overlap_summary.get("passed", False))
    )

    summary = {
        "campaign": "dense_kappa_q_lowM_lowalpha_extension_v1",
        "config": str(config_path),
        "output_root": str(output_root),
        "mach_values": [float(value) for value in config["mach_values"]],
        "alpha_min": float(config["alpha_min"]),
        "alpha_max": float(config["alpha_max"]),
        "alpha_step": float(config["alpha_step"]),
        "expected_grid_points": expected,
        "spectral_rows": int(len(all_spectral)),
        "target_rows": int(len(targets)),
        "retained_rows": int(len(retained)),
        "mode_rows": int(len(modes)),
        "missing_targets": missing_targets,
        "nonconverged_targets": nonconverged,
        "missing_modes": missing_modes,
        "extra_modes": extra_modes,
        "max_spectral_residual": (
            float(pd.to_numeric(retained["residual_norm"], errors="coerce").max())
            if "residual_norm" in retained.columns and not retained.empty
            else None
        ),
        "overlap_consistency": overlap_summary,
        "complete": bool(complete),
    }
    atomic_write_text(
        output_root / "extension_summary.json",
        json.dumps(summary, indent=2, sort_keys=True),
    )

    print("=== LOW-M / LOW-ALPHA EXTENSION ===")
    print(f"Expected grid points : {expected}")
    print(f"Target rows          : {len(targets)}")
    print(f"Retained roots       : {len(retained)}")
    print(f"Reconstructed modes  : {len(modes)}")
    print(f"Missing targets      : {len(missing_targets)}")
    print(f"Nonconverged targets : {len(nonconverged)}")
    print(f"Missing modes        : {len(missing_modes)}")
    if overlap_summary.get("available"):
        print(
            "Overlap max |delta c|: "
            f"{overlap_summary.get('max_delta_c_abs')} "
            f"(tol={overlap_summary.get('tolerance')})"
        )
    print(f"Written to           : {output_root}")
    print(f"EXTENSION STATUS     : {'PASS' if complete else 'FAIL'}")

    if not complete:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
