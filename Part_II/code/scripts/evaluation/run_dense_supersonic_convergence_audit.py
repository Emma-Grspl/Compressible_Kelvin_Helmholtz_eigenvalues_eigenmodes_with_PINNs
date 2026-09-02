#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=False)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independent convergence audit for the frozen dense classical supersonic "
            "Riccati reference. Prepare selects three points per Mach; each task reruns "
            "spectral box/integration/matching sweeps and a modal-resolution sweep."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=Path("assets/classic_supersonic/dense_kappa_q_campaign_v1_FINAL_FREEZE"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "classic_supersonic/reproducibility/results/"
            "dense_supersonic_convergence_audit_v1"
        ),
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(repo: Path, path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo / path


def load_reference(freeze_dir: Path) -> pd.DataFrame:
    path = freeze_dir / "classical_supersonic_maps/classical_supersonic_dense_reference.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"Mach", "alpha", "cr", "ci", "omega_i"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Dense reference is missing columns: {sorted(missing)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.sort_values(["Mach", "alpha"]).reset_index(drop=True)


def choose_three(group: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    group = group.sort_values("alpha").reset_index(drop=True)
    candidates = [
        ("low_alpha", 0),
        ("peak_growth", int(group["omega_i"].idxmax())),
        ("near_neutral", len(group) - 1),
    ]
    used: set[int] = set()
    selected: list[tuple[str, pd.Series]] = []
    for role, preferred in candidates:
        order = sorted(range(len(group)), key=lambda i: (abs(i - preferred), i))
        index = next(i for i in order if i not in used)
        used.add(index)
        selected.append((role, group.iloc[index]))
    return selected


def prepare(repo: Path, freeze_dir: Path, output_dir: Path, overwrite: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / "audit_points.csv"
    config_path = output_dir / "audit_config.json"
    if (points_path.exists() or config_path.exists()) and not overwrite:
        print(f"Audit preparation already exists: {output_dir}")
        return 0

    reference = load_reference(freeze_dir)
    rows: list[dict[str, Any]] = []
    task_index = 0
    for Mach, group in reference.groupby("Mach", sort=True):
        for role, row in choose_three(group):
            rows.append(
                {
                    "task_index": task_index,
                    "role": role,
                    "Mach": float(Mach),
                    "alpha": float(row["alpha"]),
                    "reference_cr": float(row["cr"]),
                    "reference_ci": float(row["ci"]),
                    "reference_omega_i": float(row["omega_i"]),
                    "reference_residual_norm": float(row.get("residual_norm", math.nan)),
                    "reference_mode_gamma_mismatch": float(
                        row.get("mode_gamma_mismatch_at_match", math.nan)
                    ),
                }
            )
            task_index += 1
    points = pd.DataFrame(rows)
    if points["Mach"].nunique() != 17 or len(points) != 51:
        raise RuntimeError(
            f"Expected 51 audit points (17 Mach x 3 roles), got {len(points)}."
        )
    atomic_csv(points_path, points)

    audit_config = {
        "created_at": utc_now(),
        "freeze_dir": str(freeze_dir),
        "reference_csv": str(
            freeze_dir / "classical_supersonic_maps/classical_supersonic_dense_reference.csv"
        ),
        "n_tasks": int(len(points)),
        "selection": ["low_alpha", "peak_growth", "near_neutral"],
        "spectral_settings": [
            {
                "id": "box_L20",
                "extent": 20.0,
                "matching_y": 1.0,
                "max_step": 0.2,
                "rtol": 1e-11,
                "atol": 1e-13,
                "sweep": "box",
                "level": 20.0,
            },
            {
                "id": "box_L30",
                "extent": 30.0,
                "matching_y": 1.0,
                "max_step": 0.2,
                "rtol": 1e-11,
                "atol": 1e-13,
                "sweep": "box",
                "level": 30.0,
            },
            {
                "id": "strict_L40_y1",
                "extent": 40.0,
                "matching_y": 1.0,
                "max_step": 0.125,
                "rtol": 1e-11,
                "atol": 1e-13,
                "sweep": "shared_reference",
                "level": 40.0,
            },
            {
                "id": "box_L50",
                "extent": 50.0,
                "matching_y": 1.0,
                "max_step": 0.125,
                "rtol": 1e-11,
                "atol": 1e-13,
                "sweep": "box_reference",
                "level": 50.0,
            },
            {
                "id": "accuracy_coarse",
                "extent": 40.0,
                "matching_y": 1.0,
                "max_step": 0.5,
                "rtol": 1e-9,
                "atol": 1e-11,
                "sweep": "integration",
                "level": 0.5,
            },
            {
                "id": "accuracy_nominal",
                "extent": 40.0,
                "matching_y": 1.0,
                "max_step": 0.25,
                "rtol": 1e-10,
                "atol": 1e-12,
                "sweep": "integration",
                "level": 0.25,
            },
            {
                "id": "matching_y0p5",
                "extent": 40.0,
                "matching_y": 0.5,
                "max_step": 0.125,
                "rtol": 1e-11,
                "atol": 1e-13,
                "sweep": "matching",
                "level": 0.5,
            },
            {
                "id": "matching_y1p5",
                "extent": 40.0,
                "matching_y": 1.5,
                "max_step": 0.125,
                "rtol": 1e-11,
                "atol": 1e-13,
                "sweep": "matching",
                "level": 1.5,
            },
        ],
        "modal_settings": [
            {
                "id": "modal_coarse",
                "modal_numerical_extent": 40.0,
                "modal_dy": 0.05,
                "modal_max_step": 0.5,
                "modal_rtol": 1e-9,
                "modal_atol": 1e-11,
            },
            {
                "id": "modal_nominal",
                "modal_numerical_extent": 40.0,
                "modal_dy": 0.025,
                "modal_max_step": 0.25,
                "modal_rtol": 1e-10,
                "modal_atol": 1e-12,
            },
            {
                "id": "modal_strict",
                "modal_numerical_extent": 50.0,
                "modal_dy": 0.0125,
                "modal_max_step": 0.125,
                "modal_rtol": 1e-11,
                "modal_atol": 1e-13,
            },
        ],
        "comparison": {
            "core_limit": 20.0,
            "common_grid_points": 4001,
            "amplitude_threshold": 1e-3,
        },
        "acceptance_thresholds": {
            "dense_reference_residual_max": 1e-8,
            "spectral_run_residual_max": 1e-8,
            "box_L40_vs_L50_complex_error_max": 1e-5,
            "accuracy_nominal_vs_strict_complex_error_max": 1e-5,
            "matching_location_complex_error_max": 1e-5,
            "modal_nominal_p_rel_l2_max": 5e-3,
        },
    }
    atomic_json(config_path, audit_config)
    (output_dir / "tasks").mkdir(exist_ok=True)
    print(f"Prepared {len(points)} audit tasks in: {output_dir}")
    return 0


def load_solver_modules(repo: Path):
    validation_dir = repo / "classic_supersonic/scripts/validation"
    for path in (repo, validation_dir):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    import scripts.evaluation.run_dense_supersonic_campaign as campaign  # type: ignore
    import scripts.shooting.solve_reconstruct_dense_supersonic_modes as modal  # type: ignore

    return campaign, modal


def interp(y: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.interp(target, y, values)


def rel_l2(predicted: np.ndarray, reference: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        predicted = predicted[mask]
        reference = reference[mask]
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return math.nan
    return float(np.linalg.norm(predicted - reference) / denominator)


def modal_errors(
    arrays: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    strict = arrays.get("modal_strict")
    if strict is None:
        return []
    comparison = config["comparison"]
    limit = float(comparison["core_limit"])
    n = int(comparison["common_grid_points"])
    threshold = float(comparison["amplitude_threshold"])
    grid = np.linspace(-limit, limit, n)
    ref_y = strict["y"]
    ref_p = interp(ref_y, strict["p_real"], grid) + 1j * interp(
        ref_y, strict["p_imag"], grid
    )
    ref_modulus = np.abs(ref_p)
    ref_kappa = interp(ref_y, strict["kappa"], grid)
    ref_q = interp(ref_y, strict["q"], grid)
    mask = ref_modulus >= threshold
    rows: list[dict[str, Any]] = []
    for setting_id, current in arrays.items():
        cur_y = current["y"]
        cur_p = interp(cur_y, current["p_real"], grid) + 1j * interp(
            cur_y, current["p_imag"], grid
        )
        denom = np.vdot(cur_p[mask], cur_p[mask])
        factor = np.vdot(cur_p[mask], ref_p[mask]) / denom if abs(denom) > 0 else 1.0 + 0j
        aligned = factor * cur_p
        cur_kappa = interp(cur_y, current["kappa"], grid)
        cur_q = interp(cur_y, current["q"], grid)
        rows.append(
            {
                "modal_setting": setting_id,
                "is_modal_reference": setting_id == "modal_strict",
                "p_rel_l2_core": rel_l2(aligned, ref_p),
                "p_rel_l2_amp_mask": rel_l2(aligned, ref_p, mask),
                "modulus_rel_l2_amp_mask": rel_l2(np.abs(aligned), ref_modulus, mask),
                "kappa_rel_l2_amp_mask": rel_l2(cur_kappa, ref_kappa, mask),
                "q_rel_l2_amp_mask": rel_l2(cur_q, ref_q, mask),
                "alignment_real": float(np.real(factor)),
                "alignment_imag": float(np.imag(factor)),
                "comparison_core_limit": limit,
                "comparison_amplitude_threshold": threshold,
            }
        )
    return rows


def run_task(repo: Path, freeze_dir: Path, output_dir: Path, task_index: int, overwrite: bool) -> int:
    points_path = output_dir / "audit_points.csv"
    config_path = output_dir / "audit_config.json"
    if not points_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Run --prepare before task execution.")
    points = pd.read_csv(points_path)
    if task_index < 0 or task_index >= len(points):
        raise IndexError(task_index)
    point = points.iloc[task_index]
    task_dir = output_dir / "tasks" / f"task_{task_index:03d}"
    done = task_dir / "DONE.json"
    if done.exists() and not overwrite:
        print(f"Task {task_index} already complete: {done}")
        return 0
    task_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    production_config_path = repo / "code/configs/legacy/dense_supersonic_campaign_config.json"
    production_config = json.loads(production_config_path.read_text(encoding="utf-8"))
    campaign, modal = load_solver_modules(repo)

    Mach = float(point["Mach"])
    alpha = float(point["alpha"])
    seed_cr = float(point["reference_cr"])
    seed_ci = float(point["reference_ci"])
    print(
        f"=== convergence audit task {task_index}/{len(points)-1}: "
        f"M={Mach:.2f}, alpha={alpha:.12g}, role={point['role']} ===",
        flush=True,
    )

    spectral_rows: list[dict[str, Any]] = []
    spectral_results: dict[str, dict[str, Any]] = {}
    for item in config["spectral_settings"]:
        setting = campaign.SolverSettings(
            str(item["id"]),
            float(item["extent"]),
            float(item["matching_y"]),
            float(item["max_step"]),
            float(item["rtol"]),
            float(item["atol"]),
        )
        base_row = {
            "task_index": task_index,
            "role": str(point["role"]),
            "Mach": Mach,
            "alpha": alpha,
            "reference_cr": seed_cr,
            "reference_ci": seed_ci,
            "reference_omega_i": float(point["reference_omega_i"]),
            "audit_setting": str(item["id"]),
            "audit_sweep": str(item["sweep"]),
            "audit_level": float(item["level"]),
        }
        try:
            result = campaign.solve_once(
                Mach=Mach,
                alpha=alpha,
                seed_cr=seed_cr,
                seed_ci=seed_ci,
                settings=setting,
                config=production_config,
            )
            spectral_results[str(item["id"])] = result
            spectral_rows.append({**base_row, **result, "error": ""})
            print(
                f"  {item['id']}: accepted={result['accepted']} "
                f"res={result['residual_norm']:.3e} "
                f"c={result['cr']:.12g}+{result['ci']:.6e}i",
                flush=True,
            )
        except Exception as exc:
            spectral_rows.append(
                {
                    **base_row,
                    "accepted": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  {item['id']}: FAILED {type(exc).__name__}: {exc}", flush=True)
    spectral_frame = pd.DataFrame(spectral_rows)
    atomic_csv(task_dir / "spectral_runs.csv", spectral_frame)

    modal_rows: list[dict[str, Any]] = []
    modal_arrays: dict[str, dict[str, np.ndarray]] = {}
    strict_spectral = spectral_results.get("box_L50")
    if strict_spectral is not None and bool(strict_spectral.get("accepted")):
        spectral_row = pd.Series(
            {
                "alpha": alpha,
                "cr": float(strict_spectral["cr"]),
                "ci": float(strict_spectral["ci"]),
                "matching_y": 1.0,
            }
        )
        for item in config["modal_settings"]:
            modal_config = dict(production_config)
            modal_config.update(item)
            base_modal = {
                "task_index": task_index,
                "role": str(point["role"]),
                "Mach": Mach,
                "alpha": alpha,
                "modal_setting": str(item["id"]),
                "spectral_cr": float(strict_spectral["cr"]),
                "spectral_ci": float(strict_spectral["ci"]),
            }
            try:
                arrays, metadata = modal.reconstruct_mode(
                    row=spectral_row,
                    Mach=Mach,
                    config=modal_config,
                )
                modal_arrays[str(item["id"])] = arrays
                modal_rows.append({**base_modal, **metadata, "modal_success": True, "error": ""})
                print(
                    f"  {item['id']}: mode match={metadata['gamma_mismatch_at_match']:.3e}, "
                    f"n={metadata['n_coordinates']}",
                    flush=True,
                )
            except Exception as exc:
                modal_rows.append(
                    {
                        **base_modal,
                        "modal_success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"  {item['id']}: FAILED {type(exc).__name__}: {exc}", flush=True)
    else:
        for item in config["modal_settings"]:
            modal_rows.append(
                {
                    "task_index": task_index,
                    "role": str(point["role"]),
                    "Mach": Mach,
                    "alpha": alpha,
                    "modal_setting": str(item["id"]),
                    "modal_success": False,
                    "error": "Strict spectral reference box_L50 did not converge.",
                }
            )

    modal_frame = pd.DataFrame(modal_rows)
    errors = modal_errors(modal_arrays, config)
    error_frame = pd.DataFrame(
        [
            {
                "task_index": task_index,
                "role": str(point["role"]),
                "Mach": Mach,
                "alpha": alpha,
                **row,
            }
            for row in errors
        ]
    )
    atomic_csv(task_dir / "modal_runs.csv", modal_frame)
    atomic_csv(task_dir / "modal_errors.csv", error_frame)

    if modal_arrays:
        payload: dict[str, np.ndarray] = {}
        for setting_id, arrays in modal_arrays.items():
            for name, values in arrays.items():
                payload[f"{setting_id}__{name}"] = values
        temporary = task_dir / ".modal_arrays.tmp.npz"
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, task_dir / "modal_arrays.npz")

    atomic_json(
        done,
        {
            "completed_at": utc_now(),
            "task_index": task_index,
            "role": str(point["role"]),
            "Mach": Mach,
            "alpha": alpha,
            "spectral_runs": int(len(spectral_frame)),
            "spectral_accepted": int(spectral_frame.get("accepted", pd.Series(dtype=bool)).fillna(False).sum()),
            "modal_runs": int(len(modal_frame)),
            "modal_success": int(modal_frame.get("modal_success", pd.Series(dtype=bool)).fillna(False).sum()),
        },
    )
    print(f"Task written to: {task_dir}")
    return 0


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    freeze_dir = resolve(repo, args.freeze_dir)
    output_dir = resolve(repo, args.output_dir)
    if args.prepare:
        return prepare(repo, freeze_dir, output_dir, args.overwrite)
    task_index = args.task_index
    if task_index is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ValueError("Provide --prepare or --task-index/SLURM_ARRAY_TASK_ID.")
        task_index = int(raw)
    return run_task(repo, freeze_dir, output_dir, task_index, args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
