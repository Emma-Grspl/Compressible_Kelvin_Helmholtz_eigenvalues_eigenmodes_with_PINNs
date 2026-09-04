#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REQUIRED_AGGREGATES = (
    "all_spectral_rows.csv",
    "dense_spectral_targets.csv",
    "dense_spectral_retained.csv",
    "all_mode_summaries.csv",
)
REQUIRED_FINAL_ASSETS = (
    "classical_supersonic_maps/cr_map.pdf",
    "classical_supersonic_maps/ci_map.pdf",
    "classical_supersonic_maps/omega_i_map.pdf",
    "classical_supersonic_maps/neutral_curve.pdf",
    "classical_supersonic_maps/retained_point_mask.pdf",
    "classical_supersonic_maps/classical_supersonic_dense_reference.csv",
    "classical_supersonic_all_modes.pdf",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_rows(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return max(0, sum(1 for _ in csv.reader(handle)) - 1)
    except Exception:
        return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo / path


def alpha_grid(config: dict[str, Any]) -> np.ndarray:
    start = float(config["alpha_min"])
    stop = float(config["alpha_max"])
    step = float(config["alpha_step"])
    count = int(round((stop - start) / step)) + 1
    values = start + step * np.arange(count, dtype=float)
    if not np.isclose(values[-1], stop, atol=1e-11, rtol=0.0):
        raise ValueError("The configured alpha grid does not end at alpha_max.")
    return values


def key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Mach", "alpha"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing key columns: {sorted(missing)}")
    result = pd.DataFrame(
        {
            "Mach": pd.to_numeric(frame["Mach"], errors="raise").round(10),
            "alpha": pd.to_numeric(frame["alpha"], errors="raise").round(12),
        }
    )
    return result


def validate_campaign(
    *,
    campaign_root: Path,
    config: dict[str, Any],
    expected_targets: int | None,
    expected_retained: int | None,
    expected_modes: int | None,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_AGGREGATES if not (campaign_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing aggregate files in {campaign_root}: {missing}")

    rows = pd.read_csv(campaign_root / "all_spectral_rows.csv")
    targets = pd.read_csv(campaign_root / "dense_spectral_targets.csv")
    retained = pd.read_csv(campaign_root / "dense_spectral_retained.csv")
    modes = pd.read_csv(campaign_root / "all_mode_summaries.csv")

    target_keys = key_frame(targets)
    retained_keys = key_frame(retained)
    mode_keys = key_frame(modes)

    for label, keys in (
        ("targets", target_keys),
        ("retained", retained_keys),
        ("modes", mode_keys),
    ):
        duplicated = keys.duplicated(["Mach", "alpha"], keep=False)
        if duplicated.any():
            duplicate_rows = keys.loc[duplicated].head(20).to_dict("records")
            raise ValueError(f"Duplicate {label} keys: {duplicate_rows}")

    target_index = pd.MultiIndex.from_frame(target_keys)
    retained_index = pd.MultiIndex.from_frame(retained_keys)
    mode_index = pd.MultiIndex.from_frame(mode_keys)
    if not retained_index.isin(target_index).all():
        raise ValueError("Some retained points are absent from the target grid.")

    retained_set = set(retained_index.tolist())
    mode_set = set(mode_index.tolist())
    missing_modes = sorted(retained_set.difference(mode_set))
    extra_modes = sorted(mode_set.difference(retained_set))
    if missing_modes or extra_modes:
        raise ValueError(
            "Retained/mode key mismatch: "
            f"missing_modes={missing_modes[:20]}, extra_modes={extra_modes[:20]}"
        )

    configured_mach = np.asarray([float(value) for value in config["mach_values"]], dtype=float)
    configured_alpha = alpha_grid(config)
    configured_pairs = {
        (round(float(Mach), 10), round(float(alpha), 12))
        for Mach in configured_mach
        for alpha in configured_alpha
    }
    configured_targets = len(configured_pairs)
    target_set = set(target_index.tolist())

    # The production runner stores every converged anchor as a target.
    # An anchor outside the regular alpha grid is therefore an intentional
    # additional target, not an erroneous grid point.
    missing_configured = sorted(configured_pairs.difference(target_set))
    if missing_configured:
        raise ValueError(
            "Configured grid points are missing from the target table: "
            f"{missing_configured[:20]}"
        )

    anchor_mask = (
        targets.get("status", pd.Series("", index=targets.index))
        .astype(str)
        .eq("anchor_converged")
        | targets.get("direction", pd.Series("", index=targets.index))
        .astype(str)
        .eq("anchor")
    )
    anchor_keys = key_frame(targets.loc[anchor_mask])
    anchor_set = set(pd.MultiIndex.from_frame(anchor_keys).tolist())

    extra_targets = sorted(target_set.difference(configured_pairs))
    extra_non_anchor = sorted(set(extra_targets).difference(anchor_set))
    if extra_non_anchor:
        raise ValueError(
            "Off-grid target rows are not converged anchors: "
            f"{extra_non_anchor[:20]}"
        )

    n_off_grid_anchor_targets = len(extra_targets)

    expectations = {
        "targets": expected_targets,
        "retained": expected_retained,
        "modes": expected_modes,
    }
    actual = {"targets": len(targets), "retained": len(retained), "modes": len(modes)}
    for key, expected in expectations.items():
        if expected is not None and actual[key] != expected:
            raise ValueError(f"Expected {expected} {key}, found {actual[key]}.")

    numeric_ci = pd.to_numeric(retained.get("ci"), errors="coerce")
    if numeric_ci.isna().any() or (numeric_ci <= 0.0).any():
        raise ValueError("Retained table contains non-finite or non-positive ci values.")

    residual = pd.to_numeric(retained.get("residual_norm"), errors="coerce")
    max_residual = float(residual.max()) if residual.notna().any() else None
    mode_status_counts = (
        modes["status"].astype(str).value_counts(dropna=False).to_dict()
        if "status" in modes.columns
        else {}
    )

    per_mach: list[dict[str, Any]] = []
    for Mach in configured_mach:
        mach_dir = campaign_root / f"M{Mach:.6f}".replace(".", "p")
        required = (
            mach_dir / "spectral_points.csv",
            mach_dir / "campaign_summary.json",
            mach_dir / "modes" / "mode_summary.csv",
            mach_dir / "modes" / "modes_compact_with_analytic_tails.npz",
        )
        absent = [str(path.relative_to(campaign_root)) for path in required if not path.is_file()]
        if absent:
            raise FileNotFoundError(f"Missing per-Mach files for M={Mach}: {absent}")
        retained_count = int(np.sum(np.isclose(retained_keys["Mach"], Mach, atol=5e-11)))
        mode_count = int(np.sum(np.isclose(mode_keys["Mach"], Mach, atol=5e-11)))
        if retained_count != mode_count:
            raise ValueError(
                f"M={Mach}: retained_count={retained_count} != mode_count={mode_count}."
            )
        per_mach.append(
            {
                "Mach": float(Mach),
                "retained": retained_count,
                "modes": mode_count,
                "directory": mach_dir.name,
            }
        )

    return {
        "validated_at": utc_now(),
        "campaign_root": str(campaign_root),
        "n_spectral_rows": int(len(rows)),
        "n_targets": int(len(targets)),
        "n_retained": int(len(retained)),
        "n_modes": int(len(modes)),
        "configured_targets": configured_targets,
        "n_off_grid_anchor_targets": n_off_grid_anchor_targets,
        "n_mach": int(configured_mach.size),
        "n_alpha": int(configured_alpha.size),
        "Mach_min": float(configured_mach.min()),
        "Mach_max": float(configured_mach.max()),
        "alpha_min": float(configured_alpha.min()),
        "alpha_max": float(configured_alpha.max()),
        "alpha_step": float(config["alpha_step"]),
        "max_retained_residual_norm": max_residual,
        "mode_status_counts": mode_status_counts,
        "per_mach": per_mach,
    }


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_optional(source: Path, destination: Path) -> bool:
    if source.is_file():
        copy_file(source, destination)
        return True
    return False


def prepare_freeze(
    *,
    repo: Path,
    config_path: Path,
    campaign_root: Path,
    freeze_dir: Path,
    expected_targets: int | None,
    expected_retained: int | None,
    expected_modes: int | None,
    include_checkpoints: bool,
    overwrite: bool,
) -> None:
    config = load_json(config_path)
    validation = validate_campaign(
        campaign_root=campaign_root,
        config=config,
        expected_targets=expected_targets,
        expected_retained=expected_retained,
        expected_modes=expected_modes,
    )

    if freeze_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Freeze destination already exists: {freeze_dir}. "
                "Use a new freeze name; overwrite is intentionally explicit."
            )
        shutil.rmtree(freeze_dir)

    freeze_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = freeze_dir.parent / f".{freeze_dir.name}.staging.{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    try:
        aggregate_dir = staging / "frozen_results" / "aggregated"
        for name in REQUIRED_AGGREGATES:
            copy_file(campaign_root / name, aggregate_dir / name)

        per_mach_root = staging / "frozen_results" / "per_mach"
        for entry in validation["per_mach"]:
            mach_name = str(entry["directory"])
            source = campaign_root / mach_name
            destination = per_mach_root / mach_name
            for name in (
                "spectral_points.csv",
                "solver_attempts.csv",
                "campaign_summary.json",
                "state_low.json",
                "state_high.json",
            ):
                copy_optional(source / name, destination / name)
            copy_file(
                source / "modes" / "mode_summary.csv",
                destination / "modes" / "mode_summary.csv",
            )
            copy_file(
                source / "modes" / "modes_compact_with_analytic_tails.npz",
                destination / "modes" / "modes_compact_with_analytic_tails.npz",
            )
            if include_checkpoints:
                copy_optional(
                    source / "modes" / "modes_checkpoint.sqlite",
                    destination / "modes" / "modes_checkpoint.sqlite",
                )

        provenance = staging / "provenance"
        copy_file(config_path, provenance / "dense_supersonic_campaign_config.json")
        scripts = (
            repo / "classic_supersonic/scripts/validation/run_dense_supersonic_campaign.py",
            repo / "classic_supersonic/scripts/validation/reconstruct_dense_supersonic_modes.py",
            repo / "classic_supersonic/scripts/validation/aggregate_dense_supersonic_campaign.py",
            repo / "classic_supersonic/scripts/validation/test_kappa_q_modulus_reconstruction.py",
        )
        for source in scripts:
            copy_optional(source, provenance / "scripts" / source.name)

        metadata = {
            "freeze_id": freeze_dir.name,
            "state": "prepared",
            "prepared_at": utc_now(),
            "source_repo": str(repo),
            "source_campaign_root": str(campaign_root),
            "include_operational_checkpoints": bool(include_checkpoints),
            "validation": validation,
        }
        (staging / "freeze_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            "# Dense classical supersonic reference - prepared freeze\n\n"
            f"Freeze ID: `{freeze_dir.name}`  \n"
            f"Prepared: `{metadata['prepared_at']}`  \n"
            f"Targets: `{validation['n_targets']}`  \n"
            f"Retained eigenpairs: `{validation['n_retained']}`  \n"
            f"Reconstructed modes: `{validation['n_modes']}`  \n\n"
            "The source results have been copied and validated. Generate the requested "
            "maps and all-modes PDF, then run this script with `--finalize-existing --seal`.\n",
            encoding="utf-8",
        )
        os.replace(staging, freeze_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Prepared freeze: {freeze_dir}")
    print(json.dumps(validation, indent=2, sort_keys=True))


def iter_manifest_files(root: Path) -> Iterable[Path]:
    excluded = {"manifest.csv", "SHA256SUMS.txt"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            yield path


def write_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in iter_manifest_files(root):
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "csv_rows": file_rows(path),
            }
        )
    manifest_path = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    checksum_lines = [f"{row['sha256']}  {row['path']}" for row in rows]
    (root / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return rows


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            mode = path.stat().st_mode
            if path.is_dir():
                path.chmod(mode & ~stat.S_IWGRP & ~stat.S_IWOTH & ~stat.S_IWUSR)
            elif path.is_file():
                path.chmod(mode & ~stat.S_IWGRP & ~stat.S_IWOTH & ~stat.S_IWUSR)
        except OSError:
            pass
    mode = root.stat().st_mode
    root.chmod(mode & ~stat.S_IWGRP & ~stat.S_IWOTH & ~stat.S_IWUSR)


def finalize_existing(*, freeze_dir: Path, seal: bool) -> None:
    if not freeze_dir.is_dir():
        raise FileNotFoundError(freeze_dir)
    missing = [name for name in REQUIRED_FINAL_ASSETS if not (freeze_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize; missing required assets: {missing}")

    metadata_path = freeze_dir / "freeze_metadata.json"
    metadata = load_json(metadata_path) if metadata_path.is_file() else {}
    metadata.update(
        {
            "state": "sealed" if seal else "finalized",
            "finalized_at": utc_now(),
            "required_assets": list(REQUIRED_FINAL_ASSETS),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    validation = metadata.get("validation", {})
    readme = (
        "# Dense classical supersonic reference - final freeze\n\n"
        f"Freeze ID: `{freeze_dir.name}`  \n"
        f"Finalized: `{metadata['finalized_at']}`  \n"
        f"Targets: `{validation.get('n_targets', 'unknown')}`  \n"
        f"Retained eigenpairs: `{validation.get('n_retained', 'unknown')}`  \n"
        f"Reconstructed modes: `{validation.get('n_modes', 'unknown')}`  \n\n"
        "## Canonical assets\n\n"
        "- `classical_supersonic_maps/cr_map.pdf`\n"
        "- `classical_supersonic_maps/ci_map.pdf`\n"
        "- `classical_supersonic_maps/omega_i_map.pdf`\n"
        "- `classical_supersonic_maps/neutral_curve.pdf`\n"
        "- `classical_supersonic_maps/retained_point_mask.pdf`\n"
        "- `classical_supersonic_maps/classical_supersonic_dense_reference.csv`\n"
        "- `classical_supersonic_all_modes.pdf`\n\n"
        "The `frozen_results/` directory contains the canonical source tables and one "
        "compressed modal NPZ per Mach. `manifest.csv` and `SHA256SUMS.txt` provide "
        "file-level provenance and integrity checks.\n"
    )
    (freeze_dir / "README.md").write_text(readme, encoding="utf-8")
    rows = write_manifest(freeze_dir)
    print(f"Finalized freeze: {freeze_dir}")
    print(f"Manifest files: {len(rows)}")
    if seal:
        make_read_only(freeze_dir)
        print("Freeze sealed read-only.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the completed dense classical supersonic campaign."
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/production/classical/dense_supersonic_campaign_config.json"),
    )
    parser.add_argument("--campaign-root", type=Path, default=None)
    parser.add_argument(
        "--freeze-root",
        type=Path,
        default=Path("assets/classic_supersonic"),
    )
    parser.add_argument(
        "--freeze-name",
        default="dense_kappa_q_campaign_v1_FINAL_FREEZE",
    )
    parser.add_argument("--expected-targets", type=int, default=None)
    parser.add_argument("--expected-retained", type=int, default=None)
    parser.add_argument("--expected-modes", type=int, default=None)
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--seal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    config_path = resolve_path(repo, args.config)
    config = load_json(config_path)
    campaign_root = (
        resolve_path(repo, args.campaign_root)
        if args.campaign_root is not None
        else resolve_path(repo, config["output_root"])
    )
    freeze_root = resolve_path(repo, args.freeze_root)
    freeze_dir = freeze_root / args.freeze_name

    if args.finalize_existing:
        finalize_existing(freeze_dir=freeze_dir, seal=args.seal)
    else:
        if args.seal:
            raise ValueError("Use --seal together with --finalize-existing after asset generation.")
        prepare_freeze(
            repo=repo,
            config_path=config_path,
            campaign_root=campaign_root,
            freeze_dir=freeze_dir,
            expected_targets=args.expected_targets,
            expected_retained=args.expected_retained,
            expected_modes=args.expected_modes,
            include_checkpoints=args.include_checkpoints,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
