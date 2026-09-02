#!/usr/bin/env python3
"""Repair/populate the curated PINN model tree from existing atlas manifests.

The operational catalogue used for plotting may not contain model paths.
This script searches all atlas CSV manifests for exact chart_id -> path
mappings, resolves the corresponding checkpoint, and materializes it under:

    pinn_subsonic/models/<chart_id>/model_state.pt

No research checkpoint is moved or deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


PATH_COLUMNS = ("path", "chart_path", "model_path", "checkpoint_path")
CHECKPOINT_NAMES = (
    "model_best.pt",
    "model_state.pt",
    "best_model.pt",
    "checkpoint_best.pt",
)
CONFIG_NAMES = (
    "config.yaml",
    "config.yml",
    "config.json",
    "args.json",
    "normalization.json",
    "ci_anchors.csv",
    "metrics.json",
    "README.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--catalog",
        default=(
            "assets/pinn_subsonic/local_atlas_v1/"
            "publication_assets_scientific_v2/data/"
            "atlas_catalog_operational.csv"
        ),
    )
    parser.add_argument("--code-dir", default="pinn_subsonic")
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_chart_id(value: object) -> str:
    return str(value).strip()


def candidate_paths(repo: Path, csv_path: Path, value: object) -> list[Path]:
    if pd.isna(value):
        return []

    raw = Path(str(value).strip()).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        repo / raw,
        csv_path.parent / raw,
        raw,
    ]

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def find_checkpoint(path: Path) -> Path | None:
    if path.is_file() and path.suffix == ".pt":
        return path

    if not path.is_dir():
        return None

    for name in CHECKPOINT_NAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate

    candidates = sorted(
        p for p in path.glob("*.pt")
        if p.is_file() and "optimizer" not in p.name.lower()
    )
    return candidates[0] if len(candidates) == 1 else None


def scan_manifest_mappings(repo: Path) -> tuple[dict[str, list[dict]], list[str]]:
    mapping: dict[str, list[dict]] = {}
    scanned: list[str] = []

    search_roots = [
        repo / "assets/pinn_subsonic",
        repo / "assets/pinn_subsonic/local_atlas_v1",
    ]

    csv_paths: set[Path] = set()
    for root in search_roots:
        if root.is_dir():
            csv_paths.update(root.rglob("*.csv"))

    for csv_path in sorted(csv_paths):
        try:
            header = pd.read_csv(csv_path, nrows=0)
        except Exception:
            continue

        if "chart_id" not in header.columns:
            continue

        path_columns = [c for c in PATH_COLUMNS if c in header.columns]
        if not path_columns:
            continue

        scanned.append(str(csv_path.relative_to(repo)))

        try:
            frame = pd.read_csv(
                csv_path,
                usecols=["chart_id", *path_columns],
            )
        except Exception:
            continue

        for _, row in frame.iterrows():
            chart_id = normalise_chart_id(row["chart_id"])
            if not chart_id or chart_id.lower() in {"nan", "none"}:
                continue

            for column in path_columns:
                for path in candidate_paths(repo, csv_path, row[column]):
                    checkpoint = find_checkpoint(path)
                    if checkpoint is None:
                        continue

                    record = {
                        "chart_id": chart_id,
                        "checkpoint": checkpoint.resolve(),
                        "chart_dir": checkpoint.parent.resolve(),
                        "manifest": csv_path.resolve(),
                        "path_column": column,
                    }
                    records = mapping.setdefault(chart_id, [])
                    if all(
                        existing["checkpoint"] != record["checkpoint"]
                        for existing in records
                    ):
                        records.append(record)

    return mapping, scanned


def materialize(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())


def copy_configs(
    chart_dir: Path,
    destination: Path,
    dry_run: bool,
) -> list[str]:
    copied: list[str] = []
    for name in CONFIG_NAMES:
        src = chart_dir / name
        if not src.is_file():
            continue

        copied.append(name)
        if not dry_run:
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, destination / name)
    return copied


def choose_mapping(records: list[dict]) -> tuple[dict | None, str]:
    if not records:
        return None, "missing"

    unique_checkpoints = {
        str(record["checkpoint"]): record
        for record in records
    }
    candidates = list(unique_checkpoints.values())

    if len(candidates) == 1:
        return candidates[0], "resolved"

    # Prefer the canonical model_best.pt when duplicate manifests point to
    # several checkpoint variants for the same chart.
    model_best = [
        record for record in candidates
        if record["checkpoint"].name == "model_best.pt"
    ]
    if len(model_best) == 1:
        return model_best[0], "resolved_prefer_model_best"

    return None, "ambiguous"


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = repo / catalog_path

    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)

    code_dir = repo / args.code_dir
    model_root = code_dir / "models"
    config_root = code_dir / "configs/charts"
    manifest_root = code_dir / "manifests"

    if not args.dry_run:
        model_root.mkdir(parents=True, exist_ok=True)
        config_root.mkdir(parents=True, exist_ok=True)
        manifest_root.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_csv(catalog_path)
    if "chart_id" not in catalog.columns:
        raise KeyError(f"{catalog_path} has no chart_id column")

    mappings, scanned_manifests = scan_manifest_mappings(repo)

    rows: list[dict] = []
    for chart_id_raw in catalog["chart_id"]:
        chart_id = normalise_chart_id(chart_id_raw)
        chosen, resolution = choose_mapping(mappings.get(chart_id, []))

        row = {
            "chart_id": chart_id,
            "resolution": resolution,
            "checkpoint_status": "missing",
            "checkpoint_source": "",
            "chart_source": "",
            "mapping_manifest": "",
            "model_destination": "",
            "checkpoint_sha256": "",
            "copied_config_files": "",
            "candidate_count": len(mappings.get(chart_id, [])),
        }

        if chosen is not None:
            checkpoint = Path(chosen["checkpoint"])
            chart_dir = Path(chosen["chart_dir"])
            destination = model_root / chart_id / "model_state.pt"

            materialize(checkpoint, destination, args.mode, args.dry_run)
            copied = copy_configs(
                chart_dir,
                config_root / chart_id,
                args.dry_run,
            )

            row.update(
                {
                    "checkpoint_status": "present",
                    "checkpoint_source": str(checkpoint),
                    "chart_source": str(chart_dir),
                    "mapping_manifest": str(chosen["manifest"]),
                    "model_destination": str(destination),
                    "checkpoint_sha256": sha256(checkpoint),
                    "copied_config_files": ",".join(copied),
                }
            )

            if not args.dry_run:
                metadata = {
                    "chart_id": chart_id,
                    "source_checkpoint": str(checkpoint),
                    "source_chart_directory": str(chart_dir),
                    "source_manifest": str(chosen["manifest"]),
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "materialization_mode": args.mode,
                    "copied_config_files": copied,
                }
                metadata_path = model_root / chart_id / "metadata.json"
                metadata_path.write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8",
                )

        rows.append(row)

    frame = pd.DataFrame(rows)
    if not args.dry_run:
        frame.to_csv(
            manifest_root / "model_manifest.csv",
            index=False,
        )
        (manifest_root / "model_resolution_report.json").write_text(
            json.dumps(
                {
                    "catalog": str(catalog_path),
                    "manifests_scanned": scanned_manifests,
                    "status_counts": (
                        frame["checkpoint_status"]
                        .value_counts()
                        .to_dict()
                    ),
                    "resolution_counts": (
                        frame["resolution"]
                        .value_counts()
                        .to_dict()
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("===== CURATED MODEL RESOLUTION =====")
    print(
        frame[
            [
                "chart_id",
                "checkpoint_status",
                "resolution",
                "candidate_count",
                "checkpoint_source",
            ]
        ].to_string(index=False)
    )
    print("\nCheckpoint status:")
    print(frame["checkpoint_status"].value_counts().to_string())
    print("\nResolution:")
    print(frame["resolution"].value_counts().to_string())
    print("\nManifest CSVs scanned:", len(scanned_manifests))

    unresolved = frame[frame["checkpoint_status"] != "present"]
    if not unresolved.empty:
        print("\nUnresolved charts:")
        print(
            unresolved[
                ["chart_id", "resolution", "candidate_count"]
            ].to_string(index=False)
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
