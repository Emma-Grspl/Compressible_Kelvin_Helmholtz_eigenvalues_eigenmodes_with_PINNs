#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo / path


def tree_size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size
    return sum(item.lstat().st_size for item in path.rglob("*") if item.is_file() or item.is_symlink())


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive every direct child of the classical supersonic results directory "
            "except explicitly kept canonical results. Dry-run is the default."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("classic_supersonic/reproducibility/results"),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("classic_supersonic/reproducibility/archive"),
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=["dense_kappa_q_campaign_v1"],
        help="Direct child name to keep in place. Repeatable.",
    )
    parser.add_argument("--archive-name", default=None)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    results_root = resolve_path(repo, args.results_root)
    archive_root = resolve_path(repo, args.archive_root)
    if not results_root.is_dir():
        raise FileNotFoundError(results_root)

    keep = set(args.keep)
    candidates = [
        child
        for child in sorted(results_root.iterdir(), key=lambda p: p.name)
        if child.name not in keep and not child.name.startswith(".")
    ]
    entries: list[dict[str, Any]] = []
    for path in candidates:
        entries.append(
            {
                "name": path.name,
                "kind": "directory" if path.is_dir() else "file",
                "size_bytes": tree_size(path),
                "source": str(path),
            }
        )

    print(f"Results root: {results_root}")
    print(f"Kept in place: {sorted(keep)}")
    print(f"Archive candidates: {len(entries)}")
    for entry in entries:
        print(f"  {entry['kind']:9s} {human_size(entry['size_bytes']):>12s}  {entry['name']}")
    print(f"Total candidate size: {human_size(sum(item['size_bytes'] for item in entries))}")

    if not candidates:
        print("Nothing to archive.")
        return 0
    if not args.execute:
        print("DRY RUN ONLY. Re-run with --execute after reviewing this list.")
        return 0

    archive_root.mkdir(parents=True, exist_ok=True)
    name = args.archive_name or f"pre_final_dense_campaign_results_{utc_stamp()}"
    archive_path = archive_root / f"{name}.tar.gz"
    manifest_path = archive_root / f"{name}.manifest.json"
    checksum_path = archive_root / f"{name}.sha256"
    if archive_path.exists() or manifest_path.exists() or checksum_path.exists():
        raise FileExistsError(f"Archive output already exists for name {name}.")

    command = ["tar", "-C", str(results_root), "-czf", str(archive_path), "--"]
    command.extend(path.name for path in candidates)
    subprocess.run(command, check=True)
    subprocess.run(["tar", "-tzf", str(archive_path)], check=True, stdout=subprocess.DEVNULL)

    digest = sha256_file(archive_path)
    checksum_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_results_root": str(results_root),
        "kept_in_place": sorted(keep),
        "archive": str(archive_path),
        "sha256": digest,
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Remove only after tar listing verification and checksum creation.
    for path in candidates:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()

    print(f"Archive created and verified: {archive_path}")
    print(f"Checksum: {checksum_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Removed {len(candidates)} archived entries from {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
