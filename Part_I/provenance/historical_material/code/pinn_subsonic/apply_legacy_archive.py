#!/usr/bin/env python3
"""Apply the SHA-validated PINN-subsonic legacy archive plan.

The command is a dry run unless ``--apply`` is supplied. It never overwrites
an archive destination and does not delete data. Tracked files are moved with
``git mv``; untracked files are renamed on the same filesystem.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "pinn_subsonic"
ARCHIVE = REPO / "archive/pinn_subsonic_legacy"
PLAN = PACKAGE / "datas/atlas/Table_legacy_archive_plan.csv"
PROVENANCE = PACKAGE / "datas/atlas/Table_asset_provenance.csv"
MANIFEST = ARCHIVE / "reports/Table_archive_manifest.csv"

ARCHIVE_DIRECTORIES = (
    "scripts/training",
    "scripts/models",
    "scripts/plot",
    "scripts/validation",
    "scripts/ambiguous",
    "configs",
    "checkpoints",
    "figures/modes",
    "figures/ci",
    "figures/atlas",
    "figures/orphan_assets",
    "figures/external_references",
    "tables/modes",
    "tables/ci",
    "tables/atlas",
    "tables/orphan_tables",
    "bundles",
    "root_files",
    "logs",
    "metadata",
    "reports",
)
MANIFEST_COLUMNS = (
    "original_path",
    "archive_path",
    "canonical_equivalent",
    "action",
    "sha256_before",
    "sha256_after",
    "size_bytes",
    "compatibility_path",
    "dependency_status",
    "provenance_status",
    "reason",
    "validation_status",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_manifest(rows: list[dict[str, object]]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def tracked_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        check=True,
        stdout=subprocess.PIPE,
    )
    return {
        value.decode("utf-8")
        for value in result.stdout.split(b"\0")
        if value
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform validated moves. Without this flag only preflight runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_rows = [
        row for row in read_csv(PLAN) if row["action"] == "MOVE_TO_ARCHIVE"
    ]
    provenance = {
        row["asset_path"]: row["provenance_status"]
        for row in read_csv(PROVENANCE)
    }
    manifest_rows: list[dict[str, object]] = []

    for row in plan_rows:
        source = REPO / row["original_path"]
        destination = REPO / row["proposed_archive_path"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing planned source: {source}")
        if destination.exists():
            raise FileExistsError(f"Archive destination already exists: {destination}")
        actual_hash = digest(source)
        if actual_hash != row["sha256"]:
            raise RuntimeError(f"SHA-256 changed since audit: {source}")
        if row["imported_by"] or row["referenced_by"]:
            raise RuntimeError(f"Planned move still has dependencies: {source}")
        canonical = row["canonical_path"]
        if canonical:
            canonical_path = REPO / canonical
            if not canonical_path.is_file():
                raise FileNotFoundError(
                    f"Missing canonical equivalent for {source}: {canonical_path}"
                )
            if digest(canonical_path) != actual_hash:
                raise RuntimeError(
                    f"Canonical SHA-256 mismatch for {source}: {canonical_path}"
                )
        manifest_rows.append(
            {
                "original_path": row["original_path"],
                "archive_path": row["proposed_archive_path"],
                "canonical_equivalent": canonical,
                "action": "MOVE_TO_ARCHIVE",
                "sha256_before": actual_hash,
                "sha256_after": "",
                "size_bytes": row["size_bytes"],
                "compatibility_path": "",
                "dependency_status": "NO_ACTIVE_REFERENCE",
                "provenance_status": provenance.get(row["original_path"], ""),
                "reason": row["reason"],
                "validation_status": "PREFLIGHT_OK",
            }
        )

    print(f"Preflight passed for {len(manifest_rows)} files.")
    if not args.apply:
        print("Dry run only. Use --apply to perform the moves.")
        return

    for directory in ARCHIVE_DIRECTORIES:
        (ARCHIVE / directory).mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_rows)

    tracked = tracked_paths()
    for index, (plan_row, manifest_row) in enumerate(
        zip(plan_rows, manifest_rows), start=1
    ):
        source_relative = plan_row["original_path"]
        destination_relative = plan_row["proposed_archive_path"]
        source = REPO / source_relative
        destination = REPO / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source_relative in tracked:
            subprocess.run(
                ["git", "mv", "--", source_relative, destination_relative],
                cwd=REPO,
                check=True,
            )
        else:
            source.rename(destination)
        after = digest(destination)
        if after != manifest_row["sha256_before"]:
            raise RuntimeError(f"Post-move SHA-256 mismatch: {destination}")
        manifest_row["sha256_after"] = after
        manifest_row["validation_status"] = "MOVED_SHA256_OK"
        if index % 100 == 0:
            write_manifest(manifest_rows)

    write_manifest(manifest_rows)
    print(f"Archived {len(manifest_rows)} files with verified SHA-256.")


if __name__ == "__main__":
    main()
