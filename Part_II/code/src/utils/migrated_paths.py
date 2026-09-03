from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def _migration_map(repository: str) -> dict[str, str]:
    manifest = Path(repository) / "provenance" / "FILE_CLASSIFICATION.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        return {
            row["old_path"]: row["new_path"]
            for row in csv.DictReader(stream)
        }


def resolve_migrated_path(
    repository: Path,
    source_relative_path: str,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve an original repository path through the migration manifest."""
    repository = repository.resolve()
    relative = _migration_map(str(repository)).get(source_relative_path)
    if relative is None:
        raise KeyError(f"Path absent from migration manifest: {source_relative_path}")
    destination = repository / relative
    if must_exist and not destination.exists():
        raise FileNotFoundError(destination)
    return destination
