from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PART_II_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PART_II_ROOT / "configs" / "CONFIG_MANIFEST.csv"
EXPECTED_COLUMNS = [
    "canonical_path",
    "classification",
    "purpose",
    "sha256",
    "legacy_path",
    "legacy_path_status",
]
EXPECTED_CLASSIFICATIONS = {"production", "validation", "experiment"}
EXPECTED_LEGACY_STATUSES = {
    "removed_phase_8b1",
    "removed_phase_8b2",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_configuration_manifest_preserves_config_content() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == EXPECTED_COLUMNS
        rows = list(reader)

    assert len(rows) == 6
    assert {row["classification"] for row in rows} == EXPECTED_CLASSIFICATIONS
    assert {row["legacy_path_status"] for row in rows} == EXPECTED_LEGACY_STATUSES
    for row in rows:
        canonical = PART_II_ROOT / row["canonical_path"]
        legacy = PART_II_ROOT / row["legacy_path"]
        assert canonical.is_file()
        assert sha256(canonical) == row["sha256"]
        if row["legacy_path_status"].startswith("removed_phase_"):
            assert not legacy.exists()
        else:
            assert legacy.is_file()
            assert sha256(legacy) == row["sha256"]
