from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PART_II_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PART_II_ROOT / "configs" / "CONFIG_MANIFEST.csv"
EXPECTED_COLUMNS = [
    "public_path",
    "runtime_source",
    "classification",
    "purpose",
    "sha256",
]
EXPECTED_CLASSIFICATIONS = {"production", "validation", "experiment"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_configuration_copies_match_runtime_sources() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == EXPECTED_COLUMNS
        rows = list(reader)

    assert len(rows) == 6
    assert {row["classification"] for row in rows} == EXPECTED_CLASSIFICATIONS
    for row in rows:
        public_copy = PART_II_ROOT / row["public_path"]
        runtime_source = PART_II_ROOT / row["runtime_source"]
        assert public_copy.is_file()
        assert runtime_source.is_file()
        assert sha256(public_copy) == row["sha256"]
        assert sha256(runtime_source) == row["sha256"]
