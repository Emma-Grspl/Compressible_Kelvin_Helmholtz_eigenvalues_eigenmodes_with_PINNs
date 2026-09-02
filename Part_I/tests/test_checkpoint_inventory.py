from __future__ import annotations

import csv
from pathlib import Path


PART_I_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PART_I_ROOT / "models_saved/CHECKPOINT_MANIFEST.csv"


def test_checkpoint_manifest_matches_public_checkpoint_files() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    discovered = sorted(
        path.relative_to(PART_I_ROOT).as_posix()
        for path in (PART_I_ROOT / "models_saved/production").rglob("*.pt")
    )
    recorded = sorted(row["checkpoint_path"] for row in rows)

    assert len(rows) == 50
    assert discovered == recorded
    assert sum(row["anchor_budget"] == "N340" for row in rows) == 49
    assert all(int(row["size_bytes"]) > 0 for row in rows)
    assert all(len(row["sha256"]) == 64 for row in rows)
