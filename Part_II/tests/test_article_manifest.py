from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image


PART_II_ROOT = Path(__file__).resolve().parents[1]


def test_curated_article_package_has_25_verified_pngs() -> None:
    manifest = PART_II_ROOT / "article" / "FIGURE_MANIFEST.csv"
    asset_dir = PART_II_ROOT / "article" / "assets"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 25
    assert {row["filename"] for row in rows} == {
        path.name for path in asset_dir.glob("*.png")
    }
    for row in rows:
        asset = asset_dir / row["filename"]
        canonical = PART_II_ROOT / row["canonical_source"]
        assert asset.is_file() and canonical.is_file()
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == row["sha256"]
        assert hashlib.sha256(canonical.read_bytes()).hexdigest() == row["sha256"]
        with Image.open(asset) as image:
            image.verify()
