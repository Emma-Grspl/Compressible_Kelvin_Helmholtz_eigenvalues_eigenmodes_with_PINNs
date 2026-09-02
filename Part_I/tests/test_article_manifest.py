from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image


PART_I_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PART_I_ROOT / "article/FIGURE_MANIFEST.csv"
ASSETS = PART_I_ROOT / "article/assets"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_article_manifest_has_exactly_17_readable_assets() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 17
    assert {row["filename"] for row in rows} == {
        path.name for path in ASSETS.glob("*.png")
    }

    for row in rows:
        asset = ASSETS / row["filename"]
        canonical = PART_I_ROOT / row["canonical_source"]
        assert asset.is_file()
        assert canonical.is_file()
        assert sha256(asset) == row["sha256"]
        assert sha256(canonical) == row["sha256"]
        with Image.open(asset) as image:
            image.verify()


def test_article_manifest_links_to_existing_entrypoints_and_data() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        assert (PART_I_ROOT / row["plot_script"]).is_file()
        assert (PART_I_ROOT / row["source_data"]).exists()
