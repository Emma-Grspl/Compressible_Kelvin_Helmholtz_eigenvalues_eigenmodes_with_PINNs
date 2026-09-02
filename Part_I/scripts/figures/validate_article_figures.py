"""Validate the curated Part I article figure package and its manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image


PART_I_ROOT = Path(__file__).resolve().parents[2]
ARTICLE_DIR = PART_I_ROOT / "article"
ASSET_DIR = ARTICLE_DIR / "assets"
MANIFEST = ARTICLE_DIR / "FIGURE_MANIFEST.csv"
EXPECTED_COLUMNS = [
    "article_label",
    "filename",
    "canonical_source",
    "plot_script",
    "source_data",
    "sha256",
    "notes",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise RuntimeError(f"Unexpected manifest columns: {reader.fieldnames}")
        rows = list(reader)

    if len(rows) != 17:
        raise RuntimeError(f"Expected 17 manifest rows, found {len(rows)}")

    expected = {row["filename"] for row in rows}
    actual = {path.name for path in ASSET_DIR.iterdir() if path.is_file()}
    if expected != actual:
        raise RuntimeError(
            f"Article asset mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    for row in rows:
        filename = row["filename"]
        if not filename.startswith("Fig") or not filename.endswith(".png"):
            raise RuntimeError(f"Invalid article filename: {filename}")

        asset = ASSET_DIR / filename
        canonical = PART_I_ROOT / row["canonical_source"]
        plot_script = PART_I_ROOT / row["plot_script"]
        source_data = PART_I_ROOT / row["source_data"]
        for label, path in (
            ("asset", asset),
            ("canonical source", canonical),
            ("plot script", plot_script),
            ("source data", source_data),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing {label} for {filename}: {path}")

        if not asset.stat().st_size:
            raise RuntimeError(f"Empty article asset: {asset}")
        actual_hash = sha256(asset)
        if actual_hash != row["sha256"]:
            raise RuntimeError(f"Manifest hash mismatch for {filename}")
        if sha256(canonical) != actual_hash:
            raise RuntimeError(f"Canonical source mismatch for {filename}")

        with Image.open(asset) as image:
            image.verify()

    print(f"validated_figures={len(rows)}")
    print(f"manifest={MANIFEST}")


if __name__ == "__main__":
    main()
