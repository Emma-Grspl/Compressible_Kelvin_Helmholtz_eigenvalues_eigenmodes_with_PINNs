#!/usr/bin/env python3
"""Validate the curated Part II article-figure package without recomputation."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image


PART_II_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PART_II_ROOT / "article" / "FIGURE_MANIFEST.csv"
ASSETS = PART_II_ROOT / "article" / "assets"
EXPECTED_COLUMNS = [
    "article_label",
    "filename",
    "canonical_source",
    "plot_script",
    "source_data",
    "sha256",
    "notes",
]
NO_SCRIPT_FOUND = "NO_SCRIPT_FOUND"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise RuntimeError(f"Unexpected manifest columns: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 25:
        raise RuntimeError(f"Expected 25 manifest rows, found {len(rows)}")

    expected = {row["filename"] for row in rows}
    actual = {path.name for path in ASSETS.glob("*.png")}
    if expected != actual:
        raise RuntimeError(
            f"Article asset mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    for row in rows:
        asset = ASSETS / row["filename"]
        source = PART_II_ROOT / row["canonical_source"]
        data = PART_II_ROOT / row["source_data"]
        plot_script = row["plot_script"]
        script = None if plot_script == NO_SCRIPT_FOUND else PART_II_ROOT / plot_script
        required_paths = [("asset", asset), ("canonical source", source), ("source data", data)]
        if script is not None:
            required_paths.append(("plot script", script))
        for label, path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing {label}: {path}")
        if sha256(asset) != row["sha256"] or sha256(source) != row["sha256"]:
            raise RuntimeError(f"Hash mismatch for {row['filename']}")
        with Image.open(asset) as image:
            image.verify()
    print(f"validated_figures={len(rows)}")


if __name__ == "__main__":
    main()
