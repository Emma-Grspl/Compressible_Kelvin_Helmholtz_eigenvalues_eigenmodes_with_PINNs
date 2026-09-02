"""Shared helper for publishing validated article assets without recomputation."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


PART_I_ROOT = Path(__file__).resolve().parents[2]


def publish_validated_asset(
    canonical_source: str,
    output_filename: str,
    argv: list[str] | None = None,
) -> Path:
    """Copy a validated project figure byte-for-byte into article/assets."""
    parser = argparse.ArgumentParser(
        description="Publish a validated processed figure without recomputing it."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PART_I_ROOT / "article" / "assets" / output_filename,
        help="Destination PNG (defaults to the curated article asset path).",
    )
    args = parser.parse_args(argv)

    source = PART_I_ROOT / canonical_source
    if not source.is_file():
        raise FileNotFoundError(f"Validated canonical asset not found: {source}")

    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination:
        shutil.copy2(source, destination)

    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    destination_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    if source_hash != destination_hash:
        raise RuntimeError("Published asset differs from its canonical validated source.")

    print(destination)
    print(f"sha256={destination_hash}")
    return destination
