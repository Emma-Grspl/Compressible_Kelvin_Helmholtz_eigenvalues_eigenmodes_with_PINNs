#!/usr/bin/env python3
"""Canonical entry point for the retained three-asset repair generator.

The wrapper delegates to ``archive/code/scripts/dev/fix_three_subsonic_assets.py`` without
changing data selection, numerical calculations, or figure construction.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = REPO_ROOT / "archive/code/scripts/dev/fix_three_subsonic_assets.py"


def main() -> None:
    repository_directory = str(REPO_ROOT)
    if repository_directory not in sys.path:
        sys.path.insert(0, repository_directory)
    source_directory = str(SOURCE.parent)
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)
    runpy.run_path(str(SOURCE), run_name="__main__")


if __name__ == "__main__":
    main()
