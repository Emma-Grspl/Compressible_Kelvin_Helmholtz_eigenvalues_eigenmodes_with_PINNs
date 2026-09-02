#!/usr/bin/env python3
"""Compatibility entry point for `archive/code/scripts/run_kh_subsonic_pinn_2d_band_M05_M07.py`.

Scientific logic remains in the original script so both entry points stay aligned.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

REPO_ROOT = Path(__file__).resolve().parents[6]
SOURCE = REPO_ROOT / "archive/code/scripts/run_kh_subsonic_pinn_2d_band_M05_M07.py"


def main() -> None:
    repository_directory = str(REPO_ROOT / "code")
    if repository_directory not in sys.path:
        sys.path.insert(0, repository_directory)
    source_directory = str(SOURCE.parent)
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)
    runpy.run_path(str(SOURCE), run_name="__main__")


if __name__ == "__main__":
    main()
