#!/usr/bin/env python3
"""Compatibility entry point for `code/plots/scripts/pinn_subsonic/development_implementation/source_tree/scripts/dev/code/plots/scripts/pinn_subsonic/canonical_source/source_tree/plot_exact_mode_M050_alpha0500_local.py`.

Scientific logic remains in the original script so both entry points stay aligned.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = REPO_ROOT / "code/plots/scripts/pinn_subsonic/development_implementation/source_tree/scripts/dev/code/plots/scripts/pinn_subsonic/canonical_source/source_tree/plot_exact_mode_M050_alpha0500_local.py"


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
