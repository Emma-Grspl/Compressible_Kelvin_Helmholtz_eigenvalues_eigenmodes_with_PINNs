#!/usr/bin/env python3
"""Compatibility entry point for `code/plots/scripts/pinn_subsonic/active_implementation/source_tree/scripts/plot_kh_subsonic_fixed_mach_mode_field_error_heatmaps.py`.

Scientific logic remains in the original script so both entry points stay aligned.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

REPO_ROOT = Path(__file__).resolve().parents[9]
SOURCE = REPO_ROOT / "code/plots/scripts/pinn_subsonic/active_implementation/source_tree/scripts/plot_kh_subsonic_fixed_mach_mode_field_error_heatmaps.py"


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
