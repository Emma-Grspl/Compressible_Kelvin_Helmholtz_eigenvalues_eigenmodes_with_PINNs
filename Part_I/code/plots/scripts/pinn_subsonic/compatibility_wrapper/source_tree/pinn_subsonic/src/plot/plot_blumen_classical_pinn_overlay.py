#!/usr/bin/env python3
"""Canonical entry point for the Blumen/classical/PINN growth-rate overlay.

Inputs and CLI arguments are defined by the retained historical generator.
Outputs include ``SuppFig_Blumen_growth_rate_comparison.png`` and ``.pdf``.
Scientific plotting logic is delegated unchanged.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[9]
SOURCE = (
    REPO_ROOT
    / "code/plots/scripts/pinn_subsonic/curated_entrypoint/source_tree/pinn_subsonic/scripts/figures/plot_blumen_classical_pinn_overlay.py"
)


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
