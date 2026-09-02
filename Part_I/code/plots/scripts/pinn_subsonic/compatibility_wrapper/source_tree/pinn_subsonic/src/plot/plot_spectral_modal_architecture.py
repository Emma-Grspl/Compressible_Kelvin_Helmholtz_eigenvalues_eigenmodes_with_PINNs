#!/usr/bin/env python3
"""Canonical entry point for the spectral-modal architecture figure.

Inputs:
    No numerical input. The diagram is defined by the historical generator.
Outputs:
    ``Fig_spectral_modal_architecture_subsonic.png`` and ``.pdf`` below
    ``assets/pinn_subsonic/article/figures``.
Provenance:
    Delegates unchanged to
    ``code/plots/scripts/pinn_subsonic/curated_entrypoint/source_tree/pinn_subsonic/scripts/figures/plot_spectral_modal_architecture.py``.
Command:
    python -m plots.scripts.pinn_subsonic.compatibility_wrapper.source_tree.plots.scripts.pinn_subsonic.plot_spectral_modal_architecture
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[9]
SOURCE = (
    REPO_ROOT
    / "code/plots/scripts/pinn_subsonic/curated_entrypoint/source_tree/pinn_subsonic/scripts/figures/plot_spectral_modal_architecture.py"
)


def main() -> None:
    repository_directory = str(REPO_ROOT)
    if repository_directory not in sys.path:
        sys.path.insert(0, repository_directory)
    runpy.run_path(str(SOURCE), run_name="__main__")


if __name__ == "__main__":
    main()
