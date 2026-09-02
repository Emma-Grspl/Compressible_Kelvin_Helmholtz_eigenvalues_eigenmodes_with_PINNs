#!/usr/bin/env python3
"""Canonical entry point for the retained joint GEP convergence diagnostic.

The historical implementation remains in ``scripts/dev`` because other
pipelines still reference it. This wrapper preserves its CLI and scientific
behavior unchanged.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = REPO_ROOT / "code/src/scripts/gep/selection/solve_joint_gep_n_convergence.py"


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
