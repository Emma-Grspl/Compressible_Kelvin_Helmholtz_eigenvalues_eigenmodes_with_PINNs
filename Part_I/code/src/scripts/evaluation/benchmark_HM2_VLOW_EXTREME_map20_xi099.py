#!/usr/bin/env python3
"""Compatibility entry point; implementation lives in benchmark_ETAEDGE_HM2B_map20_xi099.py."""

from pathlib import Path
import runpy

CANONICAL = Path(__file__).resolve().parent / "benchmark_ETAEDGE_HM2B_map20_xi099.py"

if __name__ == "__main__":
    runpy.run_path(str(CANONICAL), run_name="__main__")
