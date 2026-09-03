#!/usr/bin/env python3
"""Compute one subsonic classical Riccati-shooting eigenvalue on CPU.

Run from the Part_I directory with:
    python examples/01_classical_eigenvalue.py
"""

from __future__ import annotations

from pathlib import Path
import sys


PART_I_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART_I_ROOT / "code"))

from src.scripts.classical.solve_robust_subsonic_shooting import (  # noqa: E402
    RobustSubsonicShootingSolver,
)


def main() -> None:
    result = RobustSubsonicShootingSolver(alpha=0.5, Mach=0.5).solve(
        primary_n_scan=41
    )
    print(f"alpha={result.alpha:.3f}, Mach={result.Mach:.3f}")
    print(f"ci={result.ci:.8g}, omega_i={result.omega_i:.8g}")
    print(f"success={result.success}, source={result.source}")


if __name__ == "__main__":
    main()
