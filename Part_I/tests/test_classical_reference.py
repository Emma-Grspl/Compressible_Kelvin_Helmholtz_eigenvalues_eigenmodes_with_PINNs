from __future__ import annotations

import math

from src.scripts.classical.solve_robust_subsonic_shooting import (
    RobustSubsonicShootingSolver,
)


def test_robust_classical_reference_single_subsonic_point() -> None:
    result = RobustSubsonicShootingSolver(alpha=0.5, Mach=0.5).solve(
        primary_n_scan=21
    )

    assert result.success
    assert math.isfinite(result.ci)
    assert math.isfinite(result.omega_i)
    assert result.ci > 0.0
    assert result.omega_i > 0.0
    assert result.source.startswith("primary")
