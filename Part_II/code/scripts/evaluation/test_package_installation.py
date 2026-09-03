from pathlib import Path
import inspect

import classic_supersonic_reference
from classic_supersonic_reference.solver.mstab17_supersonic_solver import (
    Mstab17SupersonicResult,
    Mstab17SupersonicSolver,
)


PACKAGE = Path(__file__).resolve().parents[2]


def test_version_matches_version_file():
    version_file = (
        PACKAGE / "VERSION"
    ).read_text().strip()

    assert (
        classic_supersonic_reference.__version__
        == version_file
    )


def test_solver_is_importable():
    assert inspect.isclass(
        Mstab17SupersonicSolver
    )

    assert inspect.isclass(
        Mstab17SupersonicResult
    )


def test_solver_constructor_public_parameters():
    parameters = inspect.signature(
        Mstab17SupersonicSolver
    ).parameters

    required = {
        "alpha",
        "Mach",
        "match_y",
        "rtol",
        "atol",
        "max_y_limit",
        "use_mapping",
        "mapping_scale",
    }

    assert required.issubset(parameters)


def test_solver_public_methods():
    required = {
        "solve",
        "solve_eigenvalue",
        "get_trajectories",
        "stage1_mismatch",
        "stage2_objective",
    }

    observed = {
        name
        for name, value in inspect.getmembers(
            Mstab17SupersonicSolver,
        )
        if callable(value)
    }

    assert required.issubset(observed)
