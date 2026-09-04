from __future__ import annotations


def test_generic_classical_solver_package_imports() -> None:
    import classical_solver

    assert classical_solver.__path__


def test_validated_classical_reference_package_imports() -> None:
    import classic_supersonic_reference

    assert classic_supersonic_reference.__version__ == "2.0.0"
