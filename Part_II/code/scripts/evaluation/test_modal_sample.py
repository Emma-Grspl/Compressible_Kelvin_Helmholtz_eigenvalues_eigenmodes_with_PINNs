from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[2]
SAMPLE = PACKAGE / "data" / "modal" / "modal_sample_for_tests.parquet"

EXPECTED_MACHS = {
    1.10,
    1.20,
    1.25,
    1.30,
    1.33,
    1.40,
    1.50,
    1.60,
    1.70,
    1.80,
    1.90,
}

EXPECTED_VARIANTS = {
    "raw_confirmed",
    "tail_polished",
}

PHYSICAL_COLUMNS = [
    "Mach",
    "alpha",
    "y",
    "p_real",
    "p_imag",
    "rho_real",
    "rho_imag",
    "u_real",
    "u_imag",
    "v_real",
    "v_imag",
    "cr",
    "ci",
]


def load_sample() -> pd.DataFrame:
    return pd.read_parquet(SAMPLE)


def coordinate_keys(df: pd.DataFrame) -> set[tuple[float, float, float]]:
    return set(
        zip(
            df["Mach"].round(10),
            df["alpha"].round(10),
            df["y"].round(10),
        )
    )


def test_sample_exists_and_is_nonempty():
    assert SAMPLE.exists()
    assert SAMPLE.stat().st_size > 0


def test_required_columns():
    df = load_sample()

    required = {
        "dataset_variant",
        *PHYSICAL_COLUMNS,
    }

    assert required.issubset(df.columns)


def test_both_variants_are_present():
    df = load_sample()

    assert set(df["dataset_variant"].unique()) == EXPECTED_VARIANTS


def test_one_sample_point_per_mach_and_variant():
    df = load_sample()

    points = df[
        ["dataset_variant", "Mach", "alpha"]
    ].drop_duplicates()

    observed_machs = {
        round(float(value), 2)
        for value in points["Mach"].unique()
    }

    assert observed_machs == EXPECTED_MACHS

    counts = points.groupby("dataset_variant").size().to_dict()

    assert counts == {
        "raw_confirmed": 11,
        "tail_polished": 11,
    }


def test_raw_and_polished_grids_match():
    df = load_sample()

    raw = df[df["dataset_variant"] == "raw_confirmed"]
    polished = df[df["dataset_variant"] == "tail_polished"]

    assert coordinate_keys(raw) == coordinate_keys(polished)


def test_each_mode_contains_both_y_sides():
    df = load_sample()

    groups = df.groupby(
        ["dataset_variant", "Mach", "alpha"],
        sort=True,
    )

    for point, group in groups:
        assert group["y"].min() < 0, point
        assert group["y"].max() > 0, point


def test_physical_values_are_finite():
    df = load_sample()

    values = df[PHYSICAL_COLUMNS].to_numpy(dtype=float)

    assert np.isfinite(values).all()


def test_ci_is_nonnegative():
    df = load_sample()

    assert (df["ci"] >= 0).all()


def test_spectral_values_are_invariant():
    df = load_sample()

    keys = ["Mach", "alpha"]
    invariant_columns = [
        column
        for column in ["cr", "ci", "omega_i"]
        if column in df.columns
    ]

    raw = (
        df[df["dataset_variant"] == "raw_confirmed"]
        [keys + invariant_columns]
        .drop_duplicates(keys)
        .sort_values(keys)
        .reset_index(drop=True)
    )

    polished = (
        df[df["dataset_variant"] == "tail_polished"]
        [keys + invariant_columns]
        .drop_duplicates(keys)
        .sort_values(keys)
        .reset_index(drop=True)
    )

    assert np.allclose(
        raw[keys],
        polished[keys],
        rtol=0,
        atol=1e-12,
    )

    for column in invariant_columns:
        assert np.allclose(
            raw[column],
            polished[column],
            rtol=0,
            atol=1e-13,
        )
