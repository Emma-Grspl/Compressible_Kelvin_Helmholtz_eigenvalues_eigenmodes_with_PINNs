from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PACKAGE = Path(__file__).resolve().parents[2]

CASE_NAME = "witness_M150_a01625"

SAMPLE = PACKAGE / f"data/samples/{CASE_NAME}_raw.parquet"
EXPECTED = PACKAGE / f"data/samples/{CASE_NAME}_expected.json"
CONFIG = PACKAGE / f"configs/reproducibility/{CASE_NAME}.yaml"

PHYSICAL_COLUMNS = [
    "p_real",
    "p_imag",
    "rho_real",
    "rho_imag",
    "u_real",
    "u_imag",
    "v_real",
    "v_imag",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_expected() -> dict:
    return json.loads(EXPECTED.read_text())


def load_sample() -> pd.DataFrame:
    return pd.read_parquet(SAMPLE)


def test_witness_files_exist():
    assert SAMPLE.exists()
    assert EXPECTED.exists()
    assert CONFIG.exists()


def test_witness_checksum():
    expected = load_expected()

    assert sha256(SAMPLE) == expected["modal"]["sample_sha256"]


def test_witness_coordinates():
    expected = load_expected()
    df = load_sample()

    mach = expected["point"]["Mach"]
    alpha = expected["point"]["alpha"]

    assert np.allclose(df["Mach"], mach, atol=1e-12, rtol=0)
    assert np.allclose(df["alpha"], alpha, atol=1e-12, rtol=0)


def test_witness_domain_covers_both_sides():
    df = load_sample()

    assert df["y"].min() < 0
    assert df["y"].max() > 0


def test_witness_physical_fields_are_finite():
    df = load_sample()

    assert set(PHYSICAL_COLUMNS).issubset(df.columns)

    values = df[PHYSICAL_COLUMNS].to_numpy(dtype=float)

    assert np.isfinite(values).all()


def test_witness_spectral_values():
    expected = load_expected()
    df = load_sample()

    for name in ["cr", "ci", "omega_i"]:
        assert name in df.columns

        values = df[name].dropna().unique()

        assert len(values) == 1
        assert np.isclose(
            values[0],
            expected["spectral"][name],
            atol=1e-13,
            rtol=0,
        )


def test_witness_omega_relation():
    expected = load_expected()

    alpha = expected["point"]["alpha"]
    ci = expected["spectral"]["ci"]
    omega_i = expected["spectral"]["omega_i"]

    assert np.isclose(
        omega_i,
        alpha * ci,
        atol=1e-13,
        rtol=0,
    )


def test_witness_config_matches_expected():
    expected = load_expected()
    config = yaml.safe_load(CONFIG.read_text())

    assert config["case"]["Mach"] == expected["point"]["Mach"]
    assert config["case"]["alpha"] == expected["point"]["alpha"]

    for name in ["cr", "ci", "omega_i"]:
        assert np.isclose(
            config["expected_spectral"][name],
            expected["spectral"][name],
            atol=1e-13,
            rtol=0,
        )
