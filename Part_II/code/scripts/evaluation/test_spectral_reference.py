from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPECTRAL = ROOT / "data" / "spectral" / "supersonic_reference_v2_spectral.csv"

EXPECTED_COUNTS = {
    1.10: 9,
    1.20: 14,
    1.25: 11,
    1.30: 13,
    1.33: 9,
    1.40: 14,
    1.50: 12,
    1.60: 6,
    1.70: 6,
    1.80: 49,
    1.90: 40,
}

EXPECTED_STATUSES = {
    "validated_core_stable_tail_sensitive": 89,
    "modal_spectral_validated_with_exported_fields": 44,
    "validated_visual_smallM_strict": 33,
    "validated_visual_smallM_tail_sensitive": 12,
    "validated_visual_smallM_strict_boundary_flag": 5,
}


def load_reference() -> pd.DataFrame:
    df = pd.read_csv(SPECTRAL)

    if "Mach" not in df.columns and "M" in df.columns:
        df = df.rename(columns={"M": "Mach"})

    return df


def test_spectral_file_exists():
    assert SPECTRAL.exists()


def test_unique_point_count():
    df = load_reference()
    points = df.drop_duplicates(["Mach", "alpha"])
    assert len(points) == 183


def test_no_duplicate_points():
    df = load_reference()
    assert not df.duplicated(["Mach", "alpha"]).any()


def test_required_columns():
    df = load_reference()

    required = {
        "Mach",
        "alpha",
        "cr",
        "ci",
        "validation_status",
    }

    assert required.issubset(df.columns)


def test_no_missing_spectral_values():
    df = load_reference()
    assert not df[["Mach", "alpha", "cr", "ci"]].isna().any().any()


def test_counts_by_mach():
    df = load_reference()

    observed = {
        round(float(mach), 2): int(count)
        for mach, count in df.groupby("Mach").size().items()
    }

    assert observed == EXPECTED_COUNTS


def test_validation_status_counts():
    df = load_reference()

    observed = {
        str(status): int(count)
        for status, count in df["validation_status"].value_counts().items()
    }

    assert observed == EXPECTED_STATUSES


def test_ci_is_nonnegative():
    df = load_reference()
    assert (df["ci"] >= 0).all()
