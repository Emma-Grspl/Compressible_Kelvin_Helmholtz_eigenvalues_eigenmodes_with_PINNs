from pathlib import Path

import pandas as pd
import yaml


PACKAGE = Path(__file__).resolve().parents[2]
REPO = PACKAGE.parent
SHOWCASE = REPO / "assets" / "classic_supersonic" / "reference_v2"

CI_FILES = [
    SHOWCASE / "ci" / "blumen_ci_digitized_only.png",
    SHOWCASE / "ci" / "blumen_ci_digitized_only.pdf",
    SHOWCASE / "ci" / "blumen_ci_with_reference_points.png",
    SHOWCASE / "ci" / "blumen_ci_with_reference_points.pdf",
]

MODE_FILES = [
    SHOWCASE / "modes" / "supersonic_reference_v2_modes_full_y.pdf",
    SHOWCASE / "modes" / "supersonic_reference_v2_tail_polish_review_full_y.pdf",
]

SPECTRAL_TABLE = (
    SHOWCASE
    / "tables"
    / "supersonic_reference_v2_spectral.csv"
)


def pdf_pages(path: Path) -> int:
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader

    return len(PdfReader(str(path)).pages)


def test_showcase_directories_exist():
    assert (SHOWCASE / "ci").is_dir()
    assert (SHOWCASE / "modes").is_dir()
    assert (SHOWCASE / "tables").is_dir()


def test_ci_assets_exist_and_are_nonempty():
    for path in CI_FILES:
        assert path.exists(), path
        assert path.stat().st_size > 0, path


def test_modal_assets_exist_and_are_nonempty():
    for path in MODE_FILES:
        assert path.exists(), path
        assert path.stat().st_size > 0, path


def test_modal_pdfs_have_expected_page_count():
    for path in MODE_FILES:
        assert pdf_pages(path) == 184, path


def test_showcase_spectral_table():
    df = pd.read_csv(SPECTRAL_TABLE)
    assert len(df.drop_duplicates(["Mach", "alpha"])) == 183


def test_blumen_plotting_configuration_is_scatter_only():
    config_path = PACKAGE / "configs" / "plotting_ci.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["blumen"]["representation"] == "scatter"
    assert config["blumen"]["connect_points"] is False
