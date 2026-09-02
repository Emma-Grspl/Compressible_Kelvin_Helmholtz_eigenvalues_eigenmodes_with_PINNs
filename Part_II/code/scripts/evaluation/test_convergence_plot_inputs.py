from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PACKAGE_ROOT / "scripts/convergence"
sys.path.insert(0, str(SCRIPT_DIR))

from plots.scripts.classic_supersonic.plot_classical_convergence import load_plot_frame  # noqa: E402


def test_incomplete_error_csv_is_rejected(tmp_path: Path):
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"case_id": ["subsonic_interior"], "abs_error_ci": [1e-3]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_plot_frame(path)


def test_empty_error_csv_is_rejected(tmp_path: Path):
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["case_id"]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_plot_frame(path)
