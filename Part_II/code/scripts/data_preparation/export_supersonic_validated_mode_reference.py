from __future__ import annotations

"""
Export the validated supersonic modal reference used by
code/scripts/data_preparation/prepare_build_supersonic_validated_modal_package.py.

Smoke tests:
python3 code/scripts/data_preparation/export_supersonic_validated_mode_reference.py --mach 1.6 --alpha 0.175 --output-csv /tmp/supersonic_validated_mode_M160_a0175.csv
python3 code/scripts/data_preparation/export_supersonic_validated_mode_reference.py --mach 1.6 --alpha 0.200 --output-csv /tmp/supersonic_validated_mode_M160_a0200.csv
"""

import argparse
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kh_supersonic_validated_mode_export_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "kh_supersonic_validated_mode_export_cache"))

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.data_preparation.prepare_build_supersonic_validated_modal_package import (  # noqa: E402
    load_supersonic_validated_mode_reference as _load_supersonic_validated_mode_reference,
)


def load_supersonic_validated_mode_reference(
    mach: float,
    alpha: float,
    *,
    tolerance_alpha: float | None = None,
    tolerance_mach: float | None = None,
) -> dict[str, object]:
    return _load_supersonic_validated_mode_reference(
        mach=float(mach),
        alpha=float(alpha),
        tolerance_alpha=tolerance_alpha,
        tolerance_mach=tolerance_mach,
    )


def export_supersonic_validated_mode_reference(
    mach: float,
    alpha: float,
    *,
    output_csv: str | Path,
    tolerance_alpha: float | None = None,
    tolerance_mach: float | None = None,
) -> dict[str, object]:
    payload = load_supersonic_validated_mode_reference(
        mach=float(mach),
        alpha=float(alpha),
        tolerance_alpha=tolerance_alpha,
        tolerance_mach=tolerance_mach,
    )
    arrays = payload["arrays"]
    columns: dict[str, object] = {
        "y": arrays["y"],
        "p_real": arrays["p_real"],
        "p_imag": arrays["p_imag"],
        "p_abs": arrays["p_abs"],
        "p_real_norm": arrays["p_real_norm"],
        "p_abs_norm": arrays["p_abs_norm"],
    }
    for key in ("q_real", "q_imag", "gamma_real", "gamma_imag"):
        if key in arrays:
            columns[key] = arrays[key]

    df = pd.DataFrame(columns)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    payload["output_csv"] = str(output_path)
    payload["csv_columns"] = list(df.columns)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the validated supersonic modal reference for the closest strict gold point."
    )
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--tolerance-alpha", type=float, default=None)
    parser.add_argument("--tolerance-mach", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = export_supersonic_validated_mode_reference(
        mach=float(args.mach),
        alpha=float(args.alpha),
        output_csv=args.output_csv,
        tolerance_alpha=args.tolerance_alpha,
        tolerance_mach=args.tolerance_mach,
    )
    print(payload["output_csv"])
    print(pd.DataFrame([payload["metadata"]]).to_string(index=False))
    print(f"csv_columns={payload['csv_columns']}")


if __name__ == "__main__":
    main()
