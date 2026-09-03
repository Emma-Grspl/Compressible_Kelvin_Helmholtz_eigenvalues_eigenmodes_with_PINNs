#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]

MANIFEST = PACKAGE / "provenance/manifest.json"
EXPECTED = (
    PACKAGE
    / "data/samples/witness_M150_a01625_expected.json"
)


def main() -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    if not EXPECTED.exists():
        raise FileNotFoundError(EXPECTED)

    manifest = json.loads(MANIFEST.read_text())
    expected = json.loads(EXPECTED.read_text())

    manifest["reproducibility_witness"] = {
        "name": expected["name"],
        "point": expected["point"],
        "spectral": expected["spectral"],
        "modal_sample": {
            "path": expected["modal"]["sample_path"],
            "sha256": expected["modal"]["sample_sha256"],
            "n_rows": expected["modal"]["n_rows"],
            "y_min": expected["modal"]["y_min"],
            "y_max": expected["modal"]["y_max"],
        },
        "numerical_reproduction": expected[
            "numerical_reproduction"
        ],
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("Updated:", MANIFEST)


if __name__ == "__main__":
    main()
