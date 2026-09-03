#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent
SHOWCASE = REPO / "assets" / "classic_supersonic" / "reference_v2"

SPECTRAL = PACKAGE / "data" / "spectral" / "supersonic_reference_v2_spectral.csv"
CONFIG = PACKAGE / "configs" / "reference_v2.yaml"
PLOTTING_CONFIG = PACKAGE / "configs" / "plotting_ci.yaml"
OUTPUT = PACKAGE / "provenance" / "manifest.json"


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def pdf_pages(path: Path) -> int | None:
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
        return len(PdfReader(str(path)).pages)
    except Exception:
        return None


def relative(path: Path) -> str:
    return str(path.relative_to(REPO))


def main() -> None:
    reference_config = yaml.safe_load(CONFIG.read_text())
    plotting_config = yaml.safe_load(PLOTTING_CONFIG.read_text())

    df = pd.read_csv(SPECTRAL)

    if "Mach" not in df.columns and "M" in df.columns:
        df = df.rename(columns={"M": "Mach"})

    points = (
        df.drop_duplicates(["Mach", "alpha"])
        .sort_values(["Mach", "alpha"])
        .reset_index(drop=True)
    )

    counts_by_mach = {
        f"{float(mach):.2f}": int(count)
        for mach, count in points.groupby("Mach").size().items()
    }

    status_counts = {
        str(status): int(count)
        for status, count
        in points["validation_status"].astype(str).value_counts().items()
    }

    showcase_files = []

    for path in sorted(SHOWCASE.rglob("*")):
        if not path.is_file():
            continue

        item = {
            "path": relative(path),
            "size_bytes": path.stat().st_size,
        }

        if path.suffix.lower() == ".pdf":
            item["pdf_pages"] = pdf_pages(path)

        showcase_files.append(item)

    manifest = {
        "name": "classical-supersonic-kelvin-helmholtz-reference",
        "version": (PACKAGE / "VERSION").read_text().strip(),
        "scientific_reference": {
            "n_spectral_points": int(len(points)),
            "n_modal_rows_raw": int(
                reference_config["reference"]["n_modal_rows_raw"]
            ),
            "n_modal_rows_tail_polished": int(
                reference_config["reference"]["n_modal_rows_tail_polished"]
            ),
            "mach_min": float(points["Mach"].min()),
            "mach_max": float(points["Mach"].max()),
            "fields": reference_config["reference"]["fields"],
            "primary_modal_dataset": reference_config["reference"][
                "primary_modal_dataset"
            ],
            "tail_polished_is_derived": bool(
                reference_config["reference"]["tail_polished_is_derived"]
            ),
            "counts_by_mach": counts_by_mach,
            "validation_status_counts": status_counts,
        },
        "blumen": {
            "data_path": relative(
                PACKAGE / "data" / "blumen" / "blumen_ci_digitized_points.csv"
            ),
            "representation": plotting_config["blumen"]["representation"],
            "connect_points": plotting_config["blumen"]["connect_points"],
        },
        "paths": {
            "package": relative(PACKAGE),
            "showcase": relative(SHOWCASE),
            "spectral": relative(SPECTRAL),
        },
        "showcase_files": showcase_files,
        "git": {
            "branch": git_output("branch", "--show-current"),
            "head": git_output("rev-parse", "HEAD"),
        },
        "known_limitations": [
            "Raw confirmed modal fields are the primary scientific reference.",
            "Tail-polished fields are a documented derived export.",
            "Weak oscillatory tails remain sensitive for selected points.",
            "Five points retain an explicit boundary flag.",
            "Digitized Blumen points are discrete and are not connected.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {relative(OUTPUT)}")
    print(f"Spectral points: {len(points)}")
    print(f"Showcase files: {len(showcase_files)}")


if __name__ == "__main__":
    main()
