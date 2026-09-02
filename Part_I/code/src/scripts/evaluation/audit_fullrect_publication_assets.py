#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path(
            "assets/pinn_subsonic/local_atlas_v1/publication_assets_fullrect_v1"
        ),
    )
    args = parser.parse_args()

    root = args.asset_dir
    manifest_path = root / "asset_manifest.csv"
    summary_path = root / "data" / "release_summary.json"
    validation_path = root / "data" / "validation_pointwise.csv"
    coverage_path = root / "data" / "coverage_grid.csv"

    required = [manifest_path, summary_path, validation_path, coverage_path]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(map(str, missing)))

    manifest = pd.read_csv(manifest_path)
    failures = []
    for _, row in manifest.iterrows():
        path = root / str(row["relative_path"])
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        expected_size = int(row["size_bytes"])
        if path.stat().st_size != expected_size:
            failures.append(f"size mismatch: {path}")
        actual_hash = sha256(path)
        if actual_hash != str(row["sha256"]):
            failures.append(f"sha256 mismatch: {path}")

    summary = json.loads(summary_path.read_text())
    validation = pd.read_csv(validation_path)
    coverage = pd.read_csv(coverage_path)

    checks = {
        "manifest_integrity": not failures,
        "release_validated": bool(summary.get("release_validated", False)),
        "validation_rows_384": len(validation) == 384,
        "validation_unique_384": validation["point_id"].nunique() == 384,
        "coverage_no_holes": int(coverage["coverage_count"].min()) >= 1,
        "pdf_figures_at_least_10": len(list((root / "figures").glob("*.pdf"))) >= 10,
        "png_figures_at_least_10": len(list((root / "figures").glob("*.png"))) >= 10,
    }

    print("===== PUBLICATION ASSET CHECKS =====")
    for name, passed in checks.items():
        print(f"{name:32s}: {'PASS' if passed else 'FAIL'}")

    if failures:
        print("\nManifest failures:")
        for failure in failures:
            print("-", failure)

    print("\nSummary:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not all(checks.values()):
        raise SystemExit("Publication asset validation failed")

    print("\nPUBLICATION ASSETS: VALIDATED")


if __name__ == "__main__":
    main()
