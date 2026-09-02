#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def first_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    by_lower = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        found = by_lower.get(candidate.lower())
        if found is not None:
            return found
    return None


def parse_level(value: object) -> float:
    text = str(value)
    match = re.search(
        r"(?:ci|c_i|ci_sup|level|value)\s*=\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        text,
    )
    return float(numbers[-1]) if numbers else math.nan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blumen-csv", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.blumen_csv.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    if work_root.exists() and args.overwrite:
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source)
    columns = [str(column) for column in frame.columns]

    mach_col = first_column(columns, ("Mach", "M", "mach", "mach_physical"))
    alpha_col = first_column(columns, ("alpha", "Alpha", "wavenumber"))
    ci_col = first_column(
        columns,
        ("blumen_ci", "target_ci", "ci_level", "ci_value", "ci", "c_i", "contour_level"),
    )
    curve_id_col = first_column(columns, ("curve_id", "curve", "line_id"))
    curve_label_col = first_column(columns, ("curve_label", "label", "series", "name"))
    family_col = first_column(columns, ("family", "dataset_family"))

    if mach_col is None or alpha_col is None:
        raise KeyError(
            f"Mach/alpha columns not found in {source}; observed={columns}"
        )

    out = frame.copy()
    out["source_row_id"] = np.arange(len(out), dtype=int)
    out["Mach"] = pd.to_numeric(out[mach_col], errors="coerce")
    out["alpha"] = pd.to_numeric(out[alpha_col], errors="coerce")

    if ci_col is not None:
        out["blumen_ci"] = pd.to_numeric(out[ci_col], errors="coerce")
    else:
        out["blumen_ci"] = np.nan

    out["curve_id"] = (
        out[curve_id_col].astype(str) if curve_id_col is not None else "unknown"
    )
    out["curve_label"] = (
        out[curve_label_col].astype(str) if curve_label_col is not None else ""
    )
    out["family"] = (
        out[family_col].astype(str) if family_col is not None else ""
    )

    missing_ci = ~np.isfinite(out["blumen_ci"])
    if missing_ci.any():
        out.loc[missing_ci, "blumen_ci"] = (
            out.loc[missing_ci, "curve_label"].map(parse_level)
        )

    out = out.dropna(subset=["Mach", "alpha", "blumen_ci"]).copy()
    out = out.loc[(out["Mach"] >= 1.0) & (out["alpha"] > 0.0)].copy()
    out = out.loc[
        ~out["family"].astype(str).str.lower().eq("cr_special")
    ].copy()

    out = out.sort_values("source_row_id").reset_index(drop=True)
    out["blumen_row_id"] = np.arange(len(out), dtype=int)
    out["task_index"] = out["blumen_row_id"]
    out["curve_key"] = [
        f"{curve_id}__{label}__ci_{ci:.12g}"
        for curve_id, label, ci in out[
            ["curve_id", "curve_label", "blumen_ci"]
        ].itertuples(index=False)
    ]

    manifest = work_root / "blumen_pointwise_manifest.csv"
    out.to_csv(manifest, index=False)

    metadata = {
        "source_csv": str(source),
        "manifest": str(manifest),
        "n_points": int(len(out)),
        "n_positive_ci": int((out["blumen_ci"] > 0.0).sum()),
        "n_neutral_ci": int(np.isclose(out["blumen_ci"], 0.0, atol=1e-14).sum()),
        "mach_min": float(out["Mach"].min()),
        "mach_max": float(out["Mach"].max()),
        "alpha_min": float(out["alpha"].min()),
        "alpha_max": float(out["alpha"].max()),
        "status": "PASS",
    }
    (work_root / "target_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== BLUMEN POINTWISE TARGETS ===")
    print(f"Points         : {len(out)}")
    print(f"Positive ci    : {metadata['n_positive_ci']}")
    print(f"Neutral ci     : {metadata['n_neutral_ci']}")
    print(f"Manifest       : {manifest}")
    print("TARGET STATUS  : PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
