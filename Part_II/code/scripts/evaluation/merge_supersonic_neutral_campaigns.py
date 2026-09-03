#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_boolean_column(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations to bool."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge and deduplicate supersonic neutral-boundary "
            "continuation traces."
        )
    )

    parser.add_argument(
        "--alpha-traces",
        nargs="+",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_columns = {
        "Mach",
        "alpha",
        "accepted",
        "stage1_mismatch",
        "stage2_mismatch",
    }

    frames: list[pd.DataFrame] = []

    for source_path in args.alpha_traces:
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing source trace: {source_path}"
            )

        frame = pd.read_csv(source_path)

        missing_columns = (
            required_columns
            - set(frame.columns)
        )

        if missing_columns:
            raise KeyError(
                f"{source_path}: missing columns "
                f"{sorted(missing_columns)}"
            )

        frame = frame.copy()
        frame["source_file"] = str(
            source_path
        )

        frames.append(frame)

    merged = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    merged["_accepted"] = (
        parse_boolean_column(
            merged["accepted"]
        )
    )

    merged["_stage1"] = pd.to_numeric(
        merged["stage1_mismatch"],
        errors="coerce",
    ).fillna(np.inf)

    merged["_stage2"] = pd.to_numeric(
        merged["stage2_mismatch"],
        errors="coerce",
    ).fillna(np.inf)

    merged["_quality"] = (
        merged["_stage1"]
        + merged["_stage2"]
    )

    merged["_Mach_key"] = (
        pd.to_numeric(
            merged["Mach"],
            errors="raise",
        )
        .round(8)
    )

    merged["_alpha_key"] = (
        pd.to_numeric(
            merged["alpha"],
            errors="raise",
        )
        .round(10)
    )

    number_rows_raw = int(
        len(merged)
    )

    # For duplicate (M, alpha) entries:
    #   1. prefer an accepted result;
    #   2. select the smallest numerical mismatch.
    merged = merged.sort_values(
        by=[
            "_Mach_key",
            "_alpha_key",
            "_accepted",
            "_quality",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
    )

    deduplicated = (
        merged
        .drop_duplicates(
            subset=[
                "_Mach_key",
                "_alpha_key",
            ],
            keep="first",
        )
        .drop(
            columns=[
                "_accepted",
                "_stage1",
                "_stage2",
                "_quality",
                "_Mach_key",
                "_alpha_key",
            ]
        )
        .sort_values(
            by=[
                "Mach",
                "alpha",
            ]
        )
        .reset_index(drop=True)
    )

    output_csv = (
        args.output_dir
        / "alpha_traces_merged.csv"
    )

    deduplicated.to_csv(
        output_csv,
        index=False,
    )

    report = {
        "source_files": [
            str(path)
            for path in args.alpha_traces
        ],
        "n_source_files": int(
            len(args.alpha_traces)
        ),
        "n_rows_raw": (
            number_rows_raw
        ),
        "n_rows_deduplicated": int(
            len(deduplicated)
        ),
        "n_duplicates_removed": int(
            number_rows_raw
            - len(deduplicated)
        ),
        "n_mach": int(
            deduplicated["Mach"]
            .round(8)
            .nunique()
        ),
        "mach_min": float(
            deduplicated["Mach"].min()
        ),
        "mach_max": float(
            deduplicated["Mach"].max()
        ),
        "n_accepted": int(
            parse_boolean_column(
                deduplicated["accepted"]
            ).sum()
        ),
    }

    report_path = (
        args.output_dir
        / "merge_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print(
        f"\nwrote: {output_csv}"
    )

    print(
        f"wrote: {report_path}"
    )


if __name__ == "__main__":
    main()
