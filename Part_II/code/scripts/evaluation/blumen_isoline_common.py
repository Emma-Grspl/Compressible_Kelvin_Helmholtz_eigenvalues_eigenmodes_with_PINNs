"""Shared helpers for positive Blumen isoline reconstruction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


POSITIVE_LEVELS = (0.01, 0.03, 0.05, 0.07, 0.10)
RAW_LEVEL_LABELS = {
    0.01: "0.01",
    0.03: "0.03",
    0.05: "0.05",
    0.07: "0.07",
    0.10: "0.1",
}


def attach_original_digitization_order(
    frame: pd.DataFrame,
    raw_blumen_csv: Path,
    *,
    matching_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Match each canonical point to its row in the original Blumen curve.

    The combined pointwise table is ordered by Mach and therefore interleaves
    the two sides of closed isolines.  The raw WebPlotDigitizer export stores
    one X/Y column pair per level and preserves the actual path order.  Exact
    coordinate matching recovers that order without geometric sorting.
    """

    raw = pd.read_csv(raw_blumen_csv, header=None)
    header = raw.iloc[0].astype(str).tolist()
    out = frame.copy()
    out["digitization_order"] = np.nan

    for level, raw_label in RAW_LEVEL_LABELS.items():
        matches = [index for index, value in enumerate(header) if value == raw_label]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one raw Blumen column pair for ci={level:g}, "
                f"found {len(matches)}."
            )
        x_column = matches[0]
        raw_curve = pd.DataFrame(
            {
                "Mach": pd.to_numeric(raw.iloc[2:, x_column], errors="coerce"),
                "alpha": pd.to_numeric(raw.iloc[2:, x_column + 1], errors="coerce"),
            }
        ).dropna()
        raw_curve = raw_curve.reset_index(drop=True)

        level_indices = out.index[
            np.isclose(
                pd.to_numeric(out["blumen_ci"], errors="coerce"),
                level,
                rtol=0.0,
                atol=1e-12,
            )
        ]
        used_orders: set[int] = set()
        for index in level_indices:
            Mach = float(out.at[index, "Mach"])
            alpha = float(out.at[index, "alpha"])
            distance = np.hypot(raw_curve["Mach"] - Mach, raw_curve["alpha"] - alpha)
            order = int(distance.idxmin())
            minimum_distance = float(distance.loc[order])
            if minimum_distance > matching_tolerance:
                raise ValueError(
                    f"No exact raw Blumen match for ci={level:g}, "
                    f"M={Mach:.12g}, alpha={alpha:.12g}; "
                    f"nearest distance={minimum_distance:.3e}."
                )
            if order in used_orders:
                raise ValueError(
                    f"Duplicate raw Blumen order {order} for ci={level:g}."
                )
            used_orders.add(order)
            out.at[index, "digitization_order"] = order

    if out["digitization_order"].isna().any():
        missing = out.loc[
            out["digitization_order"].isna(),
            ["Mach", "alpha", "blumen_ci"],
        ]
        raise ValueError(
            "Some positive Blumen points lack an original digitization order:\n"
            + missing.to_string(index=False)
        )
    out["digitization_order"] = out["digitization_order"].astype(int)
    return out
