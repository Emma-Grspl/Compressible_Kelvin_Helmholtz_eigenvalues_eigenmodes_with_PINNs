#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.shooting.solve_reconstruct_dense_supersonic_modes as reconstruction


def selected_spectral_points(path):
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    is_target = frame["is_target"].astype(str).str.lower().isin(("true", "1"))
    ci = pd.to_numeric(frame["ci"], errors="coerce")
    cr = pd.to_numeric(frame["cr"], errors="coerce")
    selected = frame[
        is_target
        & frame["status"].astype(str).isin(("converged", "anchor_converged"))
        & frame["direction"].astype(str).isin(("low", "high", "anchor"))
        & np.isfinite(cr)
        & np.isfinite(ci)
        & (ci > 0.0)
    ].copy()
    return selected.sort_values("alpha").drop_duplicates("alpha", keep="last")


reconstruction.selected_spectral_points = selected_spectral_points

if __name__ == "__main__":
    raise SystemExit(reconstruction.main())
