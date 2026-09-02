#!/usr/bin/env python3
from pathlib import Path
import re
import pandas as pd

ROOT = Path(".")
OUTDIR = Path("assets/classic_supersonic/KH_shoot_collect/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEARCH_DIRS = [
    Path("assets/classic_supersonic"),
    Path("assets/classic_subsonic"),
    Path("assets/pinn_supersonic"),
    Path("assets/pinn_subsonic"),
    Path("model_saved"),
]

csv_files = []
for d in SEARCH_DIRS:
    if d.exists():
        csv_files.extend(d.rglob("*.csv"))

rows = []
for p in sorted(set(csv_files)):
    sp = str(p)
    name = p.name.lower()
    parent = str(p.parent).lower()

    # Keep likely KH/shooting/spectral/modal files.
    if not any(k in sp.lower() for k in ["shoot", "kh", "supersonic", "subsonic", "modal", "spectral", "validated"]):
        continue

    info = {
        "path": sp,
        "name": p.name,
        "parent": str(p.parent),
        "size_bytes": p.stat().st_size,
        "n_rows": None,
        "n_cols": None,
        "columns": "",
        "has_alpha": False,
        "has_mach": False,
        "has_cr": False,
        "has_ci": False,
        "has_fields": False,
        "read_error": "",
    }

    try:
        df = pd.read_csv(p, nrows=5)
        info["n_rows"] = sum(1 for _ in open(p, "rb")) - 1
        info["n_cols"] = len(df.columns)
        info["columns"] = " | ".join(map(str, df.columns))

        cols = {c.lower(): c for c in df.columns}
        coltext = " ".join(cols)

        info["has_alpha"] = any(c in cols for c in ["alpha", "a"])
        info["has_mach"] = any(c in cols for c in ["mach", "m"])
        info["has_cr"] = any(c in cols for c in ["cr", "c_r", "c_real", "phase_speed_real"])
        info["has_ci"] = any(c in cols for c in ["ci", "c_i", "c_imag", "phase_speed_imag"])
        info["has_fields"] = any(k in coltext for k in ["p_real", "p_imag", "q_real", "q_imag", "u_real", "v_real"])
    except Exception as e:
        info["read_error"] = repr(e)

    rows.append(info)

inv = pd.DataFrame(rows)
out = OUTDIR / "KH_shoot_csv_inventory.csv"
inv.to_csv(out, index=False)

print("[OK] wrote", out)
print(inv[["path", "n_rows", "n_cols", "has_alpha", "has_mach", "has_cr", "has_ci", "has_fields", "read_error"]].to_string(index=False))
