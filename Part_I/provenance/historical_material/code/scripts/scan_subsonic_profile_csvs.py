#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

roots = [
    "assets/pinn_subsonic/hybrid_pressure_pq_M050_a030_a070_switch_a040",
    "assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_discrete",
    "assets/pinn_subsonic/permach_pq_bootp_M050_a010070_cilearned_nodetach",
    "assets/pinn_subsonic/permach_pq_bootp_M050_a010070_TRUE_cilearned_nodetach",
    "assets/pinn_subsonic/stage1quater_pressure_path1500_xlim15",
]

for root in roots:
    root = Path(root)
    print("\n" + "=" * 120)
    print(root)
    if not root.exists():
        print("[missing]")
        continue

    for p in sorted(root.rglob("*.csv")):
        try:
            df = pd.read_csv(p, nrows=5, low_memory=False)
        except Exception as e:
            print("[read fail]", p, e)
            continue

        cols = list(df.columns)
        low = [c.lower() for c in cols]

        has_y = any(c in ["y", "yy"] for c in low)
        has_u = any("u" in c for c in low)
        has_ref_pred = any(("ref" in c or "pred" in c or "pinn" in c or "classic" in c or "target" in c) for c in low)

        if has_y and has_u:
            print("\n[PROFILE?]", p)
            print("shape_head:", df.shape)
            print("columns:")
            print(cols)
            print("u-like columns:")
            print([c for c in cols if "u" in c.lower()])
            if not has_ref_pred:
                print("[warn] u columns found, but no obvious ref/pred naming")
