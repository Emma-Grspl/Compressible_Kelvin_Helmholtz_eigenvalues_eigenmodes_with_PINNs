#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path("assets/classic_supersonic/validated_modal_points/rebuilt_aggregates_latest/curated_strict")
OUT = SRC / "final23"
OUT.mkdir(parents=True, exist_ok=True)

spec = pd.read_csv(SRC / "supersonic_reference_core_local_spectral_CURATED.csv")
mod = pd.read_csv(SRC / "supersonic_reference_core_local_modal_CURATED.csv")
fld = pd.read_csv(SRC / "supersonic_reference_core_local_modal_fields_CURATED_SINGLE_SOURCE.csv", low_memory=False)

# Points conservés. On retire M=1.4 alpha=0.175, 0.18125, 0.1875 à cause du saut de cr.
keep = [
    (1.2, 0.150000), (1.2, 0.175000), (1.2, 0.187500), (1.2, 0.200000), (1.2, 0.208333), (1.2, 0.216667),
    (1.3, 0.100000), (1.3, 0.125000), (1.3, 0.150000), (1.3, 0.162500), (1.3, 0.175000), (1.3, 0.183333), (1.3, 0.191667),
    (1.4, 0.125000), (1.4, 0.137500), (1.4, 0.150000), (1.4, 0.162500), (1.4, 0.168750),
    (1.5, 0.125000), (1.5, 0.137500), (1.5, 0.150000), (1.5, 0.156250), (1.5, 0.162500),
]

quarantine = [
    (1.4, 0.175000), (1.4, 0.181250), (1.4, 0.187500),
]

def select_points(df, pts):
    mask = np.zeros(len(df), dtype=bool)
    for M, a in pts:
        mask |= np.isclose(df["Mach"].astype(float), M, atol=1e-10) & np.isclose(df["alpha"].astype(float), a, atol=5e-7)
    return df[mask].copy()

spec_f = select_points(spec, keep).sort_values(["Mach", "alpha"])
mod_f = select_points(mod, keep).sort_values(["Mach", "alpha"])
fld_f = select_points(fld, keep).sort_values(["Mach", "alpha", "y"])

spec_q = select_points(spec, quarantine).sort_values(["Mach", "alpha"])
mod_q = select_points(mod, quarantine).sort_values(["Mach", "alpha"])
fld_q = select_points(fld, quarantine).sort_values(["Mach", "alpha", "y"])

spec_f.to_csv(OUT / "supersonic_reference_core_local_spectral_FINAL23.csv", index=False)
mod_f.to_csv(OUT / "supersonic_reference_core_local_modal_FINAL23.csv", index=False)
fld_f.to_csv(OUT / "supersonic_reference_core_local_modal_fields_FINAL23.csv", index=False)

spec_q.to_csv(OUT / "supersonic_reference_core_local_spectral_QUARANTINE_M14_highalpha.csv", index=False)
mod_q.to_csv(OUT / "supersonic_reference_core_local_modal_QUARANTINE_M14_highalpha.csv", index=False)
fld_q.to_csv(OUT / "supersonic_reference_core_local_modal_fields_QUARANTINE_M14_highalpha.csv", index=False)

print("[OK] wrote", OUT)
print("spectral final:", len(spec_f))
print("modal final   :", len(mod_f))
print("field rows    :", len(fld_f))
print("field groups  :", fld_f.groupby(["Mach", "alpha"]).ngroups)

print("\n=== FINAL23 MODAL POINTS ===")
print(mod_f[["Mach", "alpha", "reference_cr", "reference_ci", "reference_omega_i", "line_id"]].to_string(index=False))

print("\n=== QUARANTINE ===")
print(mod_q[["Mach", "alpha", "reference_cr", "reference_ci", "reference_omega_i", "line_id"]].to_string(index=False))
