#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

OUTDIR = Path("assets/classic_supersonic/KH_shoot_collect/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

refs = {
    "validated_modal_points": Path("assets/classic_supersonic/csv/modal_reconstruction/validated_modal_points/table_supersonic_validated_modal_points.csv"),
    "core_local_modal": Path("assets/classic_supersonic/shooting/supersonic_reference_core_local_modal.csv"),
    "core_local_spectral": Path("assets/classic_supersonic/shooting/supersonic_reference_core_local_spectral.csv"),
}

def standardize(path, label):
    df = pd.read_csv(path)

    needed = ["alpha", "Mach", "reference_cr", "reference_ci", "reference_omega_i"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"[FAIL] {path} missing columns: {missing}")

    out = df.copy()
    out["alpha"] = pd.to_numeric(out["alpha"], errors="coerce")
    out["Mach"] = pd.to_numeric(out["Mach"], errors="coerce")
    out["reference_cr"] = pd.to_numeric(out["reference_cr"], errors="coerce")
    out["reference_ci"] = pd.to_numeric(out["reference_ci"], errors="coerce")
    out["reference_omega_i"] = pd.to_numeric(out["reference_omega_i"], errors="coerce")
    out["source_reference_file"] = str(path)
    out["source_reference_label"] = label

    out = out.dropna(subset=["alpha", "Mach", "reference_cr", "reference_ci"])
    out = out.sort_values(["Mach", "alpha"]).reset_index(drop=True)

    return out

tables = {label: standardize(path, label) for label, path in refs.items()}

for label, df in tables.items():
    print(f"\n[{label}] shape={df.shape}")
    print(df[["Mach", "alpha", "reference_cr", "reference_ci", "reference_omega_i"]].to_string(index=False))

base_label = "validated_modal_points"
base = tables[base_label].copy()

print("\n[CONSISTENCY CHECK AGAINST validated_modal_points]")
for label, df in tables.items():
    m = base.merge(
        df,
        on=["Mach", "alpha"],
        suffixes=("_base", f"_{label}"),
        how="outer",
        indicator=True,
    )

    missing = m[m["_merge"] != "both"]
    if len(missing):
        print(f"[WARN] {label}: nonmatching rows:")
        print(missing[["Mach", "alpha", "_merge"]].to_string(index=False))

    both = m[m["_merge"] == "both"].copy()
    for col in ["reference_cr", "reference_ci", "reference_omega_i"]:
        err = np.nanmax(np.abs(both[f"{col}_base"] - both[f"{col}_{label}"]))
        print(f"{label:24s} max |Δ {col}| = {err:.3e}")

final = base.copy()
final = final[[
    "alpha",
    "Mach",
    "reference_cr",
    "reference_ci",
    "reference_omega_i",
    "line_id",
    "best_status",
    "best_stage1_mismatch",
    "best_stage2_mismatch",
    "best_spectral_success",
    "best_mode_success",
    "trusted_spectral",
    "trusted_modal",
    "valid_spectral_candidate",
    "valid_modal_candidate",
    "source_csv",
    "source_label",
    "source_group",
    "trusted_modal_bool",
    "source_reference_file",
    "source_reference_label",
]]

out = OUTDIR / "KH_shoot_FINAL23_curated_reference.csv"
final.to_csv(out, index=False)

print("\n[OK] wrote", out)
print("\n[FINAL23]")
print(final[["Mach", "alpha", "reference_cr", "reference_ci", "reference_omega_i", "best_status", "trusted_modal"]].to_string(index=False))
