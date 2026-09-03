from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("assets/classic_supersonic/shooting/_incoming_sparse_supersonic_expand1")
RAW = ROOT / "sparse_supersonic_expand1_raw_candidates.csv"
VALID = ROOT / "sparse_supersonic_expand1_valid_candidates.csv"
SELECTED = ROOT / "sparse_supersonic_expand1_selected_reference_points.csv"

OUT = ROOT / "_inspection"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)

for c in [
    "alpha", "Mach",
    "cr_init", "ci_init",
    "cr_final", "ci_final",
    "reference_cr", "reference_ci", "reference_omega_i",
    "stage1_mismatch", "stage2_mismatch",
]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

print("\n=== FILE ===")
print(RAW)

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print("\n".join(df.columns))

print("\n=== RAW COUNT BY MACH ===")
print(df.groupby("Mach").size().to_string())

print("\n=== RAW COUNT BY MACH/ALPHA ===")
print(df.groupby(["Mach", "alpha"]).size().to_string())

if "status" in df.columns:
    print("\n=== STATUS COUNTS ===")
    print(df["status"].fillna("<NA>").value_counts(dropna=False).to_string())

if "reason" in df.columns:
    print("\n=== REASON COUNTS ===")
    print(df["reason"].fillna("<NA>").value_counts(dropna=False).head(40).to_string())

if "reject_reasons" in df.columns:
    print("\n=== REJECT_REASONS COUNTS ===")
    print(df["reject_reasons"].fillna("<NA>").value_counts(dropna=False).head(60).to_string())

finite = df[df["cr_final"].notna() & df["ci_final"].notna()].copy()

# Best selon stage2 : c’est ce que tu veux pour une racine shooting propre.
best_stage2 = (
    finite
    .sort_values(["Mach", "alpha", "stage2_mismatch", "stage1_mismatch"], na_position="last")
    .groupby(["Mach", "alpha"], as_index=False)
    .head(1)
)

# Best selon stage1 : utile pour voir si une racine spectrale existe mais échoue au stage2.
best_stage1 = (
    finite
    .sort_values(["Mach", "alpha", "stage1_mismatch", "stage2_mismatch"], na_position="last")
    .groupby(["Mach", "alpha"], as_index=False)
    .head(1)
)

cols = [
    "alpha", "Mach",
    "cr_init", "ci_init",
    "cr_final", "ci_final",
    "stage1_mismatch", "stage2_mismatch",
    "accept", "status", "reason", "reject_reasons",
    "job_id", "launch_seed", "stem",
]
cols = [c for c in cols if c in df.columns]

best_stage2[cols].sort_values(["Mach", "alpha"]).to_csv(
    OUT / "best_by_stage2_per_Mach_alpha.csv", index=False
)

best_stage1[cols].sort_values(["Mach", "alpha"]).to_csv(
    OUT / "best_by_stage1_per_Mach_alpha.csv", index=False
)

print("\n=== BEST BY STAGE2 PER MACH/ALPHA ===")
print(best_stage2[cols].sort_values(["Mach", "alpha"]).to_string(index=False))

print("\n=== BEST BY STAGE1 PER MACH/ALPHA ===")
print(best_stage1[cols].sort_values(["Mach", "alpha"]).to_string(index=False))

# Comparaison directe : là où stage1 trouve quelque chose mais stage2 explose.
cmp_cols = [
    "alpha", "Mach",
    "cr_final_stage1", "ci_final_stage1",
    "stage1_mismatch_stage1", "stage2_mismatch_stage1",
    "cr_final_stage2", "ci_final_stage2",
    "stage1_mismatch_stage2", "stage2_mismatch_stage2",
    "job_id_stage1", "job_id_stage2",
]

a = best_stage1[["alpha", "Mach", "cr_final", "ci_final", "stage1_mismatch", "stage2_mismatch", "job_id"]].copy()
b = best_stage2[["alpha", "Mach", "cr_final", "ci_final", "stage1_mismatch", "stage2_mismatch", "job_id"]].copy()

a = a.rename(columns={
    "cr_final": "cr_final_stage1",
    "ci_final": "ci_final_stage1",
    "stage1_mismatch": "stage1_mismatch_stage1",
    "stage2_mismatch": "stage2_mismatch_stage1",
    "job_id": "job_id_stage1",
})
b = b.rename(columns={
    "cr_final": "cr_final_stage2",
    "ci_final": "ci_final_stage2",
    "stage1_mismatch": "stage1_mismatch_stage2",
    "stage2_mismatch": "stage2_mismatch_stage2",
    "job_id": "job_id_stage2",
})

cmp = a.merge(b, on=["alpha", "Mach"], how="outer")
cmp = cmp.sort_values(["Mach", "alpha"])

cmp[cmp_cols].to_csv(OUT / "compare_best_stage1_vs_stage2.csv", index=False)

print("\n=== COMPARE BEST STAGE1 VS BEST STAGE2 ===")
print(cmp[cmp_cols].to_string(index=False))

# Zones faibles : M=1.1–1.5 plus M=1.6 partiel.
weak = cmp[cmp["Mach"].isin([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])].copy()
weak.to_csv(OUT / "weak_zones_M110_to_M160.csv", index=False)

print("\n=== WEAK ZONES M=1.1 TO 1.6 ===")
print(weak[cmp_cols].to_string(index=False))

# Candidats suspects : très bon stage1 mais très mauvais stage2.
suspect = finite[
    (finite["stage1_mismatch"] < 1e-4)
    & (finite["stage2_mismatch"] > 1e-6)
].copy()

suspect = suspect.sort_values(["Mach", "alpha", "stage1_mismatch"])
suspect[cols].to_csv(OUT / "stage1_good_stage2_bad_candidates.csv", index=False)

print("\n=== STAGE1 GOOD BUT STAGE2 BAD ===")
if len(suspect):
    print(suspect[cols].to_string(index=False))
else:
    print("None")

# Candidats stage2 propres mais stage1 moyen.
stage2_good = finite[finite["stage2_mismatch"] <= 1e-8].copy()
stage2_good = stage2_good.sort_values(["Mach", "alpha", "stage2_mismatch"])
stage2_good[cols].to_csv(OUT / "stage2_good_candidates.csv", index=False)

print("\n=== STAGE2 GOOD CANDIDATES ===")
if len(stage2_good):
    print(stage2_good[cols].to_string(index=False))
else:
    print("None")

print("\n=== WROTE INSPECTION FILES ===")
for p in sorted(OUT.glob("*.csv")):
    print(p)
