#!/bin/bash
set -euo pipefail
REPO="${1:-${WORK:-}/These_PINN_KH_RT}"
REPO="$(cd "$REPO" && pwd)"
cd "$REPO"
ACCOUNT="${IDRPROJ:-fdb}@cpu"
mkdir -p classic_supersonic/reproducibility/logs

REPAIR="classic_supersonic/jobs/lowM_lowalpha_auto_source_repair.slurm"
MODES="classic_supersonic/jobs/lowM_lowalpha_modes_repair_all_machs.slurm"
AGG="classic_supersonic/jobs/lowM_lowalpha_repaired_aggregate.slurm"

R_RAW="$(sbatch --parsable --account="$ACCOUNT" --export=ALL,REPO="$REPO" "$REPAIR")"
R_ID="${R_RAW%%;*}"
M_RAW="$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:${R_ID}" --export=ALL,REPO="$REPO" "$MODES")"
M_ID="${M_RAW%%;*}"
A_RAW="$(sbatch --parsable --account="$ACCOUNT" --dependency="afterok:${M_ID}" --export=ALL,REPO="$REPO" "$AGG")"
A_ID="${A_RAW%%;*}"

printf 'Auto-source repair job : %s\n' "$R_ID"
printf 'All-Mach mode job      : %s (afterok:%s)\n' "$M_ID" "$R_ID"
printf 'Aggregation job        : %s (afterok:%s)\n' "$A_ID" "$M_ID"
printf 'Final log              : classic_supersonic/reproducibility/logs/kh_sup_lowM_agg_fix_%s.out\n' "$A_ID"
