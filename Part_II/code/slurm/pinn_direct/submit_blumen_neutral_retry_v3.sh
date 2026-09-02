#!/bin/bash
set -euo pipefail

REPO="${1:-$PWD}"
cd "$REPO"

REPAIR_ROOT="$PWD/classic_supersonic/reproducibility/results/blumen_true_classical_isolines_repair_v2"
OLD_ASSETS="$PWD/assets/classic_supersonic/blumen_true_classical_isolines_v1"
FINAL_ASSETS="$PWD/assets/classic_supersonic/blumen_true_classical_isolines_v2"

test -f "$REPAIR_ROOT/neutral_repair_manifest.csv" || {
  echo "Missing neutral repair manifest: $REPAIR_ROOT/neutral_repair_manifest.csv"
  exit 1
}

mkdir -p classic_supersonic/reproducibility/logs

N="$(
  sbatch --parsable \
    --array="4,8,10,13,15,18,30-33%4" \
    classic_supersonic/jobs/blumen_repair_neutral.slurm \
    "$PWD" "$REPAIR_ROOT"
)"
NID="${N%%;*}"

A="$(
  sbatch --parsable \
    --dependency="afterok:${NID}" \
    classic_supersonic/jobs/blumen_repair_assets.slurm \
    "$PWD" "$OLD_ASSETS" "$REPAIR_ROOT" "$FINAL_ASSETS"
)"
AID="${A%%;*}"

echo "Neutral retry array : $NID (tasks 4,8,10,13,15,18,30-33)"
echo "Final asset job     : $AID (afterok:$NID)"
echo "Neutral logs        : classic_supersonic/reproducibility/logs/kh_blumen_rep_neu_${NID}_<task>.out"
echo "Final log           : classic_supersonic/reproducibility/logs/kh_blumen_rep_assets_${AID}.out"
