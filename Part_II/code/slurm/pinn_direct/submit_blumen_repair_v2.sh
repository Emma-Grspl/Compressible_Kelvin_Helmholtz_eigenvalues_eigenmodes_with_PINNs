#!/bin/bash
set -euo pipefail
REPO="${1:-$PWD}"
cd "$REPO"
OLD_ASSETS="$PWD/assets/classic_supersonic/blumen_true_classical_isolines_v1"
REPAIR_ROOT="$PWD/classic_supersonic/reproducibility/results/blumen_true_classical_isolines_repair_v2"
FINAL_ASSETS="$PWD/assets/classic_supersonic/blumen_true_classical_isolines_v2"
mkdir -p classic_supersonic/reproducibility/logs

module purge
module load pytorch-gpu/py3/2.1.1
export PYTHONPATH="$PWD/classic_supersonic/scripts/validation:$PWD:${PYTHONPATH:-}"

python3 -u \
  classic_supersonic/scripts/validation/prepare_blumen_repair_targets.py \
  --asset-root "$OLD_ASSETS" \
  --work-root "$REPAIR_ROOT" \
  --overwrite

read -r N_POS N_NEU < <(
python3 - "$REPAIR_ROOT/repair_target_metadata.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
print(x['n_positive_failed'], x['n_neutral_targets'])
PY
)

(( N_POS > 0 )) || { echo 'No failed positive points to repair.'; exit 1; }
(( N_NEU > 0 )) || { echo 'No neutral points to repair.'; exit 1; }

P="$(sbatch --parsable --array="0-$((N_POS-1))%12" \
  classic_supersonic/jobs/blumen_repair_positive.slurm \
  "$PWD" "$REPAIR_ROOT")"
PID="${P%%;*}"

N="$(sbatch --parsable --array="0-$((N_NEU-1))%10" \
  classic_supersonic/jobs/blumen_repair_neutral.slurm \
  "$PWD" "$REPAIR_ROOT")"
NID="${N%%;*}"

A="$(sbatch --parsable --dependency="afterok:${PID}:${NID}" \
  classic_supersonic/jobs/blumen_repair_assets.slurm \
  "$PWD" "$OLD_ASSETS" "$REPAIR_ROOT" "$FINAL_ASSETS")"
AID="${A%%;*}"

echo "Positive repair array : $PID ($N_POS tasks)"
echo "Neutral repair array  : $NID ($N_NEU tasks)"
echo "Final asset job       : $AID (afterok:$PID:$NID)"
echo "Final log             : classic_supersonic/reproducibility/logs/kh_blumen_rep_assets_${AID}.out"
