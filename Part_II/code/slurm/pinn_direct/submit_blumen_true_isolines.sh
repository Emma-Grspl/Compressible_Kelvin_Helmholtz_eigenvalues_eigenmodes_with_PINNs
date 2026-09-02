#!/bin/bash
set -euo pipefail
REPO="${1:-$PWD}"; cd "$REPO"
WORK_ROOT="$PWD/classic_supersonic/reproducibility/results/blumen_true_classical_isolines_v1"
ASSET_ROOT="$PWD/assets/classic_supersonic/blumen_true_classical_isolines_v1"
BLUMEN_CSV="$PWD/article/tables/table_blumen_overlay_points_with_neutral.csv"
mkdir -p classic_supersonic/reproducibility/logs
module purge; module load pytorch-gpu/py3/2.1.1
export PYTHONPATH="$PWD/classic_supersonic/scripts/validation:$PWD:${PYTHONPATH:-}"
python3 -u classic_supersonic/scripts/validation/prepare_blumen_slurm_targets.py --blumen-csv "$BLUMEN_CSV" --work-root "$WORK_ROOT" --overwrite
read -r N_POS N_NEU < <(python3 - "$WORK_ROOT/targets/target_metadata.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x["n_positive_curves"],x["n_neutral_points"])
PY
)
((N_POS>0)) || { echo "No positive curves"; exit 1; }; ((N_NEU>0)) || { echo "No neutral points"; exit 1; }
P="$(sbatch --parsable --array="0-$((N_POS-1))%6" classic_supersonic/jobs/blumen_true_positive_isolines.slurm "$PWD" "$WORK_ROOT")"; PID="${P%%;*}"
N="$(sbatch --parsable --array="0-$((N_NEU-1))%8" classic_supersonic/jobs/blumen_true_neutral_line.slurm "$PWD" "$WORK_ROOT")"; NID="${N%%;*}"
A="$(sbatch --parsable --dependency="afterok:${PID}:${NID}" classic_supersonic/jobs/blumen_true_isoline_assets.slurm "$PWD" "$WORK_ROOT" "$ASSET_ROOT")"; AID="${A%%;*}"
echo "Positive-isoline array : $PID ($N_POS tasks)"
echo "Neutral-line array     : $NID ($N_NEU tasks)"
echo "Asset job              : $AID (afterok:$PID:$NID)"
echo "Final log              : classic_supersonic/reproducibility/logs/kh_blumen_iso_assets_${AID}.out"
