#!/bin/bash
set -euo pipefail

REPO="${1:-$PWD}"
cd "$REPO"

WORK_ROOT="$PWD/classic_supersonic/reproducibility/results/blumen_pointwise_classical_v1"
ASSET_ROOT="$PWD/assets/classic_supersonic/blumen_pointwise_classical_v1"
BLUMEN_CSV="$PWD/article/tables/table_blumen_overlay_points_with_neutral.csv"

mkdir -p classic_supersonic/reproducibility/logs

module purge
module load pytorch-gpu/py3/2.1.1
export PYTHONPATH="$PWD/classic_supersonic/scripts/validation:$PWD:${PYTHONPATH:-}"

python3 -u \
  classic_supersonic/scripts/validation/prepare_blumen_pointwise_targets.py \
  --blumen-csv "$BLUMEN_CSV" \
  --work-root "$WORK_ROOT" \
  --overwrite

N_POINTS="$(
python3 - "$WORK_ROOT/target_metadata.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["n_points"])
PY
)"

POINT_RAW="$(
  sbatch --parsable \
    --array="0-$((N_POINTS-1))%12" \
    classic_supersonic/jobs/blumen_pointwise_classical.slurm \
    "$PWD" "$WORK_ROOT"
)"
POINT_ID="${POINT_RAW%%;*}"

ASSET_RAW="$(
  sbatch --parsable \
    --dependency="afterany:${POINT_ID}" \
    classic_supersonic/jobs/blumen_pointwise_assets.slurm \
    "$PWD" "$WORK_ROOT" "$ASSET_ROOT"
)"
ASSET_ID="${ASSET_RAW%%;*}"

echo "Pointwise array : $POINT_ID ($N_POINTS tasks)"
echo "Asset job       : $ASSET_ID (afterany:$POINT_ID)"
echo "Final log       : classic_supersonic/reproducibility/logs/kh_blumen_point_assets_${ASSET_ID}.out"
