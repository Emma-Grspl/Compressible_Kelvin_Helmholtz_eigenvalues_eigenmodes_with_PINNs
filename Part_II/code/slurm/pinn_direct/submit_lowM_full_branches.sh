#!/bin/bash
set -euo pipefail

REPO="${1:-$PWD}"
cd "$REPO"
mkdir -p classic_supersonic/reproducibility/logs

module purge
module load pytorch-gpu/py3/2.1.1
export PYTHONPATH="$PWD/classic_supersonic/scripts/validation:$PWD:${PYTHONPATH:-}"

python3 -u \
  classic_supersonic/scripts/validation/prepare_lowM_full_branches.py \
  --repo "$PWD"

SPEC_JOB="$(
  sbatch --parsable \
    classic_supersonic/jobs/lowM_full_branches_spectral.slurm \
    "$PWD"
)"
SPEC_ID="${SPEC_JOB%%;*}"

MODE_JOB="$(
  sbatch --parsable \
    --dependency="afterok:${SPEC_ID}" \
    classic_supersonic/jobs/lowM_full_branches_modes.slurm \
    "$PWD"
)"
MODE_ID="${MODE_JOB%%;*}"

AGG_JOB="$(
  sbatch --parsable \
    --dependency="afterok:${MODE_ID}" \
    classic_supersonic/jobs/lowM_full_branches_aggregate.slurm \
    "$PWD"
)"
AGG_ID="${AGG_JOB%%;*}"

echo "Low-M full spectral job : $SPEC_ID (2 tasks)"
echo "Low-M full mode job     : $MODE_ID (afterok:$SPEC_ID)"
echo "Low-M full aggregate job: $AGG_ID (afterok:$MODE_ID)"
echo "Final log               : classic_supersonic/reproducibility/logs/kh_lowM_full_agg_${AGG_ID}.out"
