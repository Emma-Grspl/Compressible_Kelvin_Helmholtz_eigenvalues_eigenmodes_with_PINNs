#!/bin/bash
#SBATCH --job-name=KH_OFFGRID_384
#SBATCH --output=slurm/log/%A_%a_KH_OFFGRID_384.out
#SBATCH --error=slurm/log/%A_%a_KH_OFFGRID_384.err
#SBATCH --array=0-15%8
#SBATCH --time=20:00:00
#SBATCH --partition=cpu_p1
#SBATCH -A fdb@cpu
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"

module purge
module load pytorch-gpu/py3/2.1.1
set +H
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

POINTS=assets/pinn_subsonic/local_atlas_v1/offgrid_validation_points_384.csv
OUTDIR=assets/pinn_subsonic/local_atlas_v1/offgrid_validation_384/shards

mkdir -p "$OUTDIR"

printf -v SHARD "%02d" "$SLURM_ARRAY_TASK_ID"

python \
  code/src/scripts/evaluation/evaluate_subsonic_atlas_offgrid.py \
  --points-csv "$POINTS" \
  --output-csv "$OUTDIR/offgrid_shard_${SHARD}.csv" \
  --num-shards 16 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --resume
