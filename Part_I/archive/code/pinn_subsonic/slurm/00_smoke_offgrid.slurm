#!/bin/bash
#SBATCH --job-name=KH_OFFGRID_SMOKE
#SBATCH --output=slurm/log/%j_KH_OFFGRID_SMOKE.out
#SBATCH --error=slurm/log/%j_KH_OFFGRID_SMOKE.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_p1
#SBATCH -A fdb@cpu
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd "$WORK/These_PINN_KH_RT"
set +H

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python \
  scripts/dev/validate_subsonic_atlas_offgrid.py \
  --points-csv \
  assets/pinn_subsonic/local_atlas_v1/offgrid_validation_points_384.csv \
  --output-csv \
  assets/pinn_subsonic/local_atlas_v1/offgrid_validation_384/smoke_4_points.csv \
  --num-shards 1 \
  --shard-index 0 \
  --max-points 4
