#!/bin/bash
#SBATCH --job-name=KH_CONT_SMOKE
#SBATCH --output=slurm/log/%A_%a_KH_CONT_SMOKE.out
#SBATCH --error=slurm/log/%A_%a_KH_CONT_SMOKE.err
#SBATCH --array=0-2
#SBATCH --time=20:00:00
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
  scripts/dev/continue_subsonic_offgrid_neutral_N401.py \
  --targets-csv \
  assets/pinn_subsonic/local_atlas_v1/offgrid_validation_384/offgrid_continuation_smoke_targets.csv \
  --target-index "$SLURM_ARRAY_TASK_ID" \
  --output-dir \
  assets/pinn_subsonic/local_atlas_v1/offgrid_validation_384/continuation_smoke \
  --eta-low 0.92 \
  --eta-high 0.98 \
  --max-step 0.0025 \
  --N 401 \
  --mapping-scale 5 \
  --xi-max 0.98
