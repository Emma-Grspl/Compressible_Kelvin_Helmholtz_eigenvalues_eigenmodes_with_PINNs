#!/bin/bash
#SBATCH --job-name=S4M0_extremes
#SBATCH --output=assets/pinn_supersonic/hybrid_fixed_c/extremes_%A_%a.out
#SBATCH --error=assets/pinn_supersonic/hybrid_fixed_c/extremes_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --account=fdb@v100

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd "$WORK/These_PINN_KH_RT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TAGS=(110 190)
MACHS=(1.10 1.90)

TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
MACH="${MACHS[$SLURM_ARRAY_TASK_ID]}"

echo "TAG=$TAG"
echo "MACH=$MACH"

srun bash \
  scripts/dev/run_supersonic_s4m0_hybrid_validation.sh \
  "$TAG" \
  "$MACH"
