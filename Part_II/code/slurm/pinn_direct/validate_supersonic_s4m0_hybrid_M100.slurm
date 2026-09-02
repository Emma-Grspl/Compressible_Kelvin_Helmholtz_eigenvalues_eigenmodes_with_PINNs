#!/bin/bash
#SBATCH --job-name=S4M0_val_M100
#SBATCH --output=assets/pinn_supersonic/hybrid_fixed_c/logs/M100_validation_%j.out
#SBATCH --error=assets/pinn_supersonic/hybrid_fixed_c/logs/M100_validation_%j.err
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

CHECKPOINT="models_saved/supersonic_general/long_S4/best_joint_checkpoint.pt"
RUNNER="scripts/dev/run_supersonic_s4m0_hybrid_validation.sh"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing M100 checkpoint:"
    echo "  $CHECKPOINT"
    exit 2
fi

if [[ ! -f "$RUNNER" ]]; then
    echo "Missing validation runner:"
    echo "  $RUNNER"
    exit 3
fi

echo "============================================================"
echo "M100 S4M0 HYBRID VALIDATION"
echo "job   = ${SLURM_JOB_ID}"
echo "node  = $(hostname)"
echo "start = $(date)"
echo "============================================================"

srun bash \
  "$RUNNER" \
  100 \
  1.00

echo
echo "end = $(date)"
