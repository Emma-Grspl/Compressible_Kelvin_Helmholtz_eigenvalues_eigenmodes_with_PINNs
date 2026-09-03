#!/bin/bash
#SBATCH --job-name=S4M0_val_grid
#SBATCH --output=assets/pinn_supersonic/hybrid_fixed_c/logs/val_grid_%A_%a.out
#SBATCH --error=assets/pinn_supersonic/hybrid_fixed_c/logs/val_grid_%A_%a.err
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

MANIFEST="assets/pinn_supersonic/csv/pinn_direct/hybrid_fixed_c/table_s4m0_validation_missing.tsv"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))

IFS=$'\t' read -r MACH TAG < <(
    sed -n "${LINE_NUMBER}p" "$MANIFEST"
)

if [[ -z "${MACH:-}" || -z "${TAG:-}" ]]; then
    echo "Could not read manifest row for task ${SLURM_ARRAY_TASK_ID}"
    exit 2
fi

CHECKPOINT="assets/pinn_supersonic/pilots/local_M${TAG}_sparse_v1/runs/long_S4/best_joint_checkpoint.pt"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint:"
    echo "$CHECKPOINT"
    exit 3
fi

echo "============================================================"
echo "S4M0 HYBRID VALIDATION"
echo "task = ${SLURM_ARRAY_TASK_ID}"
echo "Mach = ${MACH}"
echo "tag  = ${TAG}"
echo "node = $(hostname)"
echo "start = $(date)"
echo "============================================================"

srun bash \
  scripts/dev/run_supersonic_s4m0_hybrid_validation.sh \
  "$TAG" \
  "$MACH"

echo
echo "end = $(date)"
