#!/bin/bash
#SBATCH --job-name=S4M0_grid
#SBATCH --output=assets/pinn_supersonic/logs/S4M0_grid_%A_%a.out
#SBATCH --error=assets/pinn_supersonic/logs/S4M0_grid_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --account=fdb@v100

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd "$WORK/These_PINN_KH_RT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

MANIFEST="assets/pinn_supersonic/csv/supersonic_general/hybrid_fixed_c/table_s4m0_training_missing.tsv"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))

IFS=$'\t' read -r MACH TAG CONFIG OUTPUT_DIR < <(
    sed -n "${LINE_NUMBER}p" "$MANIFEST"
)

if [[ -z "${TAG:-}" ]]; then
    echo "Could not read manifest row for task ${SLURM_ARRAY_TASK_ID}"
    exit 2
fi

echo "============================================================"
echo "S4M0 TRAINING"
echo "task       = ${SLURM_ARRAY_TASK_ID}"
echo "Mach       = ${MACH}"
echo "tag        = ${TAG}"
echo "config     = ${CONFIG}"
echo "output_dir = ${OUTPUT_DIR}"
echo "node       = $(hostname)"
echo "start      = $(date)"
echo "============================================================"

srun python \
  code/scripts/training/train_local_supersonic_kappa_q_logamp.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --prefit-steps 2000 \
  --modal-steps 4000 \
  --joint-steps 4000 \
  --spectral-lr 2e-4 \
  --modal-lr 1e-4 \
  --joint-spectral-lr 5e-5 \
  --joint-modal-lr 5e-5 \
  --n-interior 1024 \
  --n-boundary 96 \
  --modal-points-per-mode 257 \
  --print-every 100 \
  --overwrite

echo
echo "end = $(date)"
