#!/bin/bash
#SBATCH --output=assets/pinn_supersonic/logs/%x_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --account=fdb@v100

set -euo pipefail

CONFIG="${1:?Missing configuration path}"
OUTPUT_DIR="${2:?Missing output directory}"

module purge
module load pytorch-gpu/py3/2.1.1

cd "$WORK/These_PINN_KH_RT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "===== CALIBRATION M150 ====="
echo "job        : ${SLURM_JOB_ID}"
echo "node       : $(hostname)"
echo "config     : ${CONFIG}"
echo "output     : ${OUTPUT_DIR}"
echo "start      : $(date)"
echo

nvidia-smi

srun \
  python \
    code/scripts/training/train_local_supersonic_kappa_q_logamp.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --prefit-steps 500 \
    --modal-steps 1000 \
    --joint-steps 1000 \
    --spectral-lr 2e-4 \
    --modal-lr 1e-4 \
    --joint-spectral-lr 5e-5 \
    --joint-modal-lr 5e-5 \
    --n-interior 512 \
    --n-boundary 64 \
    --modal-points-per-mode 257 \
    --print-every 50 \
    --overwrite

echo
echo "end: $(date)"
