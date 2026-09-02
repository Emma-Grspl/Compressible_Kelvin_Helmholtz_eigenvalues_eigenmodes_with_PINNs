#!/bin/bash
#SBATCH --job-name=KH_G52_2D
#SBATCH --output=assets/pinn_supersonic/global/machsplit_G52_v1/logs/train_%j.out
#SBATCH --error=assets/pinn_supersonic/global/machsplit_G52_v1/logs/train_%j.err
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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p \
  assets/pinn_supersonic/global/machsplit_G52_v1/logs

CONFIG="configs/pinn_supersonic/global_S4M0_machsplit_G52_v1.json"

OUTPUT="assets/pinn_supersonic/global/machsplit_G52_v1/runs/baseline_local_schedule"

echo "============================================================"
echo "GLOBAL SUPERSONIC S4M0"
echo "52 spectral anchors"
echo "0 modal anchors"
echo "Mach-split interpolation"
echo "TEST SPLIT SEALED"
echo "node=$(hostname)"
echo "start=$(date)"
echo "============================================================"

srun python \
  code/scripts/training/train_global_supersonic_kappa_q_logamp.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT" \
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
  --print-every 100 \
  --overwrite

echo
echo "end=$(date)"
