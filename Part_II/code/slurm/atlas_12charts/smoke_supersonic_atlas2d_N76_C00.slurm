#!/bin/bash
#SBATCH --job-name=ATLAS_C00_smoke
#SBATCH --output=assets/pinn_supersonic/atlas2d_v1/N76/logs/smoke_C00_%j.out
#SBATCH --error=assets/pinn_supersonic/atlas2d_v1/N76/logs/smoke_C00_%j.err
#SBATCH --time=00:20:00
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

CONFIG="configs/pinn_supersonic/atlas2d_v1/N76/C00.json"

OUTPUT="assets/pinn_supersonic/atlas2d_v1/N76/smoke/C00"

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

echo "============================================================"
echo "ATLAS N76 C00 SMOKE"
echo "node=$(hostname)"
echo "start=$(date)"
echo "============================================================"

srun python \
  code/scripts/training/train_global_supersonic_kappa_q_logamp.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT" \
  --device cuda \
  --prefit-steps 2 \
  --modal-steps 2 \
  --joint-steps 2 \
  --spectral-lr 2e-4 \
  --modal-lr 1e-4 \
  --joint-spectral-lr 5e-5 \
  --joint-modal-lr 5e-5 \
  --n-interior 64 \
  --n-boundary 16 \
  --print-every 1 \
  --overwrite

echo
echo "SMOKE_FINISHED"
echo "end=$(date)"
