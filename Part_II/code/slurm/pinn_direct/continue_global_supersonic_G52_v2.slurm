#!/bin/bash
#SBATCH --job-name=KH_G52_v2
#SBATCH --output=assets/pinn_supersonic/global/machsplit_G52_v1/logs/v2_%j.out
#SBATCH --error=assets/pinn_supersonic/global/machsplit_G52_v1/logs/v2_%j.err
#SBATCH --time=04:00:00
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

SOURCE="models_saved/pinn_direct/spectral_fullbatch_probe/best_validation_checkpoint.pt"

OUTPUT="assets/pinn_supersonic/global/machsplit_G52_v1/runs/G52_v2_fullbatch_joint"

echo "============================================================"
echo "GLOBAL SUPERSONIC G52-v2"
echo "52 spectral anchors"
echo "0 modal anchors"
echo "full-batch spectral supervision"
echo "round-robin Mach physics"
echo "test split sealed"
echo "start=$(date)"
echo "node=$(hostname)"
echo "============================================================"

srun python \
  code/scripts/shooting/solve_continue_global_supersonic_G52_v2.py \
  --config "$CONFIG" \
  --source-checkpoint "$SOURCE" \
  --output-dir "$OUTPUT" \
  --device cuda \
  --modal-steps 13000 \
  --joint-steps 13000 \
  --modal-lr 1e-4 \
  --joint-spectral-lr 2e-5 \
  --joint-modal-lr 5e-5 \
  --n-interior 1024 \
  --n-boundary 96 \
  --print-every-cycles 10

echo
echo "end=$(date)"
