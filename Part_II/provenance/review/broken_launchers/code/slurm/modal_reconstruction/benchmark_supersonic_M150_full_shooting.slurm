#!/bin/bash
#SBATCH --job-name=M150_full_shoot
#SBATCH --output=assets/pinn_supersonic/hybrid_fixed_c/local_M150_S4M0_v1/logs/full_shooting_%A_%a.out
#SBATCH --error=assets/pinn_supersonic/hybrid_fixed_c/local_M150_S4M0_v1/logs/full_shooting_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=20:00:00
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

ROOT="assets/pinn_supersonic/hybrid_fixed_c/local_M150_S4M0_v1"

MODES=(pinn_local wide)
OUTPUTS=(
  pinn_local_corrected_all
  wide_shooting_all
)

MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="$ROOT/${OUTPUTS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p \
  "$OUTPUT_DIR" \
  "$ROOT/logs"

echo "MODE=$MODE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "START=$(date)"

srun python \
  scripts/dev/benchmark_supersonic_M150_full_shooting.py \
  --mode "$MODE" \
  --alpha-set all \
  --output-dir "$OUTPUT_DIR" \
  --mach 1.50 \
  --cr-half-width 0.040 \
  --ci-half-width 0.030 \
  --wide-cr-min 0.20 \
  --wide-cr-max 0.50 \
  --wide-ci-min 1e-6 \
  --wide-ci-max 0.12 \
  --max-iter 12 \
  --grid-size 4 \
  --tol 1e-7 \
  --match-y 1.0 \
  --mapping-scale 5.0 \
  --min-y-limit 10.0 \
  --max-y-limit 1200.0 \
  --y-limit-factor 10.0 \
  --save-fields

echo "END=$(date)"
