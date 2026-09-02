#!/bin/bash
#SBATCH --job-name=M150_fixed_c
#SBATCH --output=assets/pinn_supersonic/hybrid_fixed_c/local_M150_S4M0_v1/logs/fixed_c_%A_%a.out
#SBATCH --error=assets/pinn_supersonic/hybrid_fixed_c/local_M150_S4M0_v1/logs/fixed_c_%A_%a.err
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

export FIXED_C_ROOT="assets/pinn_supersonic/hybrid_fixed_c/local_M150_S4M0_v1"

SOURCES=(oracle pinn)

SOURCE="${SOURCES[$SLURM_ARRAY_TASK_ID]}"

OUTPUT_DIR="$FIXED_C_ROOT/fixed_c_${SOURCE}_all"

mkdir -p \
  "$OUTPUT_DIR" \
  "$FIXED_C_ROOT/logs"

echo "SOURCE=$SOURCE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "START=$(date)"

srun python \
  scripts/dev/benchmark_supersonic_M150_fixed_c_modes.py \
  --source "$SOURCE" \
  --alpha-set all \
  --output-dir "$OUTPUT_DIR" \
  --mach 1.50 \
  --match-y 1.0 \
  --mapping-scale 5.0 \
  --min-y-limit 10.0 \
  --max-y-limit 1200.0 \
  --y-limit-factor 10.0 \
  --save-fields

echo "END=$(date)"
