#!/bin/bash
#SBATCH --job-name=kh_runtime
#SBATCH --account=fdb@v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm/log/kh_runtime_%j.out
#SBATCH --error=slurm/log/kh_runtime_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load pytorch-gpu/py3/2.1.1

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ROOT="$PWD"
PAPER_OUT="$ROOT/assets/pinn_subsonic/paper_results_v1"

mkdir -p \
  "$PAPER_OUT/logs" \
  "$PAPER_OUT/tables" \
  "$PAPER_OUT/data"

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

nvidia-smi

python \
  code/src/scripts/utils/utils_benchmark_subsonic_runtime.py \
  --output-dir "$PAPER_OUT" \
  --device cuda \
  --n-points 301 \
  --fast-repeats 100 \
  --slow-repeats 10 \
  --load-repeats 5
