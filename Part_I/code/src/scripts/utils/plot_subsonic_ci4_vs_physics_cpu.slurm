#!/bin/bash
#SBATCH --job-name=sub_ci4_plot
#SBATCH --output=assets/paper/logs/%x_%j.out
#SBATCH --error=assets/paper/logs/%x_%j.err
#SBATCH --account=fdb@cpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --hint=nomultithread

set -euo pipefail

BASE_DIR="${1:?Missing WORK_COMPARE argument}"
OUT_ROOT="${2:?Missing OUT argument}"

module purge
module load pytorch-gpu/py3/2.1.1

cd "$WORK/These_PINN_KH_RT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID}"

mkdir -p \
  "$MPLCONFIGDIR" \
  "$OUT_ROOT/logs" \
  "$OUT_ROOT/figures/single_case_ci4_vs_physics"

echo "===== JOB INFORMATION ====="
echo "job        : $SLURM_JOB_ID"
echo "node       : $(hostname)"
echo "repository : $PWD"
echo "base dir   : $BASE_DIR"
echo "output root: $OUT_ROOT"
echo "cpus       : ${SLURM_CPUS_PER_TASK:-unknown}"
echo "start      : $(date)"
echo

srun --cpu-bind=cores \
  python \
    code/src/scripts/utils/plot_ci4_vs_physics_modes.py \
    --base-dir "$BASE_DIR" \
    --output-dir "$OUT_ROOT/figures/single_case_ci4_vs_physics" \
    --num-alpha 401 \
    --mode-alphas 0.5 \
    --num-mode-alpha 1 \
    --n-y-pinn 2001 \
    --device cpu \
  2>&1 | tee \
    "$OUT_ROOT/logs/single_case_ci4_vs_physics_modes.log"

echo
echo "end: $(date)"
