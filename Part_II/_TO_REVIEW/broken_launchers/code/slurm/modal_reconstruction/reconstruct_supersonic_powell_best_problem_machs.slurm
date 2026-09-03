#!/bin/bash
#SBATCH --job-name=S4M0_modes_best
#SBATCH --output=assets/pinn_supersonic/hybrid_fixed_c/logs/modes_best_%A_%a.out
#SBATCH --error=assets/pinn_supersonic/hybrid_fixed_c/logs/modes_best_%A_%a.err
#SBATCH --time=03:00:00
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

MACHS=(1.05 1.15 1.20)
TAGS=(105 115 120)

MACH="${MACHS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

ROOT="assets/pinn_supersonic/hybrid_fixed_c/local_M${TAG}_S4M0_v1"
OUTPUT="$ROOT/pinn_powell_modes_best"

mkdir -p "$OUTPUT"

echo "============================================================"
echo "FINAL MODAL RECONSTRUCTION"
echo "Mach  = ${MACH}"
echo "tag   = ${TAG}"
echo "node  = $(hostname)"
echo "start = $(date)"
echo "============================================================"

srun python \
  scripts/dev/benchmark_supersonic_M150_fixed_c_modes.py \
  --source pinn \
  --alpha-set all \
  --audit-csv \
    "$ROOT/pinn_powell_best/s4m0_modal_audit_powell_best.csv" \
  --reference-csv \
    "experiments/modal_reconstruction/support/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/classical_supersonic_final_modes_long.csv.gz" \
  --output-dir \
    "$OUTPUT" \
  --mach "$MACH" \
  --match-y 1.0 \
  --mapping-scale 5.0 \
  --min-y-limit 10.0 \
  --max-y-limit 1200.0 \
  --y-limit-factor 10.0 \
  --save-fields

echo
echo "end = $(date)"
