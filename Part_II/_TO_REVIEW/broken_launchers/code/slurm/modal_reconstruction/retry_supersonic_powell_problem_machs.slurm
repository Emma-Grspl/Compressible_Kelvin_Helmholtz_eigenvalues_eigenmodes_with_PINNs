#!/bin/bash
#SBATCH --job-name=S4M0_retry
#SBATCH --output=assets/pinn_supersonic/hybrid_fixed_c/logs/retry_%A_%a.out
#SBATCH --error=assets/pinn_supersonic/hybrid_fixed_c/logs/retry_%A_%a.err
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

MANIFEST="assets/pinn_supersonic/csv/supersonic_general/hybrid_fixed_c/table_s4m0_powell_retry_problem_machs.tsv"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))

IFS=$'\t' read -r MACH TAG ALPHA_FILE OUTPUT_DIR < <(
    sed -n "${LINE_NUMBER}p" "$MANIFEST"
)

if [[ -z "${MACH:-}" || -z "${TAG:-}" ]]; then
    echo "Invalid manifest row."
    exit 2
fi

if [[ ! -s "$ALPHA_FILE" ]]; then
    echo "No retry alpha for M=${MACH}"
    exit 0
fi

ROOT="assets/pinn_supersonic/hybrid_fixed_c/local_M${TAG}_S4M0_v1"

mkdir -p "$OUTPUT_DIR"

mapfile -t ALPHAS < "$ALPHA_FILE"

echo "============================================================"
echo "POWELL WIDE RETRY"
echo "Mach   = ${MACH}"
echo "tag    = ${TAG}"
echo "alphas = ${ALPHAS[*]}"
echo "start  = $(date)"
echo "============================================================"

srun python \
  scripts/dev/benchmark_supersonic_M150_pinn_powell_spectrum.py \
  --audit-csv \
    "$ROOT/s4m0_audit/s4m0_modal_audit.csv" \
  --output-dir \
    "$OUTPUT_DIR" \
  --mach "$MACH" \
  --alpha-values \
    "${ALPHAS[@]}" \
  --cr-half-width 0.080 \
  --ci-half-width 0.060 \
  --maxiter 40 \
  --maxfev 240 \
  --xtol 1e-9 \
  --ftol 1e-13 \
  --match-y 1.0 \
  --mapping-scale 5.0 \
  --min-y-limit 10.0 \
  --max-y-limit 1200.0 \
  --y-limit-factor 10.0

echo
echo "end = $(date)"
