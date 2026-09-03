#!/bin/bash
#SBATCH --job-name=M150_pow_modes
#SBATCH --output=assets/pinn_supersonic/hybrid_fixed_c/local_M150_S4M0_v1/logs/pinn_powell_modes_%j.out
#SBATCH --error=assets/pinn_supersonic/hybrid_fixed_c/local_M150_S4M0_v1/logs/pinn_powell_modes_%j.err
#SBATCH --time=02:00:00
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

mkdir -p \
  "$ROOT/pinn_powell_modes_all" \
  "$ROOT/logs"

echo "START=$(date)"

srun python \
  scripts/dev/benchmark_supersonic_M150_fixed_c_modes.py \
  --source pinn \
  --alpha-set all \
  --audit-csv \
    "$ROOT/pinn_powell_all/s4m0_modal_audit_powell_corrected.csv" \
  --output-dir \
    "$ROOT/pinn_powell_modes_all" \
  --mach 1.50 \
  --match-y 1.0 \
  --mapping-scale 5.0 \
  --min-y-limit 10.0 \
  --max-y-limit 1200.0 \
  --y-limit-factor 10.0 \
  --save-fields

echo "END=$(date)"
