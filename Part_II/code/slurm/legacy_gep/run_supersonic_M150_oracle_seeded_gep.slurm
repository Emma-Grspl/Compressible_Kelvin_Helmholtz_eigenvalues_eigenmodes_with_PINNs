#!/bin/bash
#SBATCH --job-name=M150_GEP_oracle
#SBATCH --output=assets/pinn_supersonic/hybrid_gep/local_M150_S4M0_v1/logs/oracle_gep_%j.out
#SBATCH --error=assets/pinn_supersonic/hybrid_gep/local_M150_S4M0_v1/logs/oracle_gep_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --account=fdb@v100

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd "$WORK/These_PINN_KH_RT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

HYBRID_ROOT="assets/pinn_supersonic/hybrid_gep/local_M150_S4M0_v1"

OUTPUT_DIR="$HYBRID_ROOT/classic_seeded/N301_L3_all_oracle"

mkdir -p "$OUTPUT_DIR"

srun python \
  code/scripts/benchmarks/benchmark_supersonic_M150_S4M0_gep.py \
  --seed-source classical \
  --alpha-set all \
  --mach 1.50 \
  --n-points 301 \
  --mapping-scale 3.0 \
  --xi-max 0.985 \
  --ci-weight 2.0 \
  --y-max 80 \
  --amplitude-floor 1e-6 \
  --spectral-reference \
    "assets/classic_supersonic/csv/modal_reconstruction/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/table_classical_supersonic_final_reference.csv" \
  --modal-reference \
    "experiments/modal_reconstruction/support/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/classical_supersonic_final_modes_long.csv.gz" \
  --output-dir "$OUTPUT_DIR" \
  --device cpu
