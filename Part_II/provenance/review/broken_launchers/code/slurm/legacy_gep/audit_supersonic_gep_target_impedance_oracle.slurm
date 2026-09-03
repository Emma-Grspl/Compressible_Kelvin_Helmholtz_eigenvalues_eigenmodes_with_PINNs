#!/bin/bash
#SBATCH --job-name=M150_GEP_Imp
#SBATCH --output=assets/pinn_supersonic/hybrid_gep/local_M150_S4M0_v1/logs/target_impedance_%A_%a.out
#SBATCH --error=assets/pinn_supersonic/hybrid_gep/local_M150_S4M0_v1/logs/target_impedance_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --account=fdb@v100

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd "$WORK/These_PINN_KH_RT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

N_VALUES=(301 401 501)
N_POINTS="${N_VALUES[$SLURM_ARRAY_TASK_ID]}"

echo "N_POINTS=$N_POINTS"
echo "START=$(date)"

srun python \
  scripts/dev/audit_supersonic_gep_target_impedance_oracle.py \
  --n-points "$N_POINTS" \
  --mapping-scale 3.0 \
  --xi-max 0.985 \
  --top-k 10

echo "END=$(date)"
