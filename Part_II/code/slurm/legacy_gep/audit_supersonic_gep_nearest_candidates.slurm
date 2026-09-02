#!/bin/bash
#SBATCH --job-name=M150_GEP_near
#SBATCH --output=assets/pinn_supersonic/hybrid_gep/local_M150_S4M0_v1/logs/gep_nearest_%j.out
#SBATCH --error=assets/pinn_supersonic/hybrid_gep/local_M150_S4M0_v1/logs/gep_nearest_%j.err
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

srun python \
  code/scripts/audits/audit_supersonic_gep_nearest_candidates.py
