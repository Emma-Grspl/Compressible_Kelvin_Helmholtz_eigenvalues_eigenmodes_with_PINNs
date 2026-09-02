#!/bin/bash
#SBATCH --job-name=N340_modes
#SBATCH --account=fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=slurm/log/N340_modes_%j.out
#SBATCH --error=slurm/log/N340_modes_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load pytorch-gpu/py3/2.1.1

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python \
  code/src/scripts/utils/build_N340_modal_assets.py \
  --device cuda
