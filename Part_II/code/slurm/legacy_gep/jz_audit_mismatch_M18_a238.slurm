#!/bin/bash
#SBATCH --job-name=kh_mismatch_M18a238
#SBATCH -A fdb@v100
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd $WORK/These_PINN_KH_RT
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python code/scripts/audits/audit_supersonic_ci_first_shooting_mismatch_map_1a94c44bbe.py \
  --mach 1.8 \
  --alpha 0.238 \
  --cr-min 0.90 \
  --cr-max 1.05 \
  --ci-min 0.0005 \
  --ci-max 0.008 \
  --n-cr 61 \
  --n-ci 61 \
  --output assets/classic_supersonic/extension_neutral_M180_M190_gep_shooting/mismatch_map_M18_a238.csv
