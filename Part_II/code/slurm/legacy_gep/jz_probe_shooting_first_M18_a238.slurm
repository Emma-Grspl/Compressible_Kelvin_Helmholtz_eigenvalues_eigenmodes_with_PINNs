#!/bin/bash
#SBATCH --job-name=kh_probe_shoot_M18a238
#SBATCH -A fdb@v100
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=01:30:00
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

python code/scripts/legacy/probe_supersonic_shooting_first_neutral.py \
  --ci-datasets assets/classic_supersonic/csv/blumen_validation/supersonic/table_ci_datasets.csv \
  --cr-datasets assets/classic_supersonic/csv/blumen_validation/supersonic/table_cr_datasets.csv \
  --mach 1.8 \
  --alpha 0.238 \
  --ci-factors 0.5,1.0,2.0 \
  --cr-half 0.08 \
  --max-iter 10 \
  --grid-size 4 \
  --output assets/classic_supersonic/extension_neutral_M180_M190_gep_shooting/shooting_first_probe_M18_a238.csv
