#!/bin/bash
#SBATCH --job-name=kh_M180_gep_only
#SBATCH -A fdb@v100
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
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

python code/scripts/data_preparation/prepare_build_supersonic_neutral_M180_M190_gep_shooting_extension.py \
  --ci-datasets assets/classic_supersonic/csv/blumen_validation/supersonic/table_ci_datasets.csv \
  --cr-datasets assets/classic_supersonic/csv/blumen_validation/supersonic/table_cr_datasets.csv \
  --max-points 1 \
  --no-shooting \
  --no-figures
