#!/bin/bash
#SBATCH --job-name=N340_seams
#SBATCH --account=fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-19%10
#SBATCH --output=slurm/log/N340_seams_%A_%a.out
#SBATCH --error=slurm/log/N340_seams_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load pytorch-gpu/py3/2.1.1

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p \
  slurm/log \
  assets/pinn_subsonic/article/N340/seam_shards

python \
  code/src/scripts/evaluation/evaluate_joint_pinn_global_validation.py \
  validate-seams \
  --plan \
    assets/pinn_subsonic/csv/article/N340/Table_N340_seam_points.csv \
  --output-dir \
    assets/pinn_subsonic/article/N340/seam_shards \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --chunk-size 10 \
  --device cuda \
  --y-max 12.0 \
  --n-y 1601 \
  --amplitude-floor 0.02
