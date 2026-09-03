#!/bin/bash
#SBATCH -J KH_SMALLM_SCAN
#SBATCH -A fdb@cpu
#SBATCH --array=0-54
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/KH_SMALLM_SCAN_%A_%a.out
#SBATCH --error=slurm_logs/KH_SMALLM_SCAN_%A_%a.err

module purge
module load pytorch-gpu/py3/2.1.1

cd $WORK/These_PINN_KH_RT
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python code/scripts/audits/audit_scan_supersonic_target_campaign_3895a18fd6.py \
  --target-csv assets/classic_supersonic/final_sparse_PINN_reference/campaign_small_M_low_high_alpha_targets.csv \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-dir assets/classic_supersonic/campaign_smallM_low_high_alpha_scan \
  --grid-size 5 \
  --max-iter 7 \
  --max-validate-per-point 8
