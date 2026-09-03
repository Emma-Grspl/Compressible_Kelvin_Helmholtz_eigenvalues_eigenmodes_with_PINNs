#!/bin/bash
#SBATCH -J kh_M18M19_strict
#SBATCH -A fdb@cpu
#SBATCH --array=0-59
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/kh_M18M19_strict_%A_%a.out
#SBATCH --error=slurm_logs/kh_M18M19_strict_%A_%a.err

module purge
module load pytorch-gpu/py3/2.1.1

cd $WORK/These_PINN_KH_RT
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python code/scripts/audits/audit_scan_supersonic_M18_M19_strict_modal_validation_14e1027f3b.py \
  --machs 1.8 1.9 \
  --alpha-min 0.01 \
  --alpha-max 0.30 \
  --alpha-step 0.01 \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-dir assets/classic_supersonic/scan_M18_M19_alpha_0_03_strict \
  --grid-size 5 \
  --max-iter 7 \
  --max-validate-per-point 6 \
  --stage1-tol 1e-3 \
  --max-edge-frac 2e-2 \
  --max-center-jump 0.25 \
  --max-adjacent-jump 0.35 \
  --max-ylimit-rel-l2 0.20
