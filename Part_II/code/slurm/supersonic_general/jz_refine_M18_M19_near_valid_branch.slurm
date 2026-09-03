#!/bin/bash
#SBATCH -J KH_REFINE_M18M19
#SBATCH -A fdb@cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/KH_REFINE_M18M19_%j.out
#SBATCH --error=slurm_logs/KH_REFINE_M18M19_%j.err

module purge
module load pytorch-gpu/py3/2.1.1

cd $WORK/These_PINN_KH_RT
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python code/scripts/shooting/solve_refine_M18_M19_near_valid_branch_7cb1f8ed3e.py \
  --alpha-step 0.0025 \
  --mapping-scale 2.0 \
  --search-y-limit 900 \
  --final-y-limit 1600 \
  --cr-width 0.012 \
  --ci-width 0.004 \
  --grid-size 5 \
  --max-iter 7
