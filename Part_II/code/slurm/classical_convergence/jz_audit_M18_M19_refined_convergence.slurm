#!/bin/bash
#SBATCH -J KH_CONV_M18M19
#SBATCH -A fdb@cpu
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/KH_CONV_M18M19_%j.out
#SBATCH --error=slurm_logs/KH_CONV_M18M19_%j.err

module purge
module load pytorch-gpu/py3/2.1.1

cd $WORK/These_PINN_KH_RT
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python code/scripts/audits/audit_M18_M19_refined_convergence_0e0240282c.py \
  --output-dir assets/classic_supersonic/scan_M18_M19_alpha_0_03_strict/refined_near_valid_branch/convergence_audit_full
