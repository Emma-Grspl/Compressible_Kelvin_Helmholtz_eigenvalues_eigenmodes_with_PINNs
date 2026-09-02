#!/bin/bash
#SBATCH --job-name=pinnM180pq
#SBATCH --output=assets/pinn_supersonic/logs/pinnM180pq_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/pinnM180pq_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --account=fdb@v100

module purge
module load pytorch-gpu/py3/2.1.1

cd $WORK/These_PINN_KH_RT
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python code/scripts/training/train_pinn_supersonic_fixedM_pq_modal_anchors.py \
  --dataset assets/pinn_supersonic/data/fixed_mach/datasets/fixedM_M180_alpha_branch_pq_modal_anchors_tail_polished.npz \
  --run-dir assets/pinn_supersonic/runs/pinn_chart_M180_alpha_070_190_pq_supervised_tail_polished \
  --epochs 3000 \
  --batch-size 8192 \
  --lr 2e-4 \
  --width 128 \
  --depth 5 \
  --activation silu \
  --modal-weight 1.0 \
  --spectral-weight 1.0 \
  --y-scale 10.0 \
  --seed 1234
