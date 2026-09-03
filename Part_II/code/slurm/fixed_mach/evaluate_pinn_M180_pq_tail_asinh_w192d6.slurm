#!/bin/bash
#SBATCH --job-name=evalM180_tail
#SBATCH --output=assets/pinn_supersonic/logs/evalM180_tail_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/evalM180_tail_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --account=fdb@v100

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd "$WORK/These_PINN_KH_RT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

RUN_DIR="assets/pinn_supersonic/runs/pinn_chart_M180_alpha_070_190_pq_supervised_tail_polished_asinh_w192_d6"

test -f "$RUN_DIR/best_model.pt"
test -f code/scripts/evaluation/evaluate_pinn_supersonic_fixedM_checkpoint.py

python code/scripts/evaluation/evaluate_pinn_supersonic_fixedM_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --device cuda
