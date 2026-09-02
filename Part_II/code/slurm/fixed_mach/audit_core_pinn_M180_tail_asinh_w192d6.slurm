#!/bin/bash
#SBATCH --job-name=coreM180_tail
#SBATCH --output=assets/pinn_supersonic/logs/coreM180_tail_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/coreM180_tail_%j.err
#SBATCH --time=00:20:00
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

test -f "$RUN_DIR/modal_predictions_all_rows.npz"
test -f code/scripts/audits/audit_pinn_supersonic_fixedM_core_windows.py

python code/scripts/audits/audit_pinn_supersonic_fixedM_core_windows.py \
  --run-dir "$RUN_DIR" \
  --windows 5 10 20 50 100 500 2000 \
  --core-window 20 \
  --edge-count 4 \
  --top 15
