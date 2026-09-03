#!/bin/bash
#SBATCH --job-name=auditM180_tail
#SBATCH --output=assets/pinn_supersonic/logs/auditM180_tail_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/auditM180_tail_%j.err
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

echo "[audit] checking required files"
ls -lh "$RUN_DIR"/metrics.json
ls -lh "$RUN_DIR"/modal_predictions_all_rows.npz
ls -lh "$RUN_DIR"/spectral_predictions.csv

echo
echo "[audit] metrics"
cat "$RUN_DIR/metrics.json"

echo
echo "[audit] by alpha"
python code/scripts/audits/audit_pinn_supersonic_fixedM_predictions.py \
  --run-dir "$RUN_DIR" \
  --top 20

echo
echo "[audit] by y windows"
python code/scripts/audits/audit_pinn_supersonic_fixedM_core_windows.py \
  --run-dir "$RUN_DIR"
