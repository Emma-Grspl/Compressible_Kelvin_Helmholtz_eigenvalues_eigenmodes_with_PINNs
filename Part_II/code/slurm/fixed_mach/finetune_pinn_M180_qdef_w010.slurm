#!/bin/bash
#SBATCH --job-name=qdefM180_w010
#SBATCH --output=assets/pinn_supersonic/logs/qdefM180_w010_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/qdefM180_w010_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --account=fdb@v100

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd "$WORK/These_PINN_KH_RT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

SOURCE_RUN="assets/pinn_supersonic/runs/pinn_chart_M180_alpha_070_190_pq_qdef_tail_polished_asinh_w192_d6_w005"
SOURCE_CHECKPOINT="${SOURCE_RUN}/best_model.pt"

RUN_DIR="assets/pinn_supersonic/runs/pinn_chart_M180_alpha_070_190_pq_qdef_tail_polished_asinh_w192_d6_w010"

mkdir -p "$RUN_DIR"

test -f "$SOURCE_CHECKPOINT"

echo "[stage 1] second qdef fine-tuning"

python code/scripts/evaluation/finetune_pinn_supersonic_fixedM_qdef.py \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --run-dir "$RUN_DIR" \
  --steps 1500 \
  --batch-size 8192 \
  --collocation-batch-size 8192 \
  --lr 1e-5 \
  --qdef-weight 0.10 \
  --qdef-warmup-steps 150 \
  --collocation-ymax 20 \
  --edge-fraction 0.35 \
  --edge-width 0.0125 \
  --eval-every 100 \
  --eval-data-size 32768 \
  --eval-collocation-size 16384 \
  --seed 5432 \
  --device cuda

echo
echo "[stage 2] checkpoint evaluation"

python code/scripts/evaluation/evaluate_pinn_supersonic_fixedM_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --device cuda

echo
echo "[stage 3] modal and spectral audit"

python code/scripts/audits/audit_pinn_supersonic_fixedM_predictions.py \
  --run-dir "$RUN_DIR" \
  --top 20

echo
echo "[stage 4] core and qdef audit"

python code/scripts/audits/audit_pinn_supersonic_fixedM_core_windows.py \
  --run-dir "$RUN_DIR" \
  --windows 5 10 20 50 100 500 2000 \
  --core-window 20 \
  --edge-count 4 \
  --top 15

echo
echo "[done] metrics"
cat "$RUN_DIR/metrics.json"
