#!/bin/bash
#SBATCH --job-name=qdefM180_w005
#SBATCH --output=assets/pinn_supersonic/logs/qdefM180_w005_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/qdefM180_w005_%j.err
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

SOURCE_RUN="assets/pinn_supersonic/runs/pinn_chart_M180_alpha_070_190_pq_supervised_tail_polished_asinh_w192_d6"

SOURCE_CHECKPOINT="${SOURCE_RUN}/best_model.pt"

RUN_DIR="assets/pinn_supersonic/runs/pinn_chart_M180_alpha_070_190_pq_qdef_tail_polished_asinh_w192_d6_w005"

mkdir -p "$RUN_DIR"

test -f "$SOURCE_CHECKPOINT"
test -f code/scripts/evaluation/finetune_pinn_supersonic_fixedM_qdef.py
test -f code/scripts/evaluation/evaluate_pinn_supersonic_fixedM_checkpoint.py
test -f code/scripts/audits/audit_pinn_supersonic_fixedM_predictions.py
test -f code/scripts/audits/audit_pinn_supersonic_fixedM_core_windows.py

echo "[stage 1] qdef fine-tuning"

python code/scripts/evaluation/finetune_pinn_supersonic_fixedM_qdef.py \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --run-dir "$RUN_DIR" \
  --steps 2000 \
  --batch-size 8192 \
  --collocation-batch-size 4096 \
  --lr 2e-5 \
  --qdef-weight 0.05 \
  --qdef-warmup-steps 200 \
  --collocation-ymax 20 \
  --edge-fraction 0.25 \
  --edge-width 0.01 \
  --eval-every 100 \
  --eval-data-size 32768 \
  --eval-collocation-size 8192 \
  --seed 4321 \
  --device cuda

echo
echo "[stage 2] checkpoint evaluation"

python code/scripts/evaluation/evaluate_pinn_supersonic_fixedM_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --device cuda

echo
echo "[stage 3] audit by alpha"

python code/scripts/audits/audit_pinn_supersonic_fixedM_predictions.py \
  --run-dir "$RUN_DIR" \
  --top 20

echo
echo "[stage 4] audit by y windows and qdef"

python code/scripts/audits/audit_pinn_supersonic_fixedM_core_windows.py \
  --run-dir "$RUN_DIR" \
  --windows 5 10 20 50 100 500 2000 \
  --core-window 20 \
  --edge-count 4 \
  --top 15

echo
echo "[done] final metrics"

cat "$RUN_DIR/metrics.json"
