#!/bin/bash
#SBATCH --job-name=anchorM180
#SBATCH --output=assets/pinn_supersonic/logs/anchorM180_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/anchorM180_%j.err
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

CHECKPOINT="models_saved/modal_reconstruction/pinn_chart_M180_alpha_070_190_pq_qdef005_ode001_tail_polished_asinh_w192_d6/best_model.pt"

OUT_DIR="assets/pinn_supersonic/runs/pinn_M180_shooting_anchor_modal_control"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

python code/scripts/audits/audit_pinn_shooting_at_classical_anchors.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUT_DIR" \
  --alphas 0.07 0.10 0.13 0.16 0.19 \
  --comparison-ymax 20 \
  --match-y 1.0 \
  --use-mapping \
  --mapping-scale 3.0 \
  --min-y-limit 20 \
  --max-y-limit 2000 \
  --y-limit-factor 6.0 \
  --cr-half-width 0.005 \
  --ci-half-width 0.002 \
  --ci-relative-half-width 0.25 \
  --shooting-max-iter 10 \
  --shooting-grid-size 5 \
  --device cuda
