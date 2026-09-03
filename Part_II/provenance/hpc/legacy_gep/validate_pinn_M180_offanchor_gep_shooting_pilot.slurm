#!/bin/bash
#SBATCH --job-name=pinnM180_offval
#SBATCH --output=assets/pinn_supersonic/logs/pinnM180_offval_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/pinnM180_offval_%j.err
#SBATCH --time=12:00:00
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

OUT_DIR="assets/pinn_supersonic/runs/pinn_M180_offanchor_gep_shooting_pilot"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

test -f "$CHECKPOINT"
test -f code/scripts/evaluation/evaluate_validate_pinn_supersonic_offanchor_gep_shooting.py

python code/scripts/evaluation/evaluate_validate_pinn_supersonic_offanchor_gep_shooting.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUT_DIR" \
  --n-cases 8 \
  --ymax 20 \
  --n-y 2001 \
  --gep-n 301 \
  --gep-mapping-kind pin \
  --gep-mapping-scale 3.0 \
  --gep-xi-max 0.98 \
  --ci-weight 2.0 \
  --match-y 1.0 \
  --use-mapping \
  --shooting-mapping-scale 3.0 \
  --min-y-limit 20 \
  --max-y-limit 2000 \
  --y-limit-factor 6.0 \
  --cr-half-width 0.01 \
  --ci-half-width 0.003 \
  --ci-relative-half-width 0.30 \
  --shooting-max-iter 10 \
  --shooting-grid-size 5 \
  --gep-seed-distance-max 0.02 \
  --gep-overlap-min 0.80 \
  --shooting-seed-distance-max 0.01 \
  --shooting-stage1-mismatch-max 1e-5 \
  --shooting-spectral-distance-max 1e-4 \
  --shooting-overlap-min 0.90 \
  --shooting-p-rel-max 0.30 \
  --shooting-q-rel-max 0.50 \
  --device cuda
