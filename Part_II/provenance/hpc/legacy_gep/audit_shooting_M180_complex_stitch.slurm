#!/bin/bash
#SBATCH --job-name=stitchM180
#SBATCH --output=assets/pinn_supersonic/logs/stitchM180_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/stitchM180_%j.err
#SBATCH --time=02:00:00
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

VALIDATION_SUMMARY="experiments/legacy_gep/legacy_outputs/pinn_M180_offanchor_gep_shooting_pilot/table_offanchor_validation_summary.csv"

OUT_DIR="assets/pinn_supersonic/runs/pinn_M180_offanchor_shooting_complex_stitch"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

test -f "$CHECKPOINT"
test -f "$VALIDATION_SUMMARY"
test -f code/scripts/audits/audit_supersonic_shooting_complex_stitch.py

python code/scripts/audits/audit_supersonic_shooting_complex_stitch.py \
  --checkpoint "$CHECKPOINT" \
  --validation-summary "$VALIDATION_SUMMARY" \
  --output-dir "$OUT_DIR" \
  --match-y 1.0 \
  --comparison-ymax 20 \
  --n-y 2001 \
  --use-mapping \
  --mapping-scale 3.0 \
  --min-y-limit 20 \
  --max-y-limit 2000 \
  --y-limit-factor 6.0 \
  --modal-overlap-min 0.90 \
  --p-rel-max 0.30 \
  --q-rel-max 0.50 \
  --q-match-jump-max 0.10 \
  --device cuda
