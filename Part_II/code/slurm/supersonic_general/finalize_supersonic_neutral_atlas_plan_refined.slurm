#!/bin/bash
#SBATCH --job-name=neutral_finalize_ref
#SBATCH --output=assets/pinn_supersonic/logs/%x_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/%x_%j.err
#SBATCH --time=01:00:00
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

LOWM="assets/pinn_supersonic/csv/classical_convergence/supersonic_neutral_boundary_lowM_M100_120/table_alpha_traces.csv"

REFINED="assets/pinn_supersonic/runs/supersonic_neutral_boundary_refined_M100_115/alpha_traces.csv"

DENSE_A="assets/pinn_supersonic/csv/classical_convergence/supersonic_neutral_boundary_dense_M120_160/table_alpha_traces.csv"

DENSE_B="assets/pinn_supersonic/csv/classical_convergence/supersonic_neutral_boundary_dense_M160_200/table_alpha_traces.csv"

OUT_DIR="assets/pinn_supersonic/runs/supersonic_neutral_boundary_master_M100_200_refined"

for path in \
  "$LOWM" \
  "$REFINED" \
  "$DENSE_A" \
  "$DENSE_B"
do
  test -s "$path"
done

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

python code/scripts/evaluation/merge_supersonic_neutral_campaigns.py \
  --alpha-traces \
    "$LOWM" \
    "$REFINED" \
    "$DENSE_A" \
    "$DENSE_B" \
  --output-dir "$OUT_DIR"

python code/scripts/evaluation/refit_supersonic_neutral_boundary_secant.py \
  --alpha-traces "$OUT_DIR/alpha_traces_merged.csv" \
  --output-dir "$OUT_DIR" \
  --max-extrapolation 0.005

python code/scripts/data_preparation/prepare_build_supersonic_atlas_sampling_plan.py \
  --neutral-refit "$OUT_DIR/neutral_boundary_secant_refit.csv" \
  --output-dir "$OUT_DIR" \
  --dense-points 1001

echo
echo "MERGE REPORT"
cat "$OUT_DIR/merge_report.json"

echo
echo "ATLAS PLAN REPORT"
cat "$OUT_DIR/atlas_plan_report.json"

echo
echo "BOUNDARY KNOTS"
cat "$OUT_DIR/neutral_boundary_knots.csv"
