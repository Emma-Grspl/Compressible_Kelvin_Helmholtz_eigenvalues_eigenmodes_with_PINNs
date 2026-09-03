#!/bin/bash
#SBATCH --job-name=neutralM100_200
#SBATCH --output=assets/pinn_supersonic/logs/neutralM100_200_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/neutralM100_200_%j.err
#SBATCH --time=20:00:00
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

DATASET="assets/pinn_supersonic/data/fixed_mach/datasets/fixedM_M180_alpha_branch_pq_modal_anchors_tail_polished.npz"

OUT_DIR="assets/pinn_supersonic/runs/supersonic_neutral_boundary_pilot_M100_200"

test -f "$DATASET"
test -f code/scripts/shooting/solve_continue_supersonic_neutral_boundary.py

python -B - <<'PY'
import inspect

from classical_solver.supersonic.mstab17_supersonic_solver import (
    Mstab17SupersonicSolver,
)

source = inspect.getsource(
    Mstab17SupersonicSolver.solve
)

assert "trial_ln_p_right" in source
assert "amp_opt" not in source
assert "minimize_scalar" not in source

print("corrected shooting solver: OK")
PY

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

python code/scripts/shooting/solve_continue_supersonic_neutral_boundary.py \
  --seed-dataset "$DATASET" \
  --output-dir "$OUT_DIR" \
  --seed-mach 1.8 \
  --seed-alpha 0.13 \
  --mach-values 1.00 1.20 1.50 1.80 2.00 \
  --mach-step-max 0.05 \
  --alpha-max 0.35 \
  --alpha-step 0.01 \
  --alpha-step-min 0.0005 \
  --alpha-step-max 0.02 \
  --ci-stop 5e-5 \
  --neutral-fit-points 7 \
  --neutral-fit-margin 0.10 \
  --match-y 1.0 \
  --use-mapping \
  --mapping-scale 3.0 \
  --min-y-limit 20 \
  --max-y-limit 2000 \
  --y-limit-factor 6.0 \
  --rtol 1e-10 \
  --atol 1e-12 \
  --max-iter 10 \
  --grid-size 5 \
  --solve-tol 1e-7 \
  --cr-half-width 0.008 \
  --ci-half-width 0.003 \
  --ci-relative-half-width 0.35 \
  --retry-scales 1 2 4 \
  --ci-floor 1e-8 \
  --ci-weight 2.0 \
  --max-seed-distance 0.03 \
  --stage1-max 1e-5 \
  --stage2-max 1e-8

echo
echo "REPORT"
cat "$OUT_DIR/report.json"

echo
echo "NEUTRAL BOUNDARY"
cat "$OUT_DIR/neutral_boundary.csv"
