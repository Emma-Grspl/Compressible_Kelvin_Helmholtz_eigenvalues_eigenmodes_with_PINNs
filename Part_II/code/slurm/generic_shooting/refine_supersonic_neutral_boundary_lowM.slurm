#!/bin/bash
#SBATCH --job-name=neutral_refine_lowM
#SBATCH --output=assets/pinn_supersonic/logs/%x_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/%x_%j.err
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

SEED_DATASET="assets/classic_supersonic/data/generic_shooting/datasets/shooting_seed_M120_alpha013.npz"

OUT_DIR="assets/pinn_supersonic/runs/supersonic_neutral_boundary_refined_M100_115"

test -f "$SEED_DATASET"
test -f code/scripts/shooting/solve_continue_supersonic_neutral_boundary.py
test -f code/scripts/evaluation/refit_supersonic_neutral_boundary_secant.py

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
  --seed-dataset "$SEED_DATASET" \
  --output-dir "$OUT_DIR" \
  --seed-mach 1.20 \
  --seed-alpha 0.13 \
  --mach-values 1.00 1.05 1.15 1.20 \
  --mach-step-max 0.02 \
  --alpha-max 0.45 \
  --alpha-step 0.005 \
  --alpha-step-min 0.0001 \
  --alpha-step-max 0.01 \
  --ci-stop 1e-5 \
  --neutral-fit-points 8 \
  --neutral-fit-margin 0.05 \
  --match-y 1.0 \
  --use-mapping \
  --mapping-scale 3.0 \
  --min-y-limit 20 \
  --max-y-limit 5000 \
  --y-limit-factor 8.0 \
  --rtol 5e-11 \
  --atol 5e-13 \
  --max-iter 14 \
  --grid-size 5 \
  --solve-tol 5e-8 \
  --cr-half-width 0.015 \
  --ci-half-width 0.006 \
  --ci-relative-half-width 0.60 \
  --retry-scales 1 2 4 8 \
  --ci-floor 1e-9 \
  --ci-weight 2.0 \
  --max-seed-distance 0.08 \
  --stage1-max 2e-5 \
  --stage2-max 1e-8

python code/scripts/evaluation/refit_supersonic_neutral_boundary_secant.py \
  --alpha-traces "$OUT_DIR/alpha_traces.csv" \
  --output-dir "$OUT_DIR" \
  --max-extrapolation 0.005

echo
echo "REPORT"
cat "$OUT_DIR/report.json"

echo
echo "REFINED BOUNDARY"
cat "$OUT_DIR/neutral_boundary_secant_refit.csv"
