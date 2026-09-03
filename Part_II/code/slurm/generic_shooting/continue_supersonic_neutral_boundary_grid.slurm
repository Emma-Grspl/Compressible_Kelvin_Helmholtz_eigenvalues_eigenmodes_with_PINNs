#!/bin/bash
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

if [[ "$#" -ne 5 ]]; then
    echo "Usage:"
    echo "  $0 SEED_DATASET SEED_MACH OUT_DIR ALPHA_MAX 'MACH_VALUES'"
    exit 2
fi

SEED_DATASET="$1"
SEED_MACH="$2"
OUT_DIR="$3"
ALPHA_MAX="$4"
MACH_VALUES_STRING="$5"

read -r -a MACH_VALUES <<< "$MACH_VALUES_STRING"

module purge
module load pytorch-gpu/py3/2.1.1

cd "$WORK/These_PINN_KH_RT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

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
  --seed-mach "$SEED_MACH" \
  --seed-alpha 0.13 \
  --mach-values "${MACH_VALUES[@]}" \
  --mach-step-max 0.025 \
  --alpha-max "$ALPHA_MAX" \
  --alpha-step 0.01 \
  --alpha-step-min 0.00025 \
  --alpha-step-max 0.02 \
  --ci-stop 5e-5 \
  --neutral-fit-points 7 \
  --neutral-fit-margin 0.15 \
  --match-y 1.0 \
  --use-mapping \
  --mapping-scale 3.0 \
  --min-y-limit 20 \
  --max-y-limit 3000 \
  --y-limit-factor 6.0 \
  --rtol 1e-10 \
  --atol 1e-12 \
  --max-iter 12 \
  --grid-size 5 \
  --solve-tol 1e-7 \
  --cr-half-width 0.012 \
  --ci-half-width 0.005 \
  --ci-relative-half-width 0.45 \
  --retry-scales 1 2 4 8 \
  --ci-floor 1e-8 \
  --ci-weight 2.0 \
  --max-seed-distance 0.06 \
  --stage1-max 2e-5 \
  --stage2-max 1e-8

python code/scripts/evaluation/refit_supersonic_neutral_boundary_secant.py \
  --alpha-traces "$OUT_DIR/alpha_traces.csv" \
  --output-dir "$OUT_DIR"

echo
echo "REPORT"
cat "$OUT_DIR/report.json"

echo
echo "SECANT REFIT"
cat "$OUT_DIR/neutral_boundary_secant_refit.csv"
