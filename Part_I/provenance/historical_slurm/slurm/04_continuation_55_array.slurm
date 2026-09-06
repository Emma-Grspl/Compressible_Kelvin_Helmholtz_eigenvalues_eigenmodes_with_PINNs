#!/bin/bash
#SBATCH --job-name=KH_CONT_55
#SBATCH --output=slurm/log/%A_%a_KH_CONT_55.out
#SBATCH --error=slurm/log/%A_%a_KH_CONT_55.err
#SBATCH --array=0-54%16
#SBATCH --time=20:00:00
#SBATCH --partition=cpu_p1
#SBATCH -A fdb@cpu
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"

module purge
module load pytorch-gpu/py3/2.1.1
set +H
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

TARGETS="assets/pinn_subsonic/local_atlas_v1/offgrid_validation_384/offgrid_near_neutral_continuation_targets.csv"
OUTPUT_DIR="assets/pinn_subsonic/local_atlas_v1/offgrid_validation_384/continuation_N401"
DRIVER="scripts/dev/continue_subsonic_offgrid_neutral_N401.py"

test -f "$TARGETS"
test -f "$DRIVER"
mkdir -p "$OUTPUT_DIR"

python "$DRIVER" \
  --targets-csv "$TARGETS" \
  --target-index "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$OUTPUT_DIR" \
  --eta-low 0.92 \
  --eta-high 0.98 \
  --max-step 0.0025 \
  --N 401 \
  --mapping-scale 5 \
  --xi-max 0.98
