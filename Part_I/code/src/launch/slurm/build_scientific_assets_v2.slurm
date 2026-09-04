#!/bin/bash
#SBATCH --job-name=KH_ASSETS_V2
#SBATCH --output=slurm/log/%j_KH_ASSETS_V2.out
#SBATCH --error=slurm/log/%j_KH_ASSETS_V2.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_p1
#SBATCH -A fdb@cpu
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
module purge
module load pytorch-gpu/py3/2.1.1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

OUT="assets/pinn_subsonic/local_atlas_v1/publication_assets_scientific_v2"

python code/src/scripts/utils/inventory_missing_inputs.py
python archive/code/scripts/assets_v2/build_ci_and_atlas_assets.py --output-dir "$OUT"
python archive/code/scripts/assets_v2/select_mode_points_20.py --output "$OUT/data/validation_mode_points_20.csv"
python code/plots/scripts/pinn_subsonic/active_implementation/source_tree/scripts/assets_v2/plot_numerical_audits.py --output-dir "$OUT"
python code/src/scripts/evaluation/audit_assets_v2.py --asset-dir "$OUT"
