#!/bin/bash
#SBATCH --job-name=KH_MODE_PDFS
#SBATCH --output=slurm/log/%j_KH_MODE_PDFS.out
#SBATCH --error=slurm/log/%j_KH_MODE_PDFS.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_p1
#SBATCH -A fdb@cpu
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
module purge
module load pytorch-gpu/py3/2.1.1

OUT="assets/pinn_subsonic/local_atlas_v1/publication_assets_scientific_v2"
python code/plots/scripts/pinn_subsonic/active_implementation/source_tree/scripts/assets_v2/plot_mode_pdfs.py \
  --points-csv "$OUT/data/validation_mode_points_20.csv" \
  --direct-profile-dir "$OUT/mode_profiles/direct" \
  --gep-profile-dir "$OUT/mode_profiles/gep" \
  --output-dir "$OUT/supplement"

python code/plots/scripts/pinn_subsonic/active_implementation/source_tree/scripts/assets_v2/plot_representative_mode_figure.py \
  --points-csv "$OUT/data/validation_mode_points_20.csv" \
  --direct-profile-dir "$OUT/mode_profiles/direct" \
  --gep-profile-dir "$OUT/mode_profiles/gep" \
  --output-stem "$OUT/figures/Fig06_representative_modes" \
  --n-points 6
