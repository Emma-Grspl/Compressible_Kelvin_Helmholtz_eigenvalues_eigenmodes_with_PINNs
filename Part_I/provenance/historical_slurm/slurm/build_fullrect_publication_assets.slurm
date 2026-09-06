#!/bin/bash
#SBATCH --job-name=KH_ASSETS_V1
#SBATCH --output=slurm/log/%j_KH_ASSETS_V1.out
#SBATCH --error=slurm/log/%j_KH_ASSETS_V1.err
#SBATCH --time=01:00:00
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
set +H
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python archive/code/scripts/dev/build_fullrect_publication_assets.py \
  --atlas-root assets/pinn_subsonic/local_atlas_v1 \
  --output-dir assets/pinn_subsonic/local_atlas_v1/publication_assets_fullrect_v1

python code/src/scripts/evaluation/audit_fullrect_publication_assets.py \
  --asset-dir assets/pinn_subsonic/local_atlas_v1/publication_assets_fullrect_v1
