#!/usr/bin/env bash
set -euo pipefail
REPO="${1:-$PWD}"
cd "$REPO"
SCRIPT="classic_supersonic/scripts/validation/build_dense_supersonic_requested_assets.py"
JOB="classic_supersonic/jobs/dense_supersonic_requested_assets.slurm"
[[ -f "$SCRIPT" ]] || { echo "Missing: $SCRIPT" >&2; exit 1; }
[[ -f "$JOB" ]] || { echo "Missing: $JOB" >&2; exit 1; }
mkdir -p classic_supersonic/reproducibility/logs
module purge
module load pytorch-gpu/py3/2.1.1
python3 -m py_compile "$SCRIPT"
PROJECT="${IDRPROJ:-fdb}"
SUBMISSION="$(sbatch --parsable --account="${PROJECT}@cpu" "$JOB")"
JOB_ID="${SUBMISSION%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "Invalid job id: $SUBMISSION" >&2; exit 1; }
echo "Requested assets job: $JOB_ID"
echo "Log: classic_supersonic/reproducibility/logs/kh_sup_requested_assets_${JOB_ID}.out"
