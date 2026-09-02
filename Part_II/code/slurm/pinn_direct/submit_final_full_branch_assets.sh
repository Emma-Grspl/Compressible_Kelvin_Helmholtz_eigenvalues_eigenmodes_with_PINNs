#!/bin/bash
set -euo pipefail

REPO="${1:-$PWD}"
cd "$REPO"
mkdir -p classic_supersonic/reproducibility/logs

JOB="$(
  sbatch --parsable \
    classic_supersonic/jobs/build_final_full_branch_assets.slurm \
    "$REPO"
)"
JOB_ID="${JOB%%;*}"

echo "Final full-branch assets job: $JOB_ID"
echo "Log: classic_supersonic/reproducibility/logs/kh_sup_final_full_${JOB_ID}.out"
