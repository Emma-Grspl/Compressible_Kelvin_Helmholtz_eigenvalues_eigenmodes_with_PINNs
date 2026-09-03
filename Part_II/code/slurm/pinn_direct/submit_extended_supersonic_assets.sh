#!/bin/bash
set -euo pipefail

REPO="${1:-$PWD}"
cd "$REPO"

mkdir -p classic_supersonic/reproducibility/logs

JOB="$(
  sbatch --parsable \
    classic_supersonic/jobs/build_extended_supersonic_assets.slurm \
    "$REPO"
)"
JOB_ID="${JOB%%;*}"

echo "Extended assets job: $JOB_ID"
echo "Log: classic_supersonic/reproducibility/logs/kh_sup_ext_assets_${JOB_ID}.out"
