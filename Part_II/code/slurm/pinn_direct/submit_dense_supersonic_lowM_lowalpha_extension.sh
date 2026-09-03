#!/bin/bash
set -euo pipefail

REPO="${1:-${WORK:-}/These_PINN_KH_RT}"
if [[ -z "$REPO" || ! -d "$REPO" ]]; then
  echo "Repository not found: $REPO" >&2
  exit 1
fi
REPO="$(cd "$REPO" && pwd)"
cd "$REPO"

ACCOUNT="${IDRPROJ:-fdb}@cpu"
CONFIG="classic_supersonic/configs/dense_supersonic_lowM_lowalpha_extension_config.json"
RUNNER="classic_supersonic/scripts/validation/run_dense_supersonic_campaign.py"
MODES="classic_supersonic/scripts/validation/reconstruct_dense_supersonic_modes.py"
AGG="classic_supersonic/scripts/validation/aggregate_dense_supersonic_lowM_lowalpha_extension.py"
SPEC_JOB="classic_supersonic/jobs/dense_supersonic_lowM_lowalpha_spectral.slurm"
MODE_JOB="classic_supersonic/jobs/dense_supersonic_lowM_lowalpha_modes.slurm"
AGG_JOB="classic_supersonic/jobs/dense_supersonic_lowM_lowalpha_aggregate.slurm"

for path in "$CONFIG" "$RUNNER" "$MODES" "$AGG" "$SPEC_JOB" "$MODE_JOB" "$AGG_JOB"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done
mkdir -p classic_supersonic/reproducibility/logs

# Lightweight anchor preflight. It performs no root solve.
module purge
module load pytorch-gpu/py3/2.1.1
export PYTHONPATH="$PWD/classic_supersonic/scripts/validation:$PWD:${PYTHONPATH:-}"
for index in 0 1 2 3 4; do
  python3 "$RUNNER" \
    --repo "$PWD" \
    --config "$CONFIG" \
    --mach-index "$index" \
    --dry-run >/dev/null
done

echo "Anchor preflight: PASS"

SPEC_SUBMISSION="$(
  sbatch --parsable \
    --account="$ACCOUNT" \
    --export=ALL,REPO="$REPO" \
    "$SPEC_JOB"
)"
SPEC_ID="${SPEC_SUBMISSION%%;*}"

MODE_SUBMISSION="$(
  sbatch --parsable \
    --account="$ACCOUNT" \
    --dependency="afterok:${SPEC_ID}" \
    --export=ALL,REPO="$REPO" \
    "$MODE_JOB"
)"
MODE_ID="${MODE_SUBMISSION%%;*}"

AGG_SUBMISSION="$(
  sbatch --parsable \
    --account="$ACCOUNT" \
    --dependency="afterok:${MODE_ID}" \
    --export=ALL,REPO="$REPO" \
    "$AGG_JOB"
)"
AGG_ID="${AGG_SUBMISSION%%;*}"

printf 'Low-M spectral array : %s (5 tasks)\n' "$SPEC_ID"
printf 'Low-M mode array     : %s (afterok:%s)\n' "$MODE_ID" "$SPEC_ID"
printf 'Low-M aggregation    : %s (afterok:%s)\n' "$AGG_ID" "$MODE_ID"
printf 'Final log            : classic_supersonic/reproducibility/logs/kh_sup_lowM_agg_%s.out\n' "$AGG_ID"
