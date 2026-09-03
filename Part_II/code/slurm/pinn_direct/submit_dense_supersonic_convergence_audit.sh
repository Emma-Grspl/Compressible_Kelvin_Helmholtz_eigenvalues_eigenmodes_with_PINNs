#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-$PWD}"
cd "$REPO"

RUNNER="classic_supersonic/scripts/validation/run_dense_supersonic_convergence_audit.py"
TASK_JOB="classic_supersonic/jobs/dense_supersonic_convergence_task.slurm"
FINAL_JOB="classic_supersonic/jobs/dense_supersonic_convergence_finalize.slurm"
RESULTS="classic_supersonic/reproducibility/results/dense_supersonic_convergence_audit_v1"
LOGS="classic_supersonic/reproducibility/logs"

for path in "$RUNNER" "$TASK_JOB" "$FINAL_JOB"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
mkdir -p "$LOGS"
module purge
module load pytorch-gpu/py3/2.1.1
python3 -m py_compile \
  "$RUNNER" \
  classic_supersonic/scripts/validation/build_dense_supersonic_convergence_assets.py
python3 -u "$RUNNER" --repo "$PWD" --prepare

N_TASKS="$(python3 - "$RESULTS/audit_points.csv" <<'PY'
import csv, sys
with open(sys.argv[1], newline='', encoding='utf-8') as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)"
[[ "$N_TASKS" =~ ^[0-9]+$ ]] && (( N_TASKS > 0 )) || { echo "Invalid task count: $N_TASKS" >&2; exit 1; }

PROJECT="${IDRPROJ:-fdb}"
ARRAY_SUBMISSION="$(
  sbatch --parsable \
    --account="${PROJECT}@cpu" \
    --array="0-$((N_TASKS - 1))%17" \
    "$TASK_JOB"
)"
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"
[[ "$ARRAY_JOB_ID" =~ ^[0-9]+$ ]] || { echo "Invalid array job id: $ARRAY_SUBMISSION" >&2; exit 1; }

FINAL_SUBMISSION="$(
  sbatch --parsable \
    --account="${PROJECT}@cpu" \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    "$FINAL_JOB"
)"
FINAL_JOB_ID="${FINAL_SUBMISSION%%;*}"
[[ "$FINAL_JOB_ID" =~ ^[0-9]+$ ]] || { echo "Invalid final job id: $FINAL_SUBMISSION" >&2; exit 1; }

printf 'Convergence array job: %s (%s tasks)\n' "$ARRAY_JOB_ID" "$N_TASKS"
printf 'Convergence assets job: %s (afterok:%s)\n' "$FINAL_JOB_ID" "$ARRAY_JOB_ID"
printf 'Final log: %s/kh_sup_conv_assets_%s.out\n' "$LOGS" "$FINAL_JOB_ID"
