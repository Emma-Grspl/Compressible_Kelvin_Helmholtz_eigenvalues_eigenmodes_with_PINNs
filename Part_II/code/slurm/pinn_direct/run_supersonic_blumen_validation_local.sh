#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/Thèse/These_PINN_KH_RT}"
WORKERS="${WORKERS:-2}"

cd "$ROOT"

if [[ -f ".venv-kh-local/bin/activate" ]]; then
    source ".venv-kh-local/bin/activate"
fi

CAMPAIGN_FILE="$(
  find "$PWD" \
    -type f \
    -name 'run_dense_supersonic_campaign.py' \
    -not -path '*/QUARANTINE*' \
    -print \
  | head -n 1
)"

if [[ -z "$CAMPAIGN_FILE" ]]; then
    echo "Missing run_dense_supersonic_campaign.py" >&2
    exit 1
fi

CAMPAIGN_DIR="$(dirname "$CAMPAIGN_FILE")"
export PYTHONPATH="$CAMPAIGN_DIR:$PWD:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

POINTWISE="$PWD/article/tables/table_supersonic_blumen_positive_pointwise_values.csv"
REFERENCE="$PWD/assets/classic_supersonic/csv/modal_reconstruction/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/table_classical_supersonic_final_reference.csv"
CONFIG="$PWD/code/configs/legacy/dense_supersonic_campaign_config.json"
RESULT_ROOT="$PWD/classic_supersonic/reproducibility/results/blumen_true_positive_isolines_local_v1"
OUTPUT_ROOT="$PWD/assets/classic_supersonic/article"

COMPUTE="$PWD/code/scripts/evaluation/compute_supersonic_blumen_true_isoline_point.py"
PLOT="$PWD/code/scripts/data_preparation/prepare_build_supersonic_blumen_validation_figures.py"

for path in "$POINTWISE" "$REFERENCE" "$CONFIG" "$COMPUTE" "$PLOT"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done

mkdir -p "$RESULT_ROOT/logs" "$OUTPUT_ROOT"

N_POS="$(
python - "$POINTWISE" <<'PY'
import sys
import pandas as pd
frame = pd.read_csv(sys.argv[1])
ci = pd.to_numeric(frame["blumen_ci"], errors="coerce")
print(int((ci > 0.0).sum()))
PY
)"

echo "Positive Blumen targets: $N_POS"
echo "Local workers: $WORKERS"

export POINTWISE REFERENCE CONFIG RESULT_ROOT COMPUTE

seq 0 $((N_POS - 1)) |
xargs -P "$WORKERS" -I{} bash -c '
    set -euo pipefail

    index="$1"
    tag="$(printf "%03d" "$index")"
    result="$RESULT_ROOT/point_${tag}/result.csv"

    if [[ -s "$result" ]]; then
        echo "Already present: point_${tag}"
        exit 0
    fi

    python -u "$COMPUTE" \
        --pointwise-csv "$POINTWISE" \
        --reference-csv "$REFERENCE" \
        --config "$CONFIG" \
        --task-index "$index" \
        --output-root "$RESULT_ROOT" \
        --alpha-min 1e-4 \
        --alpha-max 0.50 \
        --scan-step 0.0025 \
        --max-halfwidth 0.18 \
        --alpha-tolerance 2e-5 \
        --ci-tolerance 2e-5 \
        --max-refine-iterations 24 \
        > "$RESULT_ROOT/logs/point_${tag}.out" \
        2> "$RESULT_ROOT/logs/point_${tag}.err"

    echo "Finished: point_${tag}"
' _ {}

python -u "$PLOT" \
    --pointwise-csv "$POINTWISE" \
    --result-root "$RESULT_ROOT" \
    --output-root "$OUTPUT_ROOT"

echo
echo "Figures written to:"
echo "$OUTPUT_ROOT"
