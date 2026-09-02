#!/bin/bash
#SBATCH --job-name=sup_sparse_v1
#SBATCH --output=assets/pinn_supersonic/logs/%x_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --account=fdb@v100

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd "$WORK/These_PINN_KH_RT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

FINAL_REF_DIR="assets/classic_supersonic/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS"

SPECTRAL_REF="$FINAL_REF_DIR/classical_supersonic_final_reference.csv"
MODAL_REF="$FINAL_REF_DIR/classical_supersonic_final_modes_long.csv.gz"
MODAL_INDEX="$FINAL_REF_DIR/classical_supersonic_final_modes_index.csv"

OUTPUT_DIR="assets/pinn_supersonic/datasets/sparse_v1"

test -s "$SPECTRAL_REF"
test -s "$MODAL_REF"
test -s "$MODAL_INDEX"
test -s \
  code/scripts/data_preparation/prepare_build_sparse_supersonic_dataset.py

rm -rf "$OUTPUT_DIR"

python \
  code/scripts/data_preparation/prepare_build_sparse_supersonic_dataset.py \
  --spectral-reference "$SPECTRAL_REF" \
  --modal-reference "$MODAL_REF" \
  --modal-index "$MODAL_INDEX" \
  --output-dir "$OUTPUT_DIR" \
  --anchor-quantiles \
    0.10 \
    0.40 \
    0.70 \
    0.90 \
  --validation-mach \
    1.15 \
    1.45 \
    1.75 \
  --test-mach \
    1.25 \
    1.55 \
    1.85 \
  --chunksize 200000 \
  --center-safe-threshold 1e-6 \
  --allow-neutral-proxy

echo
echo "===== DATASET REPORT ====="

cat "$OUTPUT_DIR/dataset_report.json"
