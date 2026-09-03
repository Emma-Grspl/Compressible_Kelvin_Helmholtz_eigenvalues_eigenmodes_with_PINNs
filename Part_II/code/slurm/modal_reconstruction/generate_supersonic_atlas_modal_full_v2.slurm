#!/bin/bash
#SBATCH --job-name=atlas_modal_full_v2
#SBATCH --output=assets/pinn_supersonic/logs/%x_%A_%a.out
#SBATCH --error=assets/pinn_supersonic/logs/%x_%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --account=fdb@v100
#SBATCH --array=0-21%6

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd "$WORK/These_PINN_KH_RT"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PLAN="assets/pinn_supersonic/csv/supersonic_general/atlas_v1_plan/table_atlas_training_plan.csv"

SEEDS="assets/pinn_supersonic/csv/classical_convergence/supersonic_neutral_boundary_master_M100_200/table_alpha_traces_merged.csv"

OUTPUT="assets/pinn_supersonic/datasets/atlas_v1_certified_v2"

test -s "$PLAN"
test -s "$SEEDS"

python code/scripts/data_preparation/prepare_generate_supersonic_atlas_modal_mach.py \
  --plan "$PLAN" \
  --seed-traces "$SEEDS" \
  --output-dir "$OUTPUT" \
  --mach-index "$SLURM_ARRAY_TASK_ID" \
  --n-y 4097 \
  --mapping-scale 3.0 \
  --continuation-alpha-step-max 0.01 \
  --continuation-alpha-step-min 0.000625 \
  --cert-stage1 1e-5 \
  --cert-stage2 1e-8 \
  --cert-q-consistency 5e-3 \
  --cert-asymptotic-bc 1e-6 \
  --cert-tail-ratio 1e-3
