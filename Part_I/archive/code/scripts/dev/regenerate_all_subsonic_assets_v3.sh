#!/bin/bash
set -euo pipefail

cd "$WORK/These_PINN_KH_RT"
module purge
module load pytorch-gpu/py3/2.1.1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs

TRAIN_PLAN="assets/pinn_subsonic/joint_ci_mode_atlas_v2/training_plan.tsv"
CENTRAL_AUDIT="assets/pinn_subsonic/joint_ci_mode_full_gep_atlas_v2/central_branch_audit/atlas_pointwise_central_branch.csv"
MODE20_ROOT="assets/pinn_subsonic/joint_ci_mode_global_validation_v1/mode20_profiles_v3"
BLUMEN_ROOT="assets/pinn_subsonic/joint_ci_mode_global_validation_v1/blumen_exact_v3"
BOX_ROOT="assets/pinn_subsonic/joint_ci_mode_global_validation_v1/gep_box_convergence_v3"
ASSET_ROOT="assets/pinn_subsonic/joint_ci_mode_final_assets_v3"

STAMP=$(date +%Y%m%d_%H%M%S)
for path in "$MODE20_ROOT" "$BLUMEN_ROOT" "$BOX_ROOT"; do
  if [ -d "$path" ]; then
    mv "$path" "${path}_previous_${STAMP}"
  fi
done
mkdir -p "$MODE20_ROOT" "$BLUMEN_ROOT" "$BOX_ROOT"

python -m py_compile \
  scripts/dev/build_subsonic_assets_v3.py \
  scripts/dev/run_mode20_profiles_v3.py \
  scripts/dev/run_blumen_exact_joint_gep_v3.py \
  scripts/dev/run_gep_box_convergence_v3.py
bash -n slurm/run_subsonic_asset_point_v3.slurm
bash -n slurm/finalize_subsonic_assets_v3.slurm

python scripts/dev/run_mode20_profiles_v3.py prepare \
  --central-audit "$CENTRAL_AUDIT" \
  --training-plan "$TRAIN_PLAN" \
  --output-dir "$MODE20_ROOT"

python scripts/dev/run_blumen_exact_joint_gep_v3.py prepare \
  --blumen-dir KH_RT_Blumen/subsonic \
  --training-plan "$TRAIN_PLAN" \
  --output-dir "$BLUMEN_ROOT"

python scripts/dev/run_gep_box_convergence_v3.py prepare \
  --central-audit "$CENTRAL_AUDIT" \
  --training-plan "$TRAIN_PLAN" \
  --output-dir "$BOX_ROOT" \
  --target-mach 0.50 \
  --target-eta 0.50

MODE20_PLAN="$MODE20_ROOT/mode_points_20_plan.csv"
BLUMEN_PLAN="$BLUMEN_ROOT/blumen_exact_plan.csv"
BOX_PLAN="$BOX_ROOT/GEP_box_N_plan.csv"

N_MODE20=$(python -c 'import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))' "$MODE20_PLAN")
N_BLUMEN=$(python -c 'import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))' "$BLUMEN_PLAN")
N_BOX=$(python -c 'import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))' "$BOX_PLAN")

LAST_MODE20=$((N_MODE20 - 1))
LAST_BLUMEN=$((N_BLUMEN - 1))
LAST_BOX=$((N_BOX - 1))

MODE20_JOB=$(sbatch --parsable \
  --job-name="KH_MODE20_V3" \
  --array="0-${LAST_MODE20}%8" \
  slurm/run_subsonic_asset_point_v3.slurm \
  scripts/dev/run_mode20_profiles_v3.py \
  "$MODE20_PLAN" \
  "$MODE20_ROOT")

BLUMEN_JOB=$(sbatch --parsable \
  --job-name="KH_BLUMEN_EXACT_V3" \
  --array="0-${LAST_BLUMEN}%12" \
  slurm/run_subsonic_asset_point_v3.slurm \
  scripts/dev/run_blumen_exact_joint_gep_v3.py \
  "$BLUMEN_PLAN" \
  "$BLUMEN_ROOT")

BOX_JOB=$(sbatch --parsable \
  --job-name="KH_BOX_CONV_V3" \
  --array="0-${LAST_BOX}%8" \
  slurm/run_subsonic_asset_point_v3.slurm \
  scripts/dev/run_gep_box_convergence_v3.py \
  "$BOX_PLAN" \
  "$BOX_ROOT")

FINAL_JOB=$(sbatch --parsable \
  --dependency="afterok:${MODE20_JOB}:${BLUMEN_JOB}:${BOX_JOB}" \
  slurm/finalize_subsonic_assets_v3.slurm \
  "$TRAIN_PLAN" \
  "$CENTRAL_AUDIT" \
  "$MODE20_ROOT" \
  "$BLUMEN_ROOT" \
  "$BOX_ROOT" \
  "$ASSET_ROOT")

cat <<EOF
MODE20_JOB=$MODE20_JOB
BLUMEN_JOB=$BLUMEN_JOB
BOX_JOB=$BOX_JOB
FINAL_JOB=$FINAL_JOB
ASSET_ROOT=$ASSET_ROOT

Suivi:
  squeue -j $MODE20_JOB,$BLUMEN_JOB,$BOX_JOB,$FINAL_JOB -o "%.18i %.24j %.2t %.10M %.20R"

Quand FINAL_JOB est COMPLETED:
  find "$ASSET_ROOT" -type f -printf '%P\n' | sort
EOF
