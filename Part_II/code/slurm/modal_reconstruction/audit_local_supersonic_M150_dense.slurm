#!/bin/bash
#SBATCH --job-name=sup_M150_audit
#SBATCH --output=assets/pinn_supersonic/logs/%x_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/%x_%j.err
#SBATCH --time=02:00:00
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
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PILOT_DIR="assets/pinn_supersonic/pilots/local_M150_sparse_v1"

REFERENCE_DIR="assets/classic_supersonic/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS"

OUTPUT_DIR="$PILOT_DIR/runs/dense_modal_audit_S4_vs_S4M2"

rm -rf "$OUTPUT_DIR"

echo "job   : ${SLURM_JOB_ID}"
echo "node  : $(hostname)"
echo "start : $(date)"
echo

nvidia-smi

srun \
  python \
    code/scripts/audits/audit_local_supersonic_modal_dense.py \
    --s4-checkpoint \
      "$PILOT_DIR/runs/long_S4/best_joint_checkpoint.pt" \
    --s4m2-checkpoint \
      "$PILOT_DIR/runs/long_S4M2/best_joint_checkpoint.pt" \
    --modal-reference \
      "$REFERENCE_DIR/classical_supersonic_final_modes_long.csv.gz" \
    --spectral-anchor-file \
      "$PILOT_DIR/data/spectral_anchors_S4_M150.csv" \
    --modal-anchor-bank \
      "$PILOT_DIR/data/modal_bank_S4M2_M150.npz" \
    --output-dir "$OUTPUT_DIR" \
    --mach 1.50 \
    --device cuda \
    --chunksize 250000 \
    --batch-size 8192 \
    --y-max 80 \
    --amplitude-floor 1e-6

echo
echo "end: $(date)"
