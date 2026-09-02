#!/bin/bash
#SBATCH --job-name=KH_N60_CM
#SBATCH --output=assets/pinn_supersonic/atlas2d_v1_continuousM/N60/logs/train_%A_%a.out
#SBATCH --error=assets/pinn_supersonic/atlas2d_v1_continuousM/N60/logs/train_%A_%a.err
#SBATCH --array=0-11
#SBATCH --time=12:00:00
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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1


CHARTS=(
  C00 C01 C02
  C10 C11 C12
  C20 C21 C22
  C30 C31 C32
)

CHART="${CHARTS[$SLURM_ARRAY_TASK_ID]}"

CONFIG="configs/pinn_supersonic/atlas2d_v1_continuousM/N60/${CHART}.json"

OUTPUT="assets/pinn_supersonic/atlas2d_v1_continuousM/N60/runs/${CHART}"


if [[ ! -f "$CONFIG" ]]; then
    echo "Missing config:"
    echo "  $CONFIG"
    exit 2
fi


mkdir -p \
  "$OUTPUT"


echo "===================================================================================================="
echo "SUPERSONIC ATLAS2D V1 — N60"
echo "===================================================================================================="
echo "job          = ${SLURM_JOB_ID}"
echo "array task   = ${SLURM_ARRAY_TASK_ID}"
echo "chart        = ${CHART}"
echo "config       = ${CONFIG}"
echo "output       = ${OUTPUT}"
echo "node         = $(hostname)"
echo "start        = $(date)"
echo "===================================================================================================="


srun python \
  code/scripts/training/train_global_supersonic_kappa_q_logamp_continuousM.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT" \
  --device cuda \
  --prefit-steps 2000 \
  --modal-steps 4000 \
  --joint-steps 4000 \
  --spectral-lr 2e-4 \
  --modal-lr 1e-4 \
  --joint-spectral-lr 5e-5 \
  --joint-modal-lr 5e-5 \
  --n-interior 1024 \
  --n-boundary 96 \
  --print-every 100 \
  --overwrite


echo
echo "===================================================================================================="
echo "FINISHED"
echo "chart = ${CHART}"
echo "end   = $(date)"
echo "===================================================================================================="
