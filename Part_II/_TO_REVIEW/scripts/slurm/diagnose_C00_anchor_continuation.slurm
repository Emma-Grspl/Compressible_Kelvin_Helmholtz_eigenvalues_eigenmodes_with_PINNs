#!/bin/bash
#SBATCH --job-name=KH_C00_ANCHOR
#SBATCH --output=assets/pinn_supersonic/atlas2d_v1/N76/corrector_benchmark/logs/full_%j.out
#SBATCH --error=assets/pinn_supersonic/atlas2d_v1/N76/corrector_benchmark/logs/full_%j.err
#SBATCH --time=20:00:00
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

python \
  code/scripts/audits/audit_diagnose_C00_anchor_continuation_N76.py \
 

echo
echo "CORRECTOR_BENCHMARK_FINISHED"
