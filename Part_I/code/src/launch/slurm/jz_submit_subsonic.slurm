#!/bin/bash
#SBATCH --job-name=KH_Blumensub_cpu
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_cpu-t3

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"

module purge
module load python/3.11.5

mkdir -p slurm/log
mkdir -p assets/blumen_shooting_hybrid

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job subsonique hybride lance sur ${SLURMD_NODENAME}"
echo "CPUs par tache: ${SLURM_CPUS_PER_TASK}"

srun python3 code/src/scripts/classical/solve_classic_subsonic_gep_shooting_scan.py \
  --mach-min 0.0 \
  --mach-max 0.98 \
  --alpha-min 0.02 \
  --alpha-max 1.0 \
  --num-mach 41 \
  --num-alpha 41 \
  --ci-max 1.0 \
  --n-scan 81 \
  --neutral-ratio-threshold 0.85 \
  --mismatch-threshold 5e-4
