#!/bin/bash
#SBATCH --job-name=ci4_modes
#SBATCH --account=fdb@v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm/log/ci4_modes_%j.out
#SBATCH --error=slurm/log/ci4_modes_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR}"
cd "${ROOT}"

module purge
module load pytorch-gpu/py3/2.1.1

OUT="${ROOT}/assets/pinn_subsonic/paper_results_v1"
WORK_COMPARE="${OUT}/data/single_case_ci4_vs_physics"

PURE_RUN="${ROOT}/model_saved/kh_subsonic_fixed_mach_M05_alpha010_080_riccati_pure_physics_reference"
CI4_RUN="${ROOT}/model_saved/kh_subsonic_M05_alpha010_080_hybrid_stage1_mode_from_ci_stage0/hybrid_ci4"

mkdir -p "${OUT}/logs"
mkdir -p "${OUT}/figures/single_case_ci4_vs_physics"
mkdir -p "${WORK_COMPARE}"

if [[ ! -f "${PURE_RUN}/model_best.pt" ]]; then
    echo "Checkpoint absent : ${PURE_RUN}/model_best.pt" >&2
    exit 1
fi

if [[ ! -f "${CI4_RUN}/model_best.pt" ]]; then
    echo "Checkpoint absent : ${CI4_RUN}/model_best.pt" >&2
    exit 1
fi

ln -sfn "${PURE_RUN}" "${WORK_COMPARE}/physics_only"
ln -sfn "${CI4_RUN}" "${WORK_COMPARE}/hybrid_4pt"

CLASSICAL_DIR="$(
    find "${ROOT}" \
        -maxdepth 5 \
        -type d \
        -name classical_solver \
        -not -path '*/archive/*' \
        -print \
        -quit
)"

if [[ -z "${CLASSICAL_DIR}" ]]; then
    echo "Impossible de trouver classical_solver." >&2
    exit 1
fi

export PYTHONPATH="${ROOT}:$(dirname "${CLASSICAL_DIR}")${PYTHONPATH:+:${PYTHONPATH}}"

echo "Date : $(date)"
echo "Nœud : $(hostname)"
echo "Racine : ${ROOT}"
echo "classical_solver : ${CLASSICAL_DIR}"
echo "PYTHONPATH : ${PYTHONPATH}"

nvidia-smi

python - <<'PY'
import torch
import classical_solver

print("CUDA disponible :", torch.cuda.is_available())
print("Nombre de GPU :", torch.cuda.device_count())
print("classical_solver :", classical_solver.__file__)

if not torch.cuda.is_available():
    raise RuntimeError("Aucun GPU CUDA disponible dans le job.")
PY

python code/src/scripts/utils/plot_ci4_vs_physics_modes.py \
    --base-dir "${WORK_COMPARE}" \
    --output-dir "${OUT}/figures/single_case_ci4_vs_physics" \
    --num-alpha 401 \
    --mode-alphas 0.5 \
    --num-mode-alpha 9 \
    --n-y-pinn 2001 \
    --device cuda \
    2>&1 | tee "${OUT}/logs/single_case_ci4_vs_physics_modes.log"

echo
echo "Fichiers produits :"

find "${OUT}/figures/single_case_ci4_vs_physics" \
    -type f \
    -print \
    | sort

echo
echo "Terminé : $(date)"
