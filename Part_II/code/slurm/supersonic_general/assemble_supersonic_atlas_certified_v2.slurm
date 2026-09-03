#!/bin/bash
#SBATCH --job-name=atlas_assemble_v2
#SBATCH --output=assets/pinn_supersonic/logs/%x_%j.out
#SBATCH --error=assets/pinn_supersonic/logs/%x_%j.err
#SBATCH --time=01:00:00
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

INPUT="assets/pinn_supersonic/datasets/atlas_v1_certified_v2"

OUTPUT="${INPUT}/assembled"

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

python code/scripts/evaluation/assemble_supersonic_atlas_certified_v2.py \
  --input-dir "$INPUT" \
  --output-dir "$OUTPUT" \
  --expected-points 242 \
  --expected-mach 22 \
  --expected-s 11 \
  --anchor-mach 1.8 \
  --require-complete

cat "$OUTPUT/assembly_report.json"

python - <<'PY'
import json
from pathlib import Path

path = Path(
    "assets/pinn_supersonic/datasets/"
    "atlas_v1_certified_v2/assembled/"
    "assembly_report.json"
)

report = json.loads(
    path.read_text()
)

assert report["complete"] is True
assert report["n_assembled"] == 242

print("ASSEMBLED ATLAS V2: COMPLETE")
PY
