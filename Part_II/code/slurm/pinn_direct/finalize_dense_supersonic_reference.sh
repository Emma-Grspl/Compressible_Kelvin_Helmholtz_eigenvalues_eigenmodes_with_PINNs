#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-$PWD}"
FREEZE_NAME="${FREEZE_NAME:-dense_kappa_q_campaign_v1_FINAL_FREEZE}"
PYTHON="${PYTHON:-python3}"

cd "$REPO"

FREEZER="classic_supersonic/scripts/validation/freeze_dense_supersonic_results.py"
BUILDER="classic_supersonic/scripts/validation/build_classical_supersonic_assets.py"
CONFIG="configs/production/classical/dense_supersonic_campaign_config.json"
FREEZE_DIR="assets/classic_supersonic/$FREEZE_NAME"

for path in "$FREEZER" "$BUILDER" "$CONFIG"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

"$PYTHON" -m py_compile "$FREEZER" "$BUILDER"

if [[ ! -d "$FREEZE_DIR" ]]; then
  "$PYTHON" "$FREEZER" \
    --repo "$PWD" \
    --config "$CONFIG" \
    --freeze-name "$FREEZE_NAME" \
    --expected-targets 1381 \
    --expected-retained 770 \
    --expected-modes 770
else
  echo "Prepared freeze already exists; reusing: $FREEZE_DIR"
fi

REQUIRED_ASSETS=(
  "$FREEZE_DIR/classical_supersonic_maps/cr_map.pdf"
  "$FREEZE_DIR/classical_supersonic_maps/ci_map.pdf"
  "$FREEZE_DIR/classical_supersonic_maps/omega_i_map.pdf"
  "$FREEZE_DIR/classical_supersonic_maps/neutral_curve.pdf"
  "$FREEZE_DIR/classical_supersonic_maps/retained_point_mask.pdf"
  "$FREEZE_DIR/classical_supersonic_maps/classical_supersonic_dense_reference.csv"
  "$FREEZE_DIR/classical_supersonic_all_modes.pdf"
)

ASSETS_COMPLETE=1
for path in "${REQUIRED_ASSETS[@]}"; do
  [[ -s "$path" ]] || ASSETS_COMPLETE=0
done

if (( ASSETS_COMPLETE == 0 )); then
  "$PYTHON" "$BUILDER" \
    --repo "$PWD" \
    --freeze-dir "$FREEZE_DIR" \
    --overwrite \
    --modes-per-page 2 \
    --core-limit 20 \
    --max-plot-points 1600
else
  echo "All requested assets already exist; asset generation skipped."
fi

FREEZE_STATE="$($PYTHON - "$FREEZE_DIR/freeze_metadata.json" <<'PY_STATE'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
print(json.loads(path.read_text()).get("state", "")) if path.is_file() else print("")
PY_STATE
)"

if [[ "$FREEZE_STATE" != "sealed" ]]; then
  "$PYTHON" "$FREEZER" \
    --repo "$PWD" \
    --freeze-name "$FREEZE_NAME" \
    --finalize-existing \
    --seal
else
  echo "Freeze is already sealed; finalization skipped."
fi

BUNDLE="assets/classic_supersonic/${FREEZE_NAME}.tar.gz"
CHECKSUM="${BUNDLE}.sha256"

tar -C assets/classic_supersonic -czf "$BUNDLE" "$FREEZE_NAME"
sha256sum "$BUNDLE" > "$CHECKSUM"
tar -tzf "$BUNDLE" >/dev/null

printf '\nFinal freeze: %s\n' "$FREEZE_DIR"
printf 'Transfer bundle: %s\n' "$BUNDLE"
printf 'Checksum: %s\n' "$CHECKSUM"
