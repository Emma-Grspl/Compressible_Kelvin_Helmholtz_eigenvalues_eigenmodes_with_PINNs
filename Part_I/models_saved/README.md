# Production checkpoints

Only production checkpoints needed to inspect the final Part I workflow are
published here. Historical, duplicate, and exploratory checkpoint trees remain
excluded from the public execution path; their treatment is recorded in the
repository provenance.

## Contents

- `production/atlas/N340/<chart_id>/model_state.pt`: 49 local atlas charts for
  the production `N340` sparse spectral-anchor budget.
- `production/fixed_mach/reference/model_best.pt`: the retained fixed-Mach
  production reference checkpoint.

Checkpoint files are PyTorch `.pt` files. Their contents are loaded by the
existing `evaluate_pinn` function in
`code/src/scripts/gep/selection/audit_mid_joint_pinn_full_gep.py`; no PyTorch
version compatibility guarantee beyond the tracked environment is claimed.

Chart selection is deterministic. See `configs/atlas/N340_chart_routing.csv`,
the routing implementation in
`code/src/scripts/gep/selection/solve_blumen_exact_joint_gep_v3.py`, and
`examples/02_query_neural_atlas.py`.

## Integrity inventory

The complete tracked checkpoint table is
[CHECKPOINT_MANIFEST.csv](CHECKPOINT_MANIFEST.csv). It contains one row per
file with the required fields: checkpoint path, chart, anchor budget, purpose,
size in bytes, and SHA-256 checksum. The manifest currently inventories 49
`N340` local-chart checkpoints and one fixed-Mach reference checkpoint.

| Checkpoint group | Chart | Anchor budget | Purpose | Size | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `production/atlas/N340/*/model_state.pt` | individual `chart_id` | `N340` | production local atlas chart | per-file | recorded in `CHECKPOINT_MANIFEST.csv` |
| `production/fixed_mach/reference/model_best.pt` | fixed-Mach reference | not recorded | production fixed-Mach reference | per-file | recorded in `CHECKPOINT_MANIFEST.csv` |

The CSV is the full, file-level table and should be used for integrity checks
instead of inferring metadata from checkpoint contents.
