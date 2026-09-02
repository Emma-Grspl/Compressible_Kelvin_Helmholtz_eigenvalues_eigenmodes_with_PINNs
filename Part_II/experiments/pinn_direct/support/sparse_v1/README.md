# Sparse supersonic PINN dataset v1

Canonical spectral reference:
assets/classic_supersonic/csv/modal_reconstruction/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/table_classical_supersonic_final_reference.csv

Canonical modal reference:
experiments/modal_reconstruction/support/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/classical_supersonic_final_modes_long.csv.gz

Canonical modal index:
assets/classic_supersonic/csv/modal_reconstruction/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/table_classical_supersonic_final_modes_index.csv

## Sparse configurations

- S4: four spectral anchors `(cr, ci)` per Mach.
- S4M2: S4 plus modal supervision at anchor ranks 1 and 4.
- S4M4: S4 plus modal supervision at all four anchors.

## Modal variables

- `kappa = Re(p_y / p)`
- `q = Im(p_y / p)`
- `logabs_p_center_gauge = log|p| - log|p(y nearest 0)|`
- `logabs_p_peak_gauge = log|p| - max_y log|p|`

The phase is not a PINN target.

## Mach split

Validation Mach:
[1.15, 1.45, 1.75]

Test Mach:
[1.25, 1.55, 1.85]

All other Mach are training Mach.
