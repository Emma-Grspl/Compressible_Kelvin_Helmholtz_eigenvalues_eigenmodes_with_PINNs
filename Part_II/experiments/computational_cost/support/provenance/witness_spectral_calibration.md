# Witness spectral calibration

- Case: `witness_M150_a01625`
- Mach: 1.50
- alpha: 0.1625
- Selected configuration: `unmapped_y80`

## Calibration result

- Solved cr: `0.3680880042790627`
- Solved ci: `0.0380067406402456`
- Absolute cr error: `5.495188e-06`
- Absolute ci error: `1.087239e-06`
- Absolute omega_i error: `1.766764e-07`
- Stage-1 mismatch at the frozen reference: `1.563774e-06`
- Stage-1 mismatch at the solved root: `5.103537e-10`
- Runtime: `21.227 s`

## Acceptance thresholds

- `|Δcr| <= 1.0e-05`
- `|Δci| <= 2.0e-06`
- `|Δomega_i| <= 5.0e-07`
- `stage1 mismatch <= 1.0e-08`

## Configuration selection

The unmapped y_max=80, unmapped y_max=500 and mapped scale-5 y_max=500 configurations converged to the same spectral root at the reported precision.

The unmapped y_max=80 configuration is retained for the release smoke test because it is the least expensive.

This calibration validates spectral reproducibility only. Modal-field comparison is handled separately.
