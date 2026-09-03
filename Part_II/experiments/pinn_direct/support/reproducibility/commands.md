# Reproducibility commands

All commands below are executed from the repository root unless stated
otherwise.

## Validate the lightweight release

    cd classic_supersonic
    make verify

## Rebuild the complete local modal Parquet datasets

The two original complete modal CSV files must be available locally.

    cd classic_supersonic
    make prepare-modal

## Rebuild the frozen witness sample

The complete local raw modal Parquet dataset must be available.

    cd classic_supersonic
    make build-witness

## Inspect the frozen solver API

    cd classic_supersonic
    make solver-api

## Witness case

The numerical reproducibility witness is:

- Mach: 1.50
- alpha: 0.1625
- reference cr: 0.3680825090907056
- reference ci: 0.0380056534011514
- reference omega_i: 0.0061759186776871
- modal variant: raw confirmed

Tracked witness files:

- `data/samples/witness_M150_a01625_raw.parquet`
- `data/samples/witness_M150_a01625_expected.json`
- `configs/reproducibility/witness_M150_a01625.yaml`

The exact numerical tolerances are determined from repeated solver runs. They
must not be chosen arbitrarily.

## Validate spectral and modal reproduction

This command runs the lightweight regression tests, the spectral solver smoke
test and the modal reconstruction smoke test.

    cd classic_supersonic
    make release-verify

## Recalibrate modal reproduction

Three repeated calibration runs should be generated before finalizing new
modal thresholds.

    python scripts/validation/calibrate_witness_modal_reproduction.py
    python tools/check_witness_modal_reproduction.py
