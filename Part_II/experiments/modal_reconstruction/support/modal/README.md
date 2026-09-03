# Full modal datasets

The complete raw and tail-polished modal datasets contain 838847 rows each and
cover 183 unique spectral points.

Expected local files:

- `supersonic_reference_v2_modal_raw.parquet`
- `supersonic_reference_v2_modal_tail_polished.parquet`

The raw confirmed dataset is the primary scientific reference.

The tail-polished dataset is a derived export. Tail polishing does not modify
the spectral values `cr`, `ci` or `omega_i`.

## Canonical spectral columns

Historical source files use two equivalent sets of spectral columns:

- `cr`, `ci`, `omega_i`
- `reference_cr`, `reference_ci`, `reference_omega_i`

During Parquet preparation, missing values in the canonical columns are filled
from their corresponding `reference_*` columns.

When `omega_i` is missing from both representations, it is computed as:

`omega_i = alpha * ci`

Existing non-null values are never overwritten.

## Storage

The complete Parquet files are excluded from standard Git because of their
size. Their paths, sizes and SHA-256 checksums are recorded in:

- `modal_reference_metadata.json`
- `SHA256SUMS.txt`

A compact representative dataset is versioned for regression tests:

- `modal_sample_for_tests.parquet`
