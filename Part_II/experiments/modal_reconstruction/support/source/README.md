# Local complete modal source files

This directory may contain the two complete local modal CSV datasets:

- `supersonic_sparse_PINN_reference_v2_FINAL_modal_fields_raw_confirmed.csv`
- `supersonic_sparse_PINN_reference_v2_FINAL_modal_fields_tail_polished_v1.csv`

These files are intentionally excluded from Git because they exceed standard
repository size limits.

The raw-confirmed dataset is the primary modal reference. The tail-polished
dataset is derived from it; the spectral values `cr` and `ci` are unchanged.

Tracked compact Parquet samples and checksums are provided elsewhere in this
bundle for regression and reproducibility testing.
