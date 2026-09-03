# Classical supersonic reference v2 — showcase assets

This directory contains the compact article-facing outputs associated with the
classical supersonic Kelvin–Helmholtz reference.

## Contents

### `ci/`

- Digitized Blumen `ci` points.
- Frozen spectral points superposed on the digitized Blumen data.
- Blumen points are displayed as scatter points and are not connected.

### `modes/`

- Complete modal PDF for all frozen spectral points.
- Raw versus tail-polished modal review PDF.
- Modal plots include both `y < 0` and `y > 0`.

### `tables/`

- Compact spectral table for the 183 frozen reference points.

The complete implementation, configurations, validation scripts and provenance
are stored in the repository-level directory:

`classic_supersonic/`

## Validated 50-point selection

The following compact assets summarize the validated 50-point modal selection:

- `ci/supersonic_reference_v2_ci_validated_50pts.png`;
- `modes/supersonic_reference_v2_modes_validated_50pts.pdf`;
- `tables/supersonic_reference_v2_spectral_validated_50pts.csv`;
- `tables/supersonic_reference_v2_spectral_validated_50pts_summary.json`.

The complete modal-field CSV is intentionally not distributed through Git
because it exceeds the repository size limit. It remains a local
reproducibility result under `classic_supersonic/reproducibility/results/`.

The Blumen digitized points are scatter data and must not be connected by
plotting lines.
