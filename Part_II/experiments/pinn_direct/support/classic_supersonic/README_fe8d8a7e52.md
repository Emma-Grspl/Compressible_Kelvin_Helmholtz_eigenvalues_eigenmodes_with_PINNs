# Classical supersonic Kelvin–Helmholtz reference

This directory contains the reproducible classical supersonic reference used
for Kelvin–Helmholtz PINN experiments.

## Version

2.0.0

## Reference content

- 183 unique spectral points
- Mach range: 1.10 to 1.90
- Modal fields: p, rho, u and v
- 838847 raw modal rows
- 838847 tail-polished modal rows

## Structure

- `src/`: reusable solver, I/O, plotting and validation code
- `scripts/`: executable build, campaign, plotting and validation scripts
- `configs/`: numerical, plotting and Jean Zay configurations
- `data/`: spectral, Blumen and modal reference data
- `tests/`: unit, regression and smoke tests
- `reproducibility/`: commands and software environments
- `provenance/`: manifests, checksums, limitations and migration records
- `slurm/`: Jean Zay batch scripts
- `tools/`: release and repository maintenance tools

The compact article-facing figures and tables are located in:

`assets/classic_supersonic/reference_v2/`

## Scientific convention

The raw confirmed modal fields are the primary scientific reference.

The tail-polished fields are a documented derived export. Tail polishing does
not modify the spectral values `cr` or `ci`.

Digitized Blumen points are represented as scatter points and must never be
connected by plotting lines.

## Bundle structure and asset locations

`classic_supersonic/` is the complete reproducible scientific bundle.

It contains:

- `src/`: the frozen classical supersonic solver and validation utilities;
- `scripts/`: build, campaign, plotting and validation scripts;
- `configs/`: solver, reproducibility and Jean-Zay configurations;
- `tests/`: spectral, modal, asset and witness regression tests;
- `data/`: spectral, Blumen, modal and compact witness datasets;
- `assets/`: figures, modal PDFs and compact tables distributed with the bundle;
- `reproducibility/`: commands and locally generated validation results;
- `provenance/`: manifests, checksums and scientific validation summaries;
- `slurm/`: archived or reusable HPC launch scripts.

The lightweight asset showcase is maintained separately in
`assets/classic_supersonic/`.

The complete local modal Parquet datasets remain untracked because of their
size. The tracked witness sample and metadata provide a compact reproducibility
check.
