# Part II -- Supersonic Kelvin--Helmholtz spectral branches

This directory contains the supersonic part of the compressible
Kelvin--Helmholtz eigenvalue and eigenmode study. It preserves classical
reference data, external Blumen comparisons, sparse complex-eigenvalue
anchors, a piecewise-local neural spectral atlas, modal diagnostics, and
reviewer-control audits.

## Scientific workflow

1. Classical supersonic Riccati shooting builds the reference solution.
2. Blumen data provide an external spectral comparison.
3. Sparse complex eigenvalue anchors supply branch-localization information.
4. A piecewise-local neural spectral atlas is routed deterministically.
5. Physics-informed modal constraints act on the modal/differential
   representation.
6. Direct neural spectral predictions provide local information.
7. PINN-informed local Riccati shooting is assessed against generic controls.
8. Classical Riccati shooting returns the final high-accuracy eigenvalue and
   eigenmode reconstruction.
9. Independent physics, routing, threshold, and branch-localization audits
   remain available as reviewer controls.

Sparse classical anchors localize the branch; physics-informed terms constrain
the modal representation. They do not establish a measurable spectral
generalization or branch-localization improvement by themselves. The matched
anchors-only comparison is retained for that distinction. Atlas routing is
deterministic and piecewise: global continuity between charts is not imposed.

## Layout

```text
Part_II/
├── article/       # curated 25-figure publication package and manifest
├── assets/        # reusable classical, atlas, modal, and audit assets
├── code/          # solver, model, physics, configuration, and execution code
├── experiments/   # preserved experiment-specific outputs below GitHub limits
├── models_saved/  # checkpoint layout documentation; binaries remain local
├── plots/         # preserved scientific plotting code from the source workspace
├── provenance/    # migration and large-file records
├── results/       # semantically migrated complementary reviewer audits
├── scripts/       # figure publication and reviewer-control entry points
├── slurm/         # retained HPC launch material where available
├── tests/         # lightweight checks retained from the workspace
└── _TO_REVIEW/    # deliberately unresolved material
```

## Installation

From `Part_II/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/code:$PWD/code/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The original HPC environment remains authoritative for exact numerical
reproduction. No package versions are inferred here.

## Article package

The 25 curated publication figures are in [article/assets](article/assets).
Their canonical sources, hashes, source data, and publication mapping are in
[article/FIGURE_MANIFEST.csv](article/FIGURE_MANIFEST.csv). See
[article/README.md](article/README.md).

## Provenance and limitations

`complementary_phase/` no longer exists as a top-level Part II dependency;
its content is recorded in [provenance/complementary_phase_migration.md](provenance/complementary_phase_migration.md).
Files at or above GitHub's 100 MiB limit and the local checkpoint bank are
documented in [provenance/large_files_report.md](provenance/large_files_report.md)
instead of being added to Git.
