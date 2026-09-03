# Part II -- Supersonic Kelvin--Helmholtz spectral branches

## Overview

This directory contains the supersonic part of the compressible
Kelvin--Helmholtz eigenvalue and eigenmode study. It preserves classical
reference data, external Blumen comparisons, sparse complex-eigenvalue
anchors, a piecewise-local neural spectral atlas, modal diagnostics, and
reviewer-control audits.

## Physical problem

Part II considers the supersonic regime of the compressible
Kelvin--Helmholtz non-self-adjoint eigenvalue problem. It reconstructs
complex spectral branches and associated eigenmodes, with Blumen data used as
an external spectral comparison.

## Method

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

## Main results

The repository preserves validated classical Riccati-shooting references,
external Blumen comparisons, modal reconstructions, and curated article
assets. The final high-accuracy eigenvalue and eigenmode are classical
Riccati-shooting outputs. Neural spectral estimates and PINN-informed local
shooting are evaluated as branch-localization tools rather than replacements
for the final classical reference.

## Repository structure

```text
Part_II/
├── article/       # curated 25-figure publication package and manifest
├── assets/        # reusable classical, atlas, modal, and audit assets
├── code/          # active solver, model, physics, config, and execution code
├── experiments/   # preserved experiment-specific outputs below GitHub limits
├── models_saved/  # checkpoint layout documentation; binaries remain local
├── plots/         # preserved scientific plotting code from the source workspace
├── provenance/    # migration and large-file records
├── results/       # semantically migrated complementary reviewer audits
├── scripts/       # figure publication and reviewer-control entry points
├── slurm/         # empty retained HPC-compatible top-level root
├── tests/         # lightweight checks retained from the workspace
└── provenance/review/ # deliberately unresolved review material
```

The current layout is transitional: active classical solvers remain in
`code/src/classical_solver/`, active configurations in `code/configs/`,
scientific plot generators in `plots/`, and active experiment input/output
records in `experiments/`. See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
for the exact current topology.

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

## Quick start

Validate the publication package without recomputing a shooting campaign:

```bash
python scripts/figures/validate_article_figures.py
```

For available public commands and their limitations, see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Reproducibility

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) separates lightweight publication
validation from expensive shooting, production spectral evaluation, and full
training. It does not infer a one-command production workflow where one is not
publicly exposed.

## Article assets

The 25 curated publication figures are in [article/assets](article/assets).
Their canonical sources, hashes, source data, and publication mapping are in
[article/FIGURE_MANIFEST.csv](article/FIGURE_MANIFEST.csv). See
[article/README.md](article/README.md).

## Models and checkpoints

The original Part II checkpoint bank is intentionally omitted from public Git.
Its expected local layout and limitations are documented in
[models_saved/README.md](models_saved/README.md).

## Citation

Final repository citation metadata will be added once publication metadata is
finalized.

## License

License selection will be finalized at repository release.

## Provenance and limitations

`complementary_phase/` no longer exists as a top-level Part II dependency;
its content is recorded in [provenance/complementary_phase_migration.md](provenance/complementary_phase_migration.md).
Files at or above GitHub's 100 MiB limit and the local checkpoint bank are
documented in [provenance/large_files_report.md](provenance/large_files_report.md)
instead of being added to Git.
