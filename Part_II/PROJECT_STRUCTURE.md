# Part II project structure

Part II is the supersonic study. It has a validated but transitional layout:
several active components still live under `code/`, `plots/`, `experiments/`,
and `code/configs/` rather than under the future common top-level contract.

## Publication and reusable assets

- `article/`: the curated 25-figure publication package. `article/assets/`
  contains final figures and `FIGURE_MANIFEST.csv` records their canonical
  source, source data, SHA-256 hash, and upstream scientific plotting
  provenance when known.
- `assets/`: reusable classical references, Blumen validation, atlas and anchor
  data, modal outputs, PINN diagnostics, and canonical sources of article
  figures.
- `configs/`: curated public configuration facade. Its copies are mapped and
  hash-checked against current runtime sources; it is not itself the runtime
  configuration store.

The manifest does not imply a single common scientific plotting entrypoint.
Publication wrappers copy or publish authoritative assets; upstream generator
provenance is recorded separately when it is known. `NO_SCRIPT_FOUND` is an
intentional honest record for figures whose upstream script cannot be assigned
without guessing.

## Current active code topology

- `code/src/classical_solver/`: **current active supersonic solver
  implementation**.
- `code/scripts/`: training, shooting, audits, evaluation, benchmarks, and
  data-preparation entrypoints.
- `code/configs/`: current active and legacy campaign configurations.
- `code/slurm/`: retained non-GEP HPC launcher and campaign material.
- `plots/`: current scientific plotting code retained in its source-workspace
  location.
- `scripts/`: publication and reviewer-control entrypoints. In particular,
  `scripts/figures/` publishes curated assets without duplicating scientific
  computations.

Numerical algorithms were not changed during repository integration. The final
classical reference is high-accuracy Riccati shooting; GEP material is retained
as diagnostic material rather than as the final reference workflow.

The public `configs/` facade intentionally excludes mixed source-side snapshot
families. Current executable workflows continue to resolve configuration paths
under `code/configs/` unless a script documents otherwise.

## Experiments and results

- `experiments/`: current active experiment input/output root, organized by
  scientific campaign. It is preserved in place because active scripts refer
  to this location.
- `results/`: processed complementary reviewer-control outputs, including
  physics residuals, threshold sensitivity, multiseed ablation, matched branch
  localization, failure basin, routing, computational cost, and Blumen `c_r`
  validation.

## Models and provenance

- `models_saved/`: documentation of the expected local checkpoint layout. The
  original binary checkpoint bank is intentionally outside public Git.
- `provenance/`: migration, review, large-file exclusion, and historical
  traceability. Historical paths may appear here without becoming active
  execution dependencies.
- `provenance/review/`: ambiguous files, missing-input snapshots, and broken
  launchers retained without destructive cleanup. This material is not part of
  the active production workflow.
- `provenance/hpc/legacy_gep/`: historical GEP launcher family, preserved for
  diagnostic traceability rather than as the final reference workflow.
- `scripts/hpc/`: public navigation facade and family-level HPC manifest; it
  documents the source-side launcher collections without duplicating them.
