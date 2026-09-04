# Experiment workspace

`experiments/` is the active Part II research campaign workspace. It is
organized by scientific campaign and currently contains both run-specific input
data and derived campaign outputs. Current workflows still read from and write
to paths below this directory, so it is intentionally retained in place.

This directory is not equivalent to [`../results/`](../results/). `results/`
contains publication-facing and canonical validated outputs where that
normalization has already been completed. [`../assets/`](../assets/) contains
stable reusable scientific inputs and canonical figure sources where those have
already been normalized. [`../provenance/`](../provenance/) contains historical
or frozen material where it has already been classified.

Some experiment-side outputs have verified canonical copies under `results/`:

- threshold-sensitivity outputs are represented under
  `results/threshold_sensitivity/`;
- physics-residual-audit outputs are represented under
  `results/physics_residual_audit/`.

The experiment-side copies are retained because they record campaign context
and because the active input/output dependencies have not yet been separated.
They must not be removed or deduplicated as part of documentation work.

The family-level status, known public canonical locations, article provenance,
and active consumers are recorded in
[`EXPERIMENT_REGISTRY.csv`](EXPERIMENT_REGISTRY.csv). Current article provenance
still directly references `experiments/` for Main 1, Main 6--9, Main 14, and
Supplementary S4c--S4d; those paths remain part of the validated figure
manifest contract.

Part I normalizes more material into `assets/`, `results/`, and
`models_saved/`, and does not have an equivalent `experiments/` runtime root.
This is an intentional implementation difference, not a scientific
inconsistency. Part II remains transitional until input and output dependencies
can be separated safely.
