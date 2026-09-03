# Repository structure

This repository contains two scientifically distinct studies of the
compressible Kelvin--Helmholtz eigenvalue problem:

- `Part_I/`: subsonic spectral branches;
- `Part_II/`: supersonic spectral branches.

Both parts are intended to converge toward the same public organizational
contract. This document defines the semantic role of that contract; it does
not claim that both parts already use identical filesystem layouts.

```text
README.md
PROJECT_STRUCTURE.md
REPRODUCIBILITY.md
requirements.txt
article/
assets/
classical_solver/
code/
configs/
docs/
examples/
models_saved/
provenance/
results/
scripts/
tests/
```

## Shared semantic contract

- `article/`: final publication assets, the figure manifest, and
  article-facing documentation.
- `assets/`: reusable scientific assets and processed inputs.
- `classical_solver/`: classical numerical solver implementations and
  reference calculations.
- `code/`: core scientific implementation.
- `configs/`: curated public or final configurations.
- `docs/`: human-readable technical and scientific documentation.
- `examples/`: small executable demonstrations of public workflows.
- `models_saved/`: distributed production checkpoints, or documentation of
  intentionally omitted checkpoint binaries.
- `provenance/`: migration records, historical material, review/audit traces,
  excluded files, HPC history, and other traceability records.
- `results/`: scientific numerical outputs, validation tables, and processed
  results.
- `scripts/`: public executable analysis, plotting, workflow, and launcher
  entrypoints.
- `tests/`: lightweight tests for repository integrity and public workflows.

## Current transitional layout

Part I already exposes much of this contract at its top level. Its active
subsonic classical implementation currently lives under
`Part_I/code/src/scripts/classical/`, not under the empty compatibility root
`Part_I/classical_solver/`.

Part II is intentionally documented as transitional. Its active supersonic
solver implementation is under `Part_II/code/src/classical_solver/`; its
training, shooting, audit, evaluation, and preparation scripts are under
`Part_II/code/scripts/`; its configurations remain under
`Part_II/code/configs/`; its scientific plotting modules remain under
`Part_II/plots/`; and `Part_II/experiments/` is an active experiment
input/output root. These locations are current facts, not aliases created by
this document.

The two parts therefore share a common documentation contract while retaining
their validated, regime-specific workflows and current path dependencies.
