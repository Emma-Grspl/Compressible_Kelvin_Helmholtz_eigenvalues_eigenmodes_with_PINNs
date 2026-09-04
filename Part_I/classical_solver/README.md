# Classical solver facade

This directory is a public navigation and documentation facade. It contains no
Python implementation yet. Physical normalization is intentionally deferred to
Phase 8, after ownership and duplication decisions have been made safely.

## Current active implementation

- Canonical subsonic reference: [solve_robust_subsonic_shooting.py](../code/src/scripts/classical/solve_robust_subsonic_shooting.py).
- Primary shooting implementation: [solve_shooting_subsonic.py](../code/src/scripts/classical/solve_shooting_subsonic.py).
- Near-neutral/refined shooting implementation: [solve_mstab17_subsonic_solver.py](../code/src/scripts/classical/solve_mstab17_subsonic_solver.py).
- Dense GEP implementation: [solve_dense_gep_notebook_style.py](../code/src/scripts/gep/selection/solve_dense_gep_notebook_style.py).
- Modal reconstruction is currently workflow-local, principally in the GEP
  selection workflows; it is not yet a standalone canonical module.

The active implementations remain under `code/src/scripts/` because their
current import paths and command-line workflows depend on that source layout.
Archived copies are historical provenance and are not active solver entrypoints.

## Phase 8 ownership questions

- The dense GEP implementation is byte-identical to the active Part II dense
  GEP implementation.
- Part I retains archived historical solver copies that must not be treated as
  current owners.
- A future move requires stable public imports and lightweight fixed-reference
  tests before changing any physical path.

