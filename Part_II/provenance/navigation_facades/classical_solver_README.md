# Classical solver facade

This directory is a public navigation and documentation facade. It contains no
Python implementation yet. Physical normalization is intentionally deferred to
Phase 8, after ownership and duplication decisions have been made safely.

## Current active implementations

Part II has two active roles that must remain distinct for now.

### Generic active solver package

- Generic Riccati shooting: [shooting_supersonic.py](../code/src/classical_solver/supersonic/shooting_supersonic.py).
- Generic refined shooting: [mstab17_supersonic_solver.py](../code/src/classical_solver/supersonic/mstab17_supersonic_solver.py).
- Dense GEP diagnostic: [dense_gep_notebook_style.py](../code/src/classical_solver/gep/dense_gep_notebook_style.py).

### Validated/frozen reference workflow

[classic_supersonic_reference](../code/src/classic_supersonic_reference/) contains
the frozen validated reference implementation used by current final-reference
and modal-reconstruction workflows. Its modal reconstruction is implemented in
[modal_reconstruction.py](../code/src/classic_supersonic_reference/validation/modal_reconstruction.py).

The generic and frozen Mstab17 implementations are semantically related but
are not byte-identical and are not interchangeable. Phase 7B does not select a
sole owner.

## Phase 8 ownership questions

- The dense GEP implementation is byte-identical to the active Part I dense
  GEP implementation.
- Three Part II Blumen-reference modules are byte-identical.
- Two Part II modal-reconstruction modules are byte-identical.
- An exact frozen Mstab17 copy remains in an evaluation path.
- The generic and frozen Mstab17 implementations are related but non-identical.
- Archived/historical solver copies remain provenance, not active ownership.

