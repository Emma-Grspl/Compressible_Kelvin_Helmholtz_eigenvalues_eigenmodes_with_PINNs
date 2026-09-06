# Part I project structure

Part I is the subsonic study. Public, maintained implementation is contained
under `code/`; experiment outputs and historical records are deliberately
separate.

## Publication and reproducibility

- `article/`: the 17 curated manuscript PNG assets, their manifest, and
  publishing wrappers. The wrappers publish validated data without retraining
  or recomputing a GEP.
- `assets/`: canonical classical, PINN/GEP, and publication-facing processed
  inputs and figures.
- `models_saved/`: production checkpoints and their integrity manifest.
- `tests/`: lightweight integrity and classical single-point regression tests.

## Maintained code

- `code/configs/`: validated routing, GEP, classical, and anchor-budget
  configurations.
- `code/src/`: importable numerical, classical, PINN, GEP, and evaluation
  implementation (`PYTHONPATH=code`).
- `code/plots/article/`: article publishing wrappers and the figure-package
  validator.
- `code/plots/generators/`: maintained data and figure generators.
- `code/slurm/`: supported N340 seam, modal-asset, and runtime launchers.

## Experiments and provenance

- `experiments/`: supporting scientific experiments and their retained audit
  outputs, including the complementary audits and legacy ablations.
- `provenance/`: migration records, protocols, exclusions, historical plotting
  and Slurm collections, and other non-public-workflow material.

Historical names recorded beneath `provenance/` are not current runtime
dependencies. The root contains no compatibility facades or duplicate code
trees.
