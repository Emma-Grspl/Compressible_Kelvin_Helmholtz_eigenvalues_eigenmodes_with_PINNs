# Part II scientific code

This is the current active implementation area for the supersonic study.

- `src/classical_solver/`: active supersonic classical solver implementation.
- `scripts/`: shooting, training, evaluation, audit, benchmark, and data
  preparation entrypoints.
- `configs/`: current active and legacy campaign configurations.
- `slurm/`: retained HPC launchers and campaign metadata.

`plots/` remains at `../plots/` because it contains preserved scientific
plotting modules. `experiments/` remains at `../experiments/` because current
scripts use it as an experiment input/output root. These locations are
intentional current topology, not duplicated implementations.

The final classical reference method is high-accuracy Riccati shooting. GEP
code is retained as diagnostic material and is not the final reference path.
