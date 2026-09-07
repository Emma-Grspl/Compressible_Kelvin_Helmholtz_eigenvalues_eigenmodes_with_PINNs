# HPC launcher navigation

## Purpose

This directory is a public navigation and documentation facade. It does not
duplicate the research launcher collection and is not itself a runtime
launcher tree.

## Authoritative launcher collection

The current Part II research launchers remain in
[`code/slurm/`](../../code/slurm/). They cover classical Riccati-shooting
reference construction, neutral-boundary continuation, fixed-Mach PINN
experiments, modal reconstruction, and atlas campaigns.

Many jobs are Jean-Zay-specific: they assume Slurm, site modules, GPU or CPU
allocations, and the historical HPC working-tree layout. They may also require
seed data, prior campaign outputs, or external checkpoints. Consequently, not
every retained launcher is intended to run from a fresh public clone.

## Known public limitations

The original Part II checkpoint bank is intentionally omitted from public Git;
see [`../../models_saved/README.md`](../../models_saved/README.md). Eight
distinct configuration templates under `configs/pinn_supersonic/...` are also
absent. No safe replacement has been identified, and resolved configuration
snapshots must not be substituted automatically.

## Reproducibility status

[`HPC_MANIFEST.csv`](HPC_MANIFEST.csv) describes launcher families and their
reproducibility status. The manifest distinguishes public source availability
from external inputs, private checkpoints, missing configurations, and
historical-only material. It does not replace
[`../../REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md).

The GEP launcher family is historical diagnostic material and now resides in
[`../../provenance/hpc/legacy_gep/`](../../provenance/hpc/legacy_gep/). The
final classical reference remains high-accuracy Riccati shooting.
