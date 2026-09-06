# HPC launcher navigation

## Purpose

This directory is a public navigation and documentation facade. It does not
duplicate the research launcher collection and is not itself a runtime
launcher tree.

## Authoritative launcher collection

The original Part I launchers remain in
[`code/src/launch/slurm/`](../../code/src/launch/slurm/). They record the
subsonic Kelvin-Helmholtz research campaigns used during development,
including classical-asset generation, fixed-Mach PINN studies, and staged
hybrid `c_i` experiments.

Many jobs are Jean-Zay-specific: they assume Slurm, site modules, GPU or CPU
allocations, and an HPC working-tree layout. They may also require classical
reference assets, seed data, or outputs from an earlier campaign. Therefore,
not every historical launcher is intended to run from a fresh public clone.

## Reproducibility status

[`HPC_MANIFEST.csv`](HPC_MANIFEST.csv) documents launcher families and their
status. The manifest distinguishes supported public interfaces from families
requiring external inputs, private checkpoints, missing configurations, or
historical-only execution context. It does not replace the Part I
reproducibility guide in [`../../REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md).

The existing launcher files remain authoritative until a later, explicit
promotion creates a portable public HPC workflow.
