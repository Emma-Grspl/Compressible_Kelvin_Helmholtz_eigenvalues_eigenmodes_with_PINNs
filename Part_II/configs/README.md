# Curated public configurations

This directory is the canonical, human-readable reusable configuration root
for Part II. It mirrors the role of `Part_I/configs/` without refactoring the
entire supersonic runtime layout.

Phase 8B.1 normalized the five reusable validation and fixed-Mach smoke
configurations into this directory. Current workflows must use these canonical
paths rather than duplicate files under `code/configs/legacy/`.

`CONFIG_MANIFEST.csv` records the canonical path, SHA-256 checksum, and the
status of any former legacy location. The dense classical campaign remains
under `code/configs/legacy/` pending a later phase; it is not normalized by
this batch.

The following source-side areas are intentionally excluded from this facade:

- `../code/configs/anchor_budgets/`: per-chart resolved experiment snapshots;
- `../code/configs/fixed_mach/`: fixed-Mach and neutral-boundary snapshots;
- the remaining `../code/configs/legacy/` material: mixed active, historical,
  and unresolved configuration records.

Some retained Slurm launchers reference absent
`configs/pinn_supersonic/...` paths. This is an existing reproducibility gap;
these launchers require a dedicated audit, and this facade does not guess or
provide replacements for those paths.
