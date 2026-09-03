# Curated public configurations

This directory is the stable, human-readable configuration facade for Part II.
It mirrors the role of `Part_I/configs/` without refactoring the current
supersonic runtime layout.

The authoritative configuration store for current executable workflows remains
[`../code/configs/`](../code/configs/). The files here are verified public
copies for production, validation, and one retained experimental smoke case;
they are not loaded automatically by runtime code.

`CONFIG_MANIFEST.csv` maps every public record to its runtime source and
records the SHA-256 checksum. Each public copy must remain byte-identical to
its listed source.

The following source-side areas are intentionally excluded from this facade:

- `../code/configs/anchor_budgets/`: per-chart resolved experiment snapshots;
- `../code/configs/fixed_mach/`: fixed-Mach and neutral-boundary snapshots;
- the remaining `../code/configs/legacy/` material: mixed active, historical,
  and unresolved configuration records.

Some retained Slurm launchers reference absent
`configs/pinn_supersonic/...` paths. This is an existing reproducibility gap;
these launchers require a dedicated audit, and this facade does not guess or
provide replacements for those paths.
