# Part I: subsonic compressible Kelvin-Helmholtz spectral branches

This directory contains the subsonic part of the compressible
Kelvin-Helmholtz eigenvalue study. It preserves the reproducible numerical
pipeline, validated article assets, complementary audits, and historical
provenance needed to interpret the results.

## Scientific scope

The classical reference solves the subsonic eigenproblem with Riccati shooting
and validates the reconstructed growth-rate map against digitized Blumen
isolines. Fixed-Mach experiments study spectral branch identification and modal
reconstruction. The parametric piecewise-local neural atlas covers 49 charts
in `(alpha, Mach)` with a global sparse-eigenvalue budget of `N340` for the
production configuration.

The central methodological distinction is:

- sparse scalar eigenvalues localize the spectral branch;
- physics constrains spectral-modal coupling and the direct neural mode;
- a dense GEP is diagonalized independently;
- the scalar neural estimate selects an eigenpair only after diagonalization.

The physics-only fixed-Mach experiment is retained as a branch-identification
failure baseline. It must not be interpreted as evidence that physics alone
systematically improves interpolation of `c_i`. Direct neural prediction does
not replace the classical reference or the selected GEP eigenpair.

## Experiment families

- Fixed-Mach physics-only `A0` baseline.
- Fixed-Mach four-anchor Stage 0 `A4` for sparse spectral localization.
- Fixed-Mach Stage 1 for physics-constrained modal reconstruction without
  direct modal supervision.
- Parametric atlas with budgets `N96`, `N224`, `N340`, `N520`, `N640`, and
  `N705`.
- Direct-neural, anchors-only, seed-sensitivity, holdout, overlap, modal,
  near-neutral, dense-audit, runtime, and scalar-guided GEP selection audits.

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the experiment inventory and
[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the directory conventions.

## Installation

Create an environment appropriate for the target CPU/GPU platform, then:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
```

No package versions are invented here. Reproduce the documented Jean-Zay
environment when exact HPC compatibility is required.

## Main entry points

Classical reference:

```bash
python -m src.scripts.classical.solve_robust_subsonic_shooting --help
python -m src.scripts.classical.solve_classic_subsonic_gep_shooting_scan --help
```

Fixed-Mach PINN:

```bash
python code/src/scripts/training/fixed_mach/physics_only/train_kh_subsonic_pinn.py --help
python code/src/scripts/training/fixed_mach/physics_only/train_kh_subsonic_ci_stage0_anchor_lock.py --help
```

Atlas and GEP:

```bash
python code/src/scripts/training/atlas/direct_pinn/train_atlas_chart_joint_ci_mode.py --help
python -m src.scripts.gep.selection.solve_dense_gep_notebook_style --help
```

These commands only identify entry points. Consult the experiment inventory
and Slurm launchers before starting expensive calculations.

## Checkpoints and assets

`models_saved/production/` contains checkpoints directly associated with the
final pipeline. Supporting and historical checkpoint families remain listed in
the migration provenance but are intentionally excluded from this public Git
repository because they are not required by the supported execution path.

Canonical figures and tables are under `assets/classic_subsonic/`,
`assets/pinn_subsonic/`, and `assets/complementary_audits/`. The definitive
curated manuscript set is `article/assets/`; every entry is documented in
`article/FIGURE_MANIFEST.csv` and has a publishing entrypoint under
`scripts/figures/`.

## Reproducibility

Migration provenance, SHA-256 values, exclusions, checkpoint aliases, and
complementary-audit decisions are recorded under `docs/` and `provenance/`.
Historical files are kept under `archive/` and are not part of the supported
execution path.
