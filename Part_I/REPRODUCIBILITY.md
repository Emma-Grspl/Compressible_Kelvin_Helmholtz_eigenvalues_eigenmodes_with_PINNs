# Reproducibility guide

Run commands below from the `Part_I/` directory after completing the
[installation](README.md#installation). The levels intentionally separate
lightweight inspection from calculations that require dense linear algebra or
GPU/HPC resources.

## Level 1 -- Replot published figures

**Purpose.** Recreate a curated figure from its tracked processed data. No
training is performed.

**Inputs.** The relevant files under `article/`, `assets/`, and the plotting
entrypoint recorded in `article/FIGURE_MANIFEST.csv`.

**Example command.**

```bash
python scripts/figures/validate_article_figures.py
```

**Outputs.** An integrity report for the 17 curated PNG assets. To replot a
specific figure, use the exact `plot_script` column in
`article/FIGURE_MANIFEST.csv`; the human-readable mapping is in
[article/README.md](article/README.md).

**Hardware/cost.** CPU; lightweight validation. Individual plotting cost
depends on the selected figure but does not train a model.

## Level 2 -- Classical reference calculations

**Purpose.** Compute a single subsonic classical eigenvalue with the Riccati
shooting reference.

**Inputs.** `alpha`, `Mach`, and optional solver scan/tolerance settings.

```bash
python code/src/scripts/classical/solve_robust_subsonic_shooting.py \
  --alpha 0.5 --mach 0.5
```

**Outputs.** A `RobustSubsonicResult` summary printed by the solver, including
the selected growth-rate result and diagnostics. The same code path is used by
`examples/01_classical_eigenvalue.py`.

**Hardware/cost.** CPU; a single point is lightweight. Dense scans or maps are
more expensive and should be run deliberately.

## Level 3 -- Evaluate pretrained neural charts

**Purpose.** Inspect production atlas routing and evaluate a checkpoint with
the existing atlas code.

**Inputs.** A query `(Mach, eta)`, the routing table under `configs/atlas/`,
and a production checkpoint under `models_saved/production/atlas/N340/`.

```bash
python examples/02_query_neural_atlas.py
```

**Outputs.** The deterministically selected chart metadata and its expected
checkpoint path. Checkpoint deserialization itself is implemented by
`code/src/scripts/gep/selection/audit_mid_joint_pinn_full_gep.py` in
`evaluate_pinn`.

**Hardware/cost.** CPU routing is lightweight. Loading/evaluating a checkpoint
requires PyTorch; GPU is optional for a single evaluation.

## Level 4 -- GEP evaluation

**Purpose.** Diagonalize a dense GEP and inspect selected eigenpair machinery.

**Inputs.** `alpha`, `Mach`, resolution, mapping scale, and optional selection
windows.

```bash
python code/src/scripts/gep/selection/solve_dense_gep_notebook_style.py --help
python code/src/scripts/gep/selection/solve_blumen_exact_joint_gep_v3.py --help
```

**Outputs.** Dense GEP spectra or selected-eigenpair diagnostics, depending on
the entrypoint and its required arguments.

**Hardware/cost.** CPU dense linear algebra; potentially expensive as
resolution increases. A complete production GEP workflow is not yet exposed
as a single public entrypoint, so no inferred command is provided here.

## Level 5 -- Full training

**Purpose.** Retrain the local neural atlas.

**Inputs.** Chart routing, sparse-anchor data, training metadata, and an
appropriate compute environment.

```bash
python code/src/scripts/training/atlas/direct_pinn/train_atlas_chart_joint_ci_mode.py --help
```

**Outputs.** Training artifacts depend on the selected historical training
entrypoint.

**Hardware/cost.** GPU/HPC expected. Full training is not yet exposed as a
single public production command with a compact public configuration; do not
infer one from this guide. The traceable production summaries are in
`configs/training/` and `configs/atlas/`.

## Configuration provenance

Only configuration values directly traceable to tracked source scripts, run
metadata, or production tables are exposed in [configs/README.md](configs/README.md).
They are documentation/configuration records, not a promise that every
historical training launcher is a supported public workflow.
