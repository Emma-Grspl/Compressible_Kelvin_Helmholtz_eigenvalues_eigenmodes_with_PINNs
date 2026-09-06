# Part I project structure

Part I is the subsonic study. Its active public implementation is centred on
`code/`; the layout below describes the current repository, not a future
normalized layout.

## Publication, assets, and results

- `article/`: exactly 17 curated Main and Supplementary PNG assets,
  `FIGURE_MANIFEST.csv`, and article-facing documentation. Its figure scripts
  are publishing entrypoints that copy validated processed assets byte-for-byte;
  they do not retrain a model or recompute a GEP.
- `assets/`: canonical classical, PINN/GEP, and complementary-audit processed
  inputs and figures.
- `results/`: tracked validation tables, numerical audit outputs, and retained
  normalization or representation ablations.

## Active code and public workflows

- `code/src/`: importable numerical and PINN implementation (`PYTHONPATH=code`).
- `code/src/scripts/classical/`: **current active subsonic classical Riccati
  shooting solvers and reconstructions**.
- `classical_solver/`: public navigation/documentation facade for the current
  classical interface. It contains no implementation code yet; physical
  normalization is deferred until Phase 8.
- `code/src/scripts/training/`: fixed-Mach and atlas training entrypoints.
- `code/src/scripts/gep/`: GEP selection, resolution, modal refinement, and
  benchmark entrypoints.
- `code/src/scripts/evaluation/`: quantitative validation and audits.
- `code/slurm/`: supported N340 seam, modal-asset, and runtime benchmark
  launchers. Historical Jean-Zay campaign launchers remain under
  `code/src/launch/slurm/`.
- `code/plots/article/`: public publication wrappers and figure-package
  validation. They publish the tracked validated PNGs byte-for-byte.
- `code/plots/generators/`: final-article data and figure-generation helpers.
- `code/plots/scripts/`: retained historical plotting provenance and
  compatibility wrappers.
- `experiments/`: scientific audits and supporting experiments, organized by
  conclusion rather than publication figure.
- `scripts/`: retained analysis and provenance utilities not yet in the public
  plotting interface.
- `examples/`: small repository-relative CPU demonstrations; they do not train
  a model or run a dense GEP sweep.
- `tests/`: lightweight integrity and classical single-point tests used by CI.

## Configurations, models, and documentation

- `code/configs/`: curated public copies or summaries of validated classical,
  atlas-routing, GEP-policy, and `N340` anchor-budget configurations.
- `models_saved/production/`: final fixed-Mach and 49-chart atlas checkpoints.
  `models_saved/CHECKPOINT_MANIFEST.csv` is the file-level size and SHA-256
  inventory. Excluded supporting or historical checkpoint trees are recorded
  in `provenance/PUBLIC_REPOSITORY_EXCLUSIONS.csv`.
- `docs/`: technical notes, protocols, migration tables, and scientific
  documentation.
- `provenance/`: migration records, public-repository exclusions, and
  complementary-audit traceability.

## Historical and compatibility material

- `archive/`: deliberately separate historical code, configurations, assets,
  CSV files, and development outputs. Historical names such as `hybrid` or
  `frozen` are not canonical terminology for the current pipeline.
- `KH_RT_Blumen/` and `src/`: currently empty compatibility or legacy-looking
  roots. They are retained without claiming that they contain active
  implementation. `classical_solver/` is now a documentation facade only; the
  active solver code remains under `code/src/scripts/classical/` and
  `code/src/scripts/gep/selection/` pending Phase 8.

Two legacy Jean-Zay/Lustre links remain recorded as
`EXTERNAL_LINK_UNRESOLVED`; they were not replaced by fabricated local paths.
