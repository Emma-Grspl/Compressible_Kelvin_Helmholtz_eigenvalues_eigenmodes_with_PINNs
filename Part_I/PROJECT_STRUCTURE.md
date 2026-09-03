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
- `code/src/scripts/training/`: fixed-Mach and atlas training entrypoints.
- `code/src/scripts/gep/`: GEP selection, resolution, modal refinement, and
  benchmark entrypoints.
- `code/src/scripts/evaluation/`: quantitative validation and audits.
- `code/src/launch/slurm/`: retained Jean-Zay launchers.
- `code/plots/scripts/`: scientific figure-generation modules.
- `scripts/`: public analysis, paper, and publication-figure entrypoints.
- `examples/`: small repository-relative CPU demonstrations; they do not train
  a model or run a dense GEP sweep.
- `tests/`: lightweight integrity and classical single-point tests used by CI.

## Configurations, models, and documentation

- `configs/`: curated public copies or summaries of validated classical,
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
- `classical_solver/`, `KH_RT_Blumen/`, and `src/`: currently empty
  compatibility or legacy-looking roots. They are retained without claiming
  that they contain active implementation.

Two legacy Jean-Zay/Lustre links remain recorded as
`EXTERNAL_LINK_UNRESOLVED`; they were not replaced by fabricated local paths.
