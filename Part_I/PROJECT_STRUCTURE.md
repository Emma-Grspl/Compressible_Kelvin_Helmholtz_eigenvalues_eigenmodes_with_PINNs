# Project structure

## Article package

- `article/assets/`: exactly the 17 curated Main and Supplementary PNG files.
- `article/FIGURE_MANIFEST.csv`: filename, canonical source, publishing script,
  source data, SHA-256 hash, and scientific note for every figure.
- `scripts/figures/`: one lightweight, unambiguous publishing entrypoint per
  article figure. These scripts copy validated processed assets byte-for-byte;
  they do not retrain a model or recompute a GEP.

## Reproducible code

- `code/src/`: importable numerical and PINN implementation (`PYTHONPATH=code`).
- `code/src/scripts/classical/`: Riccati/shooting reference solvers and
  classical reconstructions.
- `code/src/scripts/training/`: fixed-Mach and atlas training entry points.
- `code/src/scripts/gep/`: GEP selection, resolution, modal refinement, and
  benchmark entry points.
- `code/src/scripts/evaluation/`: quantitative validation and audits.
- `code/src/launch/slurm/`: migrated Jean-Zay launchers.
- `code/plots/scripts/`: scripts whose primary role is figure generation.

## Scientific outputs

- `assets/classic_subsonic/{csv,png,pdf}/`: canonical classical tables and
  figures.
- `assets/pinn_subsonic/{csv,png,pdf}/`: canonical PINN/GEP tables and figures.
- `assets/complementary_audits/final_figures/`: final corrected figures migrated
  from the temporary complementary-audit workspace.
- `results/complementary_audits/`: numerical outputs from audit phases A-I,
  including the curated input tables used by final figures.
- `models_saved/production/`: final fixed-Mach and 49-chart atlas checkpoints.

Supporting, imported, and historical checkpoint trees are documented but not
published in Git. Their exclusion is recorded in
`provenance/PUBLIC_REPOSITORY_EXCLUSIONS.csv`.

## Archive

`archive/` is deliberately separate from executable code. It preserves old
code, configurations, assets, CSV files, and miscellaneous development output
for traceability. Historical filenames may retain old terminology such as
`hybrid` or `frozen`; those names are not the canonical terminology of the
current pipeline.

## Documentation

The `docs/` tables form the migration ledger. In particular:

- `Table_subsonic_migration_manifest.csv` maps every audited source path to its
  actual destination and validation status.
- `Table_checkpoint_inventory.csv` maps duplicate checkpoint paths to one
  physical canonical content.
- `Table_article_figures.csv` keeps manuscript status separate from canonical
  scientific-asset status.
- `SUBSONIC_MIGRATION_FINAL_REPORT.md` records validation results and remaining
  intervention points.
- `provenance/complementary_audits/COMPLEMENTARY_PHASE_INVENTORY.csv` records
  every file from the former temporary workspace, its hash, decision, and
  verified destination.
- `provenance/complementary_audits/COMPLEMENTARY_MIGRATION_REPORT.md` summarizes
  the semantic migration and exclusions.

Two legacy links to Jean-Zay/Lustre data could not be resolved locally. They
are recorded as `EXTERNAL_LINK_UNRESOLVED` and were not replaced by fabricated
local paths.
