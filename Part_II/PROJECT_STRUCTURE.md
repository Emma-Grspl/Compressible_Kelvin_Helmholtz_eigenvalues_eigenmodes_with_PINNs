# Part II project structure

## `article/`

The curated 25-figure publication package. `assets/` contains only final Main
and Supplementary figures; `FIGURE_MANIFEST.csv` maps them to hashes, sources,
and the common publication entrypoint.

## `assets/`

The reusable scientific asset collection from the supersonic workspace:
classical references, Blumen validation, atlas and anchor data, modal outputs,
PINN diagnostics, and the canonical sources of curated article figures.

## `code/`

Reusable classical solver, physics, model, configuration, training, shooting,
evaluation, audit, and legacy diagnostic code. Numerical algorithms were not
changed during repository integration.

## `experiments/`

Experiment-specific outputs preserved below the public file-size threshold.
Large raw legacy exports are not added to Git and are listed in provenance.

## `results/`

Complementary reviewer audits migrated by scientific role: physics residuals,
threshold sensitivity, multiseed ablation, matched branch localization,
failure basin, routing, computational cost, and Blumen `c_r` validation.

## `scripts/`

`figures/` publishes the curated article package from validated assets.
`reviewer_controls/` retains complementary reviewer-audit scripts distinctly
from production entrypoints.

## `plots/`

The original figure-generator modules remain preserved here because their
scientific plotting logic is still useful. The publication package is exposed
through `scripts/figures/` rather than duplicating those computations.

## `models_saved/`

Documents the expected local checkpoint layout. The original binary checkpoint
bank remains intentionally outside public Git.

## `provenance/`

Migration records, large-file exclusions, and historical records. Historical
paths may be retained here without becoming active execution dependencies.

## `_TO_REVIEW/`

Ambiguous files, missing-input snapshots, and broken launchers retained without
destructive cleanup.
