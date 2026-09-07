# Plotting architecture

## Public interface

[`../scripts/figures/`](../scripts/figures/) is the public figure-facing
interface. It publishes and validates the curated article package; it is not a
replacement for the scientific plotting workflow.

## Reusable plotting utilities

Part II has importable plotting packages under `plots/`, but most current files
are executable research generators rather than utility-only modules. The
repository does not yet enforce a single utility-only directory for every
reusable helper.

## Scientific figure generators

Scientific figure generators currently live mainly under `plots/`,
`scripts/article/`, and selected `code/scripts/` data-preparation or evaluation
locations. They process scientific inputs and may write canonical or
intermediate figures. These locations are retained because their relative-path
and workflow assumptions are part of the preserved implementation.

## Publication wrappers

`scripts/figures/publish_article_assets.py` copies validated assets into
`article/assets/`; it does not generate scientific results. The article manifest
uses `plot_script` for a verified upstream generator when known, and otherwise
uses `NO_SCRIPT_FOUND` without inventing provenance.

## Analysis scripts

Part II analysis, audit, and evaluation scripts are mainly retained under
`code/scripts/`. They may generate diagnostic figures while their main product
remains numerical validation, reference construction, or campaign analysis.

## Historical plotting provenance

Historical and diagnostic plotting sources remain in their existing locations,
including unreferenced variants under `plots/`. The repository does not claim
that the current physical layout fully matches an ideal future plotting
taxonomy, and no plotting-code migration is currently required.
