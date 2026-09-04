# Plotting architecture

## Public interface

[`../scripts/figures/`](../scripts/figures/) is the public figure-facing
interface. It publishes and validates the curated article package; it is not a
replacement for the scientific plotting workflow.

## Reusable plotting utilities

Importable helper functions exist where the workflow benefits from them. For
example, `code/plots/scripts/pinn_subsonic/utils_asset_common.py` contains
shared plotting, data-selection, and serialization helpers. The repository does
not yet enforce a single utility-only directory for every reusable helper.

## Scientific figure generators

Scientific generators remain primarily under `code/plots/` and in selected
solver, evaluation, and analysis scripts under `code/src/`. These scripts may
process data, construct diagnostics, and write intermediate or scientific
figures. Their current locations preserve valid research-workflow provenance.

## Publication wrappers

`scripts/figures/plot_Fig*.py` are publication wrappers. They use
`_publish_validated_asset.py` to copy canonical validated PNGs into
`article/assets/` without recomputing science. In the Part I figure manifest,
`plot_script` refers to these wrappers.

## Analysis scripts

`scripts/analysis/` contains analyses whose principal result is numerical or
diagnostic. Such scripts may also create figures, but are not automatically
article-figure generators.

## Historical plotting provenance

Historical and source-tree plotting material is retained under `code/plots/`.
Some embedded source trees and compatibility bridges are intentionally kept in
place. The repository does not claim that the current physical layout fully
matches an ideal future plotting taxonomy, and no plotting-code migration is
currently required.
