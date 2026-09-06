# Public figure interface

## Purpose

This directory is the public figure-facing interface for Part I. It exposes
safe publication and validation entrypoints without rerunning the subsonic
classical solvers, GEP calculations, or PINN training campaigns.

## Publication figures and provenance

The final 17 publication PNGs are in [`../../article/assets/`](../../article/assets/).
Their canonical validated sources, source-data locations, hashes, and public
entrypoints are recorded in [`../../article/FIGURE_MANIFEST.csv`](../../article/FIGURE_MANIFEST.csv).

## Publication wrappers

Every `plot_Fig*.py` file is a publication wrapper. It copies an already
validated canonical PNG into `article/assets/`; it does not recompute a
scientific result. [`_publish_validated_asset.py`](_publish_validated_asset.py)
contains the shared byte-for-byte copy and SHA-256 verification logic.

For Part I, the manifest field `plot_script` deliberately identifies these
publication wrappers. It does not claim that they are the upstream scientific
generators. Their docstrings record the original generator when that provenance
is known.

## Scientific generation

Scientific generators and their historical plotting sources remain in
[`../../code/plots/`](../../code/plots/) and related retained source-tree
material. Their layout reflects the research workflow and is intentionally not
physically normalized. See [`../../docs/PLOTTING_ARCHITECTURE.md`](../../docs/PLOTTING_ARCHITECTURE.md).

[`validate_article_figures.py`](validate_article_figures.py) validates the
curated package and manifest; it is a validation tool, not a figure generator.
