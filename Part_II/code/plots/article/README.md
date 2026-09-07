# Public figure interface

## Purpose

This directory is the public figure-facing interface for Part II. It exposes
publication and validation tools without rerunning the supersonic shooting,
PINN, or atlas campaigns.

## Publication figures and provenance

The final 25 publication PNGs are in [`../../article/assets/`](../../article/assets/).
Their canonical validated sources, source-data locations, hashes, and verified
scientific provenance are recorded in [`../../article/FIGURE_MANIFEST.csv`](../../article/FIGURE_MANIFEST.csv).

## Publication wrappers

[`publish_article_assets.py`](publish_article_assets.py) copies already
validated assets into `article/assets/`. It is publication logic, not a
scientific figure generator. [`validate_article_figures.py`](validate_article_figures.py)
validates the curated package and manifest; it is a validation tool, not a
scientific figure generator.

## Scientific generation

Scientific figure generators currently live mainly under
[`../../plots/`](../../plots/), [`../article/`](../article/), and selected
[`../../code/scripts/`](../../code/scripts/) locations. The manifest field
`plot_script` identifies an upstream scientific generator only when its
provenance has been verified.

Nine figures intentionally use `NO_SCRIPT_FOUND`: no verified upstream
generator has been identified for them. This is honest provenance, not a
missing publication asset. See [`../../docs/PLOTTING_ARCHITECTURE.md`](../../docs/PLOTTING_ARCHITECTURE.md).
