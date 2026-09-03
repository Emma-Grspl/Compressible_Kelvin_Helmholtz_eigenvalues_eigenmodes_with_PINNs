# Curated public configurations

This directory exposes only configuration records needed to interpret the Part
I production workflow. They are traceable copies or summaries of tracked
source files, not a migration of every historical experiment.

- `classical/` contains exact copies of coupled shooting convergence YAML
  files from `code/src/scripts/utils/`.
- `atlas/` contains the machine-independent routing fields for the 49-chart
  `N340` atlas, derived from the tracked production training plan.
- `gep/` documents the resolution policy encoded in the production selection
  script.
- `training/` records the tracked `N340` anchor-budget metadata.

Every generated summary identifies its source and intentionally omits
machine-specific absolute paths.
