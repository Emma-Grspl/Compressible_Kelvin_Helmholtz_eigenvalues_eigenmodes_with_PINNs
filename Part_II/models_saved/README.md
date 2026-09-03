# Local checkpoint bank

The original supersonic workspace contains a local checkpoint bank for fixed-
Mach studies, atlas budgets, modal reconstruction, and reviewer controls.
These checkpoint files are intentionally not copied into this public Git tree:
they are local binary artifacts and are not required to inspect the curated
article figures or run the lightweight static checks.

Their original locations and sizes are recorded in
`provenance/large_files_report.md`. Reproduction that requires checkpoint
evaluation must obtain the checkpoint bank separately and place it under this
directory following the original workspace layout.

This directory is documentation of intentionally omitted binaries, not an
active distributed checkpoint set. The authoritative public spectral and modal
reference data are in `../assets/`, `../experiments/`, and `../results/`.
