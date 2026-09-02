# Complementary-audit migration report

## Source and scope

- Source repository commit: `0df43cee63bcf4be20dab5d7da7b4f30c381153b`.
- Temporary source tree: `complementary_phase/` (untracked in the source
  worktree and left unchanged).
- Files inventoried: 11,846.
- Useful files migrated and hash-verified: 1,535.
- Files requiring manual review: 0.

## Semantic destinations

- Phase A-I and F-plus numerical outputs: `results/complementary_audits/`.
- Curated CSV inputs formerly under `assets/`:
  `results/complementary_audits/curated/`.
- Corrected final figures: `assets/complementary_audits/final_figures/`.
- Current analysis and plotting scripts:
  `scripts/analysis/complementary_audits/`.
- Methodological table and environment note: `docs/complementary_audits/`.
- Original manifest and complete inventory: `provenance/complementary_audits/`.

## Exclusions

- 10,291 virtual-environment files: local environment, reproducible from the
  recorded requirements and not suitable for Git.
- 4 transient log files.
- 2 macOS metadata files.
- 8 script backups superseded by current scripts.
- 6 superseded figure assets, including pre-title-fix copies and the old GEP
  convergence image. The selected GEP figure is the corrected `FIXED` PNG.

No scientific numerical result was edited. Every migrated source-to-target copy
has matching SHA-256 content. See `COMPLEMENTARY_PHASE_INVENTORY.csv` for the
file-level decision and `COMPLEMENTARY_DUPLICATE_GROUPS.csv` for byte-identical
groups found outside the virtual environment.

One migrated Phase C CSV is retained locally but excluded from public Git:
`p1c_pointwise_interior_residuals.csv` is 101,251,075 bytes, above GitHub's
100 MiB per-file limit. Its source hash and local target are preserved in the
inventory; the smaller Phase C summary and diagnostic outputs remain public.
