# Part II repository reorganization manifest

## Original local structure

The source workspace contained `article/`, `assets/`, `code/`, `complementary_phase/`, `experiments/`, `models_saved/`, `plots/`, `provenance/`, `scripts/`, `slurm/`, `tests/`, and `_TO_REVIEW/`.

## Integration decisions

- The reusable source tree was copied into `Part_II/` without numerical recomputation.
- `complementary_phase/` was semantically migrated into `results/` and `scripts/reviewer_controls/`; see `complementary_phase_migration.md`.
- Curated publication figures are in `article/assets/`; their byte-level hashes and source mappings are in `article/FIGURE_MANIFEST.csv`.
- The old `plots/scripts/` directory remains preserved because it contains scientific plotting logic.
- Checkpoint binaries and files at or above 100 MiB are excluded from public Git; see `large_files_report.md`.
- Two `*.bak_before_final_layout_fix` article-script snapshots are retained in
  `_TO_REVIEW/scripts/article_backups/` rather than alongside active scripts.
- No source scientific content was deleted from the original workspace.

## Unresolved items

- `_TO_REVIEW/` remains intentionally intact.
- No authoritative complete LaTeX manuscript source was found; only table fragments were supplied under `article/tex/`.
- Three final supplementary PNGs are documented renderings of authoritative PDFs because no corresponding authoritative PNG was available.
