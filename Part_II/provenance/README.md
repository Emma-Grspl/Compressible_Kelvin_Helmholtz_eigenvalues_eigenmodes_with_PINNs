# Migration provenance

## Source

- Source repository: `/Users/emma.grospellier/Thèse/These_PINN_KH_RT`
- Source branch at initial migration: `feature/supersonic-neutral-M180-M190-gep-shooting-clean`
- Source commit at initial migration: `94e63a95a5c2ba731e34df01f8f6674b6e659789`
- Recorded source worktree-state hash before initial migration:
  `2cbb77fd9bf394112632c39247957b52680577bf9ab9e6e60b1707f76cd92254`
- Initial migration date: 2026-08-25
- Supplementary package integration date: 2026-08-26

The target was created as a Git worktree on branch
`These_PINN_KH_RT_supersonic`. The source repository and sibling
`These_PINN_KH_RT_subsonic` were not reorganized or cleaned by this migration.
The later supplementary integration copied the new untracked source package
`assets/p3-supersonic-results/supplementary_material/` without modifying it.

## Inventory

- 9,277 source rows classified.
- 1,562 canonical source rows.
- 4,792 experiment source rows.
- 2,880 legacy source rows.
- 43 ambiguous source rows mapping to 34 unique review destinations.
- 609 checkpoint paths, 579 unique checkpoint hashes.
- 30 duplicate checkpoint paths.
- 1,989,482,985 bytes across checkpoint paths before deduplication.
- 45 supplementary source files mapped to 43 unique target files; two duplicate PDFs were deduplicated.

Exact current counts are also machine-readable in `AUDIT_SUMMARY.json`.

## Files

- `FILE_CLASSIFICATION.csv`: one row per selected source file, including
  scientific family, experiment, role, status, source hash, and destination.
- `MIGRATION_MANIFEST.csv`: migration action and post-migration hash status.
- `RENAMING_MAP.csv`: complete old-path to new-path mapping.
- `SUPPLEMENTARY_MATERIAL_MANIFEST.csv`: exact supplementary source-to-target
  mapping, deduplications, sizes, and SHA256 hashes.
- `AUDIT_SUMMARY.json`: aggregate audit counts.
- `PATH_REPAIR_SUMMARY.json`: initial path-repair operation counts.
- `PATH_REFERENCE_AUDIT.csv`: static audit of path-like literals.
- `VALIDATION_REPORT.md`: final lightweight validation results and known gaps.
- `AMBIGUOUS_FILES.md`: deduplicated list of manual-review destinations.
- `SCRIPT_NAMING_REVIEW.csv`: historical/importable Python filenames not renamed automatically.

## Checkpoint policy

All locally available supersonic `.pt` and `.pth` files were inventoried by
SHA256 before migration. Unique immutable checkpoints were hard-linked from
the source where safe. Byte-identical duplicate paths point to the canonical
migrated content and were not copied unnecessarily. No checkpoint was
downloaded and no source checkpoint was deleted.

## Scientific policy

Numerical algorithms, loss weights, grids, tolerances, anchor sets, seeds,
and training schedules were not changed. Only organization, names, imports,
and filesystem paths were adjusted. The production hybrid workflow is
PINN-seeded Riccati shooting; GEP material is retained as legacy diagnostics.

The supplementary convergence summary is preserved with `audit_status = FAIL`.
The supplied S6 outputs are retained, but exact regeneration requires six raw
reviewer V64 tables that were absent from the source package. These limitations
are documented in `article/supplementary/README.md`.
