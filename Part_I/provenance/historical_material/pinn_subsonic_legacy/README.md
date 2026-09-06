# PINN Subsonic Legacy Archive

This archive preserves historical PINN-subsonic material moved out of the
active canonical roots during the 2026-07-30 provenance cleanup.

Active code and assets remain under:

- `pinn_subsonic/`
- `assets/pinn_subsonic/`

## Rules

- No file is deleted or overwritten.
- Every move is preflighted against the SHA-256 recorded in
  `pinn_subsonic/datas/atlas/Table_legacy_archive_plan.csv`.
- A canonical equivalent is hash-checked when one is declared.
- Files with unresolved active dependencies remain at their historical path.
- Historical names are retained inside the archive.

The completed move manifest is
`reports/Table_archive_manifest.csv`. It records original and archive paths,
hashes before and after movement, dependency state, provenance state, and
validation status.

The current manifest contains 1,873 validated moves. Four entries are
historical copies of `Fig_supp_ci_anchor_training_convergence` retained as
unresolved orphan assets; the remaining moves have a canonical byte-identical
equivalent or are self-contained bundles.

## Restoration

To restore one entry, locate it in the manifest, verify its archived SHA-256,
and move it back to `original_path` only if that destination does not exist.
Tracked files should be restored with `git mv`; untracked files can be renamed
on the same filesystem.

## Compatibility

No compatibility symlink was required by the current plan. Historical scripts
still executed by canonical wrappers are explicitly retained in place and are
listed as `KEEP_SHARED` in the plan.

## Not Archived

Shared modules, wrapper targets, ambiguous experiments, and candidates lacking
a validated canonical equivalent remain in place. Their reasons are recorded
in `Table_legacy_archive_plan.csv` and `SCRIPT_DEPENDENCIES.md`.
