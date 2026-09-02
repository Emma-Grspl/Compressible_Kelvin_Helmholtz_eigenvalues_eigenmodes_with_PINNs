# Subsonic migration final report

Date: 2026-08-21

## Decision and provenance

- Source repository: `/Users/emma.grospellier/Thèse/These_PINN_KH_RT`.
- Target repository: `/Users/emma.grospellier/Thèse/These_PINN_KH_RT_subsonic`.
- Authorized source-state SHA-256: `d9ab617796b21fd61d827747096b0dc7d8e9ab32abab4897af3fd46457f5c10a`.
- The source-state hash was recomputed from `git status --porcelain=v1 -z` and matched the authorized value before and after reconciliation.
- The original repository was not restored, edited, committed, or pushed.
- The existing migration was resumed; it was not rebuilt from scratch.

The prior migration manifest was used as the comparison baseline. Only source
paths already classified inside the subsonic scope were eligible for
synchronization. Recent supersonic work remained outside the target.

## Source-state reconciliation

The comparison covered 11,997 unique in-scope source paths recorded by the
migration manifest:

| Result | Count | Action |
|---|---:|---|
| Byte-identical to the migration baseline | 11,836 | Preserved unchanged |
| Changed in the authorized source state | 108 | Synchronized to existing destinations and SHA-256 verified |
| Missing from the authorized source state | 53 | Existing migrated destination preserved |
| Ambiguous baseline | 0 | None |

The 108 synchronized files belong to existing scientific-output paths under
`pinn_subsonic/data/scientific_outputs/`: 100 are `ARCHIVE` and 8 are
`SUPPORTING`. They do not add new scientific code or launchers. All 108 source
and destination hashes match after synchronization. The detailed ledger is
`docs/Table_source_resync_2026-08-21.csv`.

The 53 source-missing paths were intentionally not deleted from the target:
their migrated copies preserve scientific provenance and checkpoints. Their
manifest status is `DESTINATION_PRESERVED_SOURCE_MISSING`.

There were 145 Git-status paths containing subsonic markers that were not
baseline source rows. They were reviewed as pre-existing aliases, duplicate
checkpoint paths, deleted source aliases, historical material, or paths
outside the defined migration scope. No additional file from that set was
introduced into the canonical migration.

## Final inventory

The final manifest contains 12,897 rows:

| Layer | Rows |
|---|---:|
| `CORE` | 318 |
| `SUPPORTING` | 1,891 |
| `ARCHIVE` | 9,788 |
| `EXCLUDE` | 900 |

There are 2,208 unique canonical `CORE` or `SUPPORTING` destinations,
including:

- 234 Python files;
- 109 Slurm launchers;
- 8 YAML configuration files;
- 539 CSV tables;
- 431 PNG, 123 PDF, and 1 SVG figure files;
- 514 canonical checkpoint paths in `CORE` or `SUPPORTING`.

The complete checkpoint inventory contains 1,021 historical paths mapped to
872 unique physical checkpoint files. The 149 duplicate paths are represented
by canonical-content mappings rather than duplicate physical checkpoint
copies.

The migration operation ledger records:

| Operation | Count |
|---|---:|
| `COPY` | 9,326 |
| `COPY_ALREADY_PRESENT` | 1,300 |
| `GIT_MV` | 803 |
| `KEEP_IN_PLACE` | 381 |
| `EXCLUDED` | 216 |
| `GIT_MV_ALREADY_APPLIED` | 170 |
| `DUPLICATE_NOT_COPIED` | 131 |
| `GIT_RM_EXCLUDED` | 18 |
| `GIT_MV_ALREADY_APPLIED+KEEP_LOCAL` | 8 |
| `GIT_MV+KEEP_LOCAL` | 2 |
| `EXTERNAL_LINK_UNRESOLVED` | 2 |
| `DEDUPLICATED` | 1 |

The final manifest also contains post-audit classifications and exclusions
that are not individual migration operations, notably 431 additional
supersonic exclusions, 463 deferred article-copy candidates, and 6 canonical
package markers.

## Path and code repairs

Migration-only path repairs affected 381 files according to
`docs/Table_code_path_repairs.csv`, for 5,665 recorded replacements. These
repairs adapt repository paths and imports to the target layout; they do not
change equations, losses, numerical parameters, or scientific results.

The final strict launcher audit found three historical Python targets that
still referenced their old locations. Only those target paths were repaired:

- `code/src/launch/slurm/01_offgrid_384_array.slurm` now calls
  `code/src/scripts/evaluation/evaluate_subsonic_atlas_offgrid.py`;
- `code/src/launch/slurm/02_retry_failed_73.slurm` uses the same canonical
  evaluator;
- `code/src/scripts/utils/03_validate_N340_seams.slurm` now calls
  `code/src/scripts/evaluation/evaluate_joint_pinn_global_validation.py`.

## Integrity validation

All lightweight final checks passed:

| Check | Result |
|---|---|
| Source-state SHA-256 | Exact match with `d9ab617796b21fd61d827747096b0dc7d8e9ab32abab4897af3fd46457f5c10a` |
| Resynchronized files | 108/108 destination hashes match the authorized source |
| Python syntax | 257/257 files pass `py_compile` |
| Static internal imports | 185 imports checked, 0 unresolved |
| Slurm syntax | 109/109 files pass `bash -n` |
| Slurm Python targets | 85 invocations detected, 0 missing target |
| YAML syntax | 9/9 repository YAML files parse successfully |
| Resynchronized CSV files | 18/18 parse successfully |
| Resynchronized JSON files | 27/27 parse successfully |
| Checkpoint presence | 872/872 unique physical files present |
| Checkpoint SHA-256 | 872/872 unique physical hashes match |
| Unexpected missing manifest destinations | 0 |
| Residual `REVIEW` rows | 0 |
| Absolute local paths in `CORE` text files | 0 |
| `frozen` in canonical filenames | 0 |

No training, solver campaign, GPU computation, or expensive scientific
recalculation was launched. The checks validate migration integrity and code
structure, not numerical reproducibility on Jean-Zay.

## Article assets

`docs/Table_article_figures.csv` contains 99 candidate article assets. Every
canonical candidate exists, but all remain marked `CANDIDATE` because an
authoritative manuscript was not available to determine the definitive Main
and Supplementary selection.

Three non-archived LaTeX table sources are preserved:

- `code/src/scripts/utils/SuppTable_multiseed_robustness.tex`;
- `code/src/scripts/utils/Table_PINN_hyperparameters.tex`;
- `code/src/scripts/utils/Table_global_validation_metrics.tex`.

## Remaining manual or external requirements

1. Two legacy article-data links cannot be resolved locally:
   `assets/pinn_subsonic/csv/hybrid_4pt` and
   `assets/pinn_subsonic/csv/physics_only`. They remain explicitly marked
   `EXTERNAL_LINK_UNRESOLVED`; no local content was fabricated.
2. The authoritative manuscript is required before copying the 99 candidate
   assets into a final article inventory.
3. The local validation environment does not provide `numpy`, `pandas`,
   `scipy`, `matplotlib`, `torch`, or `Pillow`. Runtime imports, plotting, and
   solver smoke tests therefore require the documented scientific or
   Jean-Zay environment.
4. Exact end-to-end reproduction still requires running the documented
   classical, PINN, GEP, and post-processing commands in that environment.

## Git state

The target remains on branch `These_PINN_KH_RT_subsonic`. The migration is
intentionally uncommitted and unpushed. At final inspection, the target worktree
contains the expected large migration diff and untracked migrated files. No
history operation was performed.

## Conclusion

The subsonic migration is structurally complete relative to the authorized
source state. All newly changed in-scope files were synchronized, preserved
checkpoints remain byte-identical, code and launcher integrity checks pass,
and no recent supersonic resource was imported into the canonical subsonic
scope. The remaining limitations are external data links, unavailable runtime
dependencies, and manuscript-level asset selection.

## Finalization audit (2026-08-27)

The final dependency and full-hash audits pass. Active Python is 257/257,
internal imports are 195/195, shell syntax and targets are 121/121 and 67/67,
manifest destinations are 11,994/11,994 hash-valid, and checkpoints are
872/872 hash-valid. Active invalid runtime paths and cross-worktree runtime
dependencies are both zero. See `provenance/DEPENDENCY_AUDIT.md` and the dated
repair ledger. Heavy numerical reproduction was not run.
