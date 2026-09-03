# Final supersonic migration report

## Git arrangement

- Target: `/Users/emma.grospellier/Thèse/These_PINN_KH_RT_supersonic`.
- Branch: `These_PINN_KH_RT_supersonic`.
- Mechanism: Git worktree, matching the existing subsonic organization.
- Source commit at initial migration: `94e63a95a5c2ba731e34df01f8f6674b6e659789`.
- Supplementary package integrated on 2026-08-26 from the source worktree without modifying it.
- The source and `These_PINN_KH_RT_subsonic` remain outside this reorganization.

## Inventory

- Classified source rows: 9,277.
- Canonical: 1,562.
- Experiment: 4,792.
- Legacy: 2,880.
- Ambiguous: 43 source rows, 34 unique destinations.
- Physical files in the organized target: 6,921, including root documentation.
- Current migration actions: 5,956 copies, 2,383 deduplications,
  325 Git moves, 28 review moves,
  6 logical-module copies, and 579 hard links.

## Supplementary package

- 45 source files mapped to 43 unique target files under `article/supplementary/`.
- Final figures and tables are separated from reusable data and source diagnostics.
- The two byte-identical duplicate S2 PDFs were deduplicated.
- Exact source-to-target paths and SHA256 hashes are in
  `provenance/SUPPLEMENTARY_MATERIAL_MANIFEST.csv`.
- The supplied convergence summary is retained with `audit_status = FAIL`.
- Exact S6 regeneration is not claimed because six raw reviewer V64 input tables
  referenced by the supplied plotting script were absent from the package.

## Checkpoints

- Source checkpoint paths: 609.
- Unique checkpoint hashes retained: 579.
- Duplicate source paths: 30.
- Logical size of unique migrated checkpoints: 1,889,540,749 bytes (about 1.8 GiB).
- All 579 source-to-target hard links passed inode verification.

## Legacy families

Historical material is retained for GEP, modal reconstruction, direct PINN pilots,
classical convergence, Blumen validation, generic shooting, chart overlap,
four-anchor/fixed-Mach pilots, training-stage comparisons, and early atlas/cost runs.
The production hybrid method remains PINN-seeded Riccati shooting.

## Validation

- Python compilation: 341/341.
- Bash/Slurm syntax: 132/132.
- Active launcher Python targets: 151/151.
- Static internal imports: 267 checked, 0 unresolved in active code.
- Manifest destinations: 9,277/9,277 present.
- Checkpoint hard links: 579/579.
- Naming: 0 active figure, CSV, or `frozen`-filename violations.
- Missing complete static checkpoint references: 0.
- Historical/importable Python names pending role-specific review: 93; see
  `SCRIPT_NAMING_REVIEW.csv`.

All supplementary manifest hashes and PDFs were checked separately. No heavy
numerical run was launched. Runtime smoke tests were not possible in the current
interpreter because the scientific dependencies listed in `requirements.txt`
are not installed.

## Remaining review points

- `run_classical_convergence_sweep.py` is a zero-byte source placeholder; dependent
  scripts are isolated rather than repaired by guesswork.
- Five preserved metadata/provenance path occurrences point to historical Jean Zay
  locations; two belong to the supplementary source metadata and remain unchanged.
- 81 static data-path literals require runtime or manual interpretation; many are
  outputs, path fragments, or historical layouts. They are enumerated in
  `PATH_REFERENCE_AUDIT.csv`.
- The 34 ambiguous destinations are listed in `AMBIGUOUS_FILES.md`.

## Authoritative maps

- Classification: `provenance/FILE_CLASSIFICATION.csv`.
- Migration: `provenance/MIGRATION_MANIFEST.csv`.
- Renaming: `provenance/RENAMING_MAP.csv`.
- Supplementary integration: `provenance/SUPPLEMENTARY_MATERIAL_MANIFEST.csv`.
- Validation: `provenance/VALIDATION_REPORT.md`.

## Finalization audit (2026-08-27)

The final dependency and full-hash audits pass. Active Python is 328/328,
internal imports are 267/267, shell syntax and launcher targets are 132/132 and
151/151, manifest destinations are 6,889/6,889 hash-valid, checkpoint hardlinks
are 579/579, and supplementary destinations are 43/43 hash-valid. Active
invalid runtime paths and cross-worktree runtime dependencies are both zero.

Two reproducible complete modal CSV exports are retained locally and excluded
using exact `.gitignore` rules because each exceeds 100 MiB. Their checksums,
producers, equivalents, and rationale are recorded in
`provenance/LARGE_FILE_AUDIT.csv` and `provenance/LARGE_FILE_POLICY.md`. Heavy
numerical reproduction was not run.
