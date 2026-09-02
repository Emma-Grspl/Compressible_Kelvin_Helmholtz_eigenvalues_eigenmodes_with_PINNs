# Final dependency audit

Audit date: 2026-08-27.
Authorized source-state SHA-256: `d9ab617796b21fd61d827747096b0dc7d8e9ab32abab4897af3fd46457f5c10a`.

## Structural checks

- Active Python: 257/257 compiled.
- Static internal imports: 195/195 resolved.
- Bash/Slurm: 121/121 syntax checks passed.
- Parsed launcher targets: 67/67 resolved.
- Config files: 127/127 structurally parsed.
- Active invalid runtime paths: 0.
- Absolute cross-worktree runtime dependencies: 0.
- Secrets and sensitive filenames: 0.
- Cache files proposed for commit: 0.

One old path remains in
`assets/pinn_subsonic/csv/curated/pinn_subsonic/data/audits/article/audit_summary.json`.
It is historical audit metadata, not an active runtime dependency; its mapped
archive destination exists.

## Integrity

- Manifest rows: 12,897.
- Unique hashed destinations: 11,994/11,994 present and hash-valid.
- Checkpoint inventory rows: 1,021.
- Unique checkpoint contents: 872/872 present and hash-valid.
- Unexplained destination hash mismatches: 0.

Fifteen migrated code files received path-only repairs. Their source hashes
remain unchanged; destination hashes and reasons are recorded in
`provenance/FINAL_DEPENDENCY_PATH_REPAIRS_2026-08-27.csv`.

## Scope

No training, solver campaign, shooting sweep, GPU job, or scientific benchmark
was run. Result: **PASS WITH DOCUMENTED EXCEPTIONS** (historical metadata path
and heavy numerical reproduction not run).
