# Final dependency audit

Audit date: 2026-08-27.
Authorized source-state SHA-256: `d9ab617796b21fd61d827747096b0dc7d8e9ab32abab4897af3fd46457f5c10a`.

## Structural checks

- Active Python: 328/328 compiled.
- Static internal imports: 267/267 resolved.
- Bash/Slurm: 132/132 syntax checks passed.
- Active launcher Python targets: 151/151 resolved.
- Config files: 1,242/1,242 structurally parsed.
- Active invalid runtime paths: 0.
- Absolute cross-worktree runtime dependencies: 0.
- Secrets and sensitive filenames: 0.
- Cache files proposed for commit: 0.

The generic shell parser reports 15 non-target strings. They are false
positives from Python heredocs, an optional local virtual-environment activation,
and `python -m py_compile`; the launcher-specific audit resolves 151/151 real
Python targets.

## Integrity

- Manifest rows: 9,277.
- Unique hashed destinations: 6,889/6,889 present and hash-valid.
- Checkpoint hardlink rows: 579/579 present and inode-consistent locally.
- Supplementary rows: 45; unique destinations: 43/43 hash-valid.
- Unexplained destination hash mismatches: 0.

Three migrated code files received path-only repairs and one Makefile received
terminal-whitespace normalization after preserving its pre-existing
`article-assets` target. Source hashes remain unchanged; destination hashes and
reasons are recorded in
`provenance/FINAL_DEPENDENCY_PATH_REPAIRS_2026-08-27.csv`.

## Scope

No training, solver campaign, shooting sweep, GPU job, or scientific benchmark
was run. Result: **PASS WITH DOCUMENTED EXCEPTIONS** (heavy numerical
reproduction not run).
