# Supersonic migration validation

Validation date: 2026-08-26. These are lightweight static checks only; no
training, shooting campaign, T401, COST500, or GPU job was launched.

## Summary

- Target branch: `These_PINN_KH_RT_supersonic`.
- Source branch/commit: `feature/supersonic-neutral-M180-M190-gep-shooting-clean` at `94e63a95a5c2ba731e34df01f8f6674b6e659789`.
- Python compilation: 341/341 passed.
- Shell/Slurm syntax: 132/132 passed.
- Active launcher Python targets: 151/151 resolved.
- Static internal imports checked: 267; unresolved: 0.
- Manifest destinations present: 9277/9277.
- Source-to-target checkpoint hard links verified: 579/579.
- Stale absolute source/Lustre prefixes in active text: 5.
- Explicit article references checked: 0; missing: 0.
- Active figure names violating `Fig_`: 0.
- Active CSV names violating `table_`: 0.
- Active filenames containing `frozen`: 0.

## Static path audit

`PATH_REFERENCE_AUDIT.csv` records 1277 path-like Python string
literals. Status counts: {'ABSOLUTE_MISSING': 1, 'BASENAME_FRAGMENT': 1091, 'DYNAMIC_OR_GLOB': 21, 'EXISTS': 67, 'LIKELY_OUTPUT_OR_RUNTIME_PATH': 16, 'MISSING_STATIC_LITERAL_REVIEW': 81}.

Missing static checkpoint literals requiring review: 0.
Dynamic paths, globs, and likely output paths are not treated as broken inputs.
Path expressions assembled dynamically at runtime cannot be certified by this
static check and must be exercised in the original numerical environment.

## Known unresolved dependency

`run_classical_convergence_sweep` is imported by historical convergence tools,
but both source copies of `run_classical_convergence_sweep.py` are zero-byte
placeholders. The migration does not invent an implementation. This dependency
remains a documented legacy gap rather than silently changing the algorithm.

Unresolved static imports: [].

## Article scope

Only three standalone LaTeX table fragments were present. No authoritative
full manuscript was found, so the files in `article/figures/` and
`article/tables/` are preserved as candidates. The added S2/S4/S6/S7 package
is organized under `article/supplementary/`. There are no definite broken
`includegraphics` references in the available TeX fragments.

## Runtime limitations

The local interpreter is missing these imported dependencies: ['PIL', 'PyPDF2', 'matplotlib', 'numpy', 'pandas', 'pyarrow', 'pypdf', 'pytest', 'scipy', 'torch', 'yaml'].
Consequently CLI `--help`, package import, and numerical smoke tests were not
claimed as successful here. Install `requirements.txt` in a compatible
environment, export the documented `PYTHONPATH`, and run lightweight CLI checks
before submitting HPC jobs.

## Manifest status

Migration hash/status counts: {'CANONICAL_CONTENT_POST_MIGRATION': 125, 'POST_MIGRATION_PATH_REPAIR': 1008, 'SHA256_OK': 8144}.

Ambiguous files, including launchers whose Python target was absent from the
source and the supplementary plotting snapshot with incomplete raw inputs,
are isolated under `_TO_REVIEW/`; they are not counted as active components.

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
