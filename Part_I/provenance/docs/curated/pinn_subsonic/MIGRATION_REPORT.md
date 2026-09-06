# PINN subsonic migration report

## Summary

This migration is additive and non-destructive. No source file was moved,
deleted or overwritten. Existing uncommitted Git changes were preserved.

Final organized content:

- 89 Python files under `pinn_subsonic/src/`;
- 180 checkpoint/metadata files;
- 3,823 organized PNG/PDF files;
- 1,332 organized CSV files;
- 24 stable modal assets, 13 stable spectral assets and 16 stable atlas assets;
- 85 pre-existing figures in the explicit article bundle.

The final inventory contains 3,084 source-classification records:

- 1,956 `ASSET_MODES`;
- 64 `ASSET_CI`;
- 26 `ASSET_ATLAS`;
- 461 `DATA_MODES`;
- 166 `DATA_CI`;
- 106 `DATA_ATLAS`;
- 180 `CHECKPOINT_SUBSONIC`;
- 57 `PLOT_SUBSONIC`;
- 26 `TRAINING_SUBSONIC`;
- 2 `MODEL_SUBSONIC`;
- 6 package/configuration records;
- 34 ambiguous files left in place.

## Actions

Two idempotent migration passes were required because the first pass gave
directory-level `atlas` context too much priority over scientific filenames.
The second pass corrected the classification without deleting the first-pass
links.

Cumulative actions:

- copied files: 0;
- hard links created: 5,371;
- symbolic links created: 0;
- generated/copied Python/package files: 91;
- ambiguous files left in place: 34;
- destructive actions: 0.

The apparent target size is larger than the additional disk allocation because
the scientific assets and checkpoints are hard links.

## Conflicts

Four stable modal filenames existed with different hashes between
`comparaisons_modales/` and `article/modes/`. Neither variant was overwritten.
The article variants use descriptive `_from_modes` destinations:

- `Fig_07_final_modal_error_heatmaps_from_modes.pdf`;
- `Fig_07_final_modal_error_heatmaps_from_modes.png`;
- `Fig_supp_direct_pinn_modal_error_heatmaps_from_modes.pdf`;
- `Fig_supp_direct_pinn_modal_error_heatmaps_from_modes.png`.

The first-pass broad atlas links are byte-identical redundant hard links. They
are retained because deletion was explicitly forbidden.

## Shared dependencies

- `classical_solver/subsonic/`;
- historical scripts under `scripts/`, `scripts/dev/`, `scripts/assets_v2/`
  and `scripts/paper/subsonic_results/`;
- PyTorch, NumPy, pandas, SciPy and Matplotlib;
- source checkpoints and CSV assets at their historical locations.

## Ambiguous files

The exhaustive list is stored in
`datas/atlas/Table_file_inventory.csv` with category `AMBIGUOUS`. It includes:

- historical direct modal-supervision experiments;
- explicitly broken pressure/velocity-core experiments;
- obsolete first-order and edge-repair experiments;
- generic runtime/convergence assets whose scientific category cannot be
  inferred safely;
- pointwise tables without enough semantic context.

These files were not copied automatically.

## Article assets

No manuscript LaTeX source referencing PINN-subsonic assets was found. The
pre-existing `assets/pinn_subsonic/article/` bundle is therefore retained as
the declared article selection. No LaTeX file was modified.

## Missing inputs

Tracked legacy files currently shown as deleted under
`assets/pinn_subsonic/mach_fixed/` and related old directories do not exist in
the worktree. They were not restored or synthesized, in order to preserve the
existing Git state.

## Traceability

The complete source-to-destination mapping is:

`archive/csv/pinn_subsonic/datas/atlas/Table_file_inventory.csv`

Columns:

- `original_path`;
- `new_path`;
- `category`;
- `action`;
- `sha256`;
- `size_bytes`;
- `reason`;
- `status`.

Core copied modules are the only files whose hashes intentionally differ from
their sources, because imports were changed from `src.*` to
`pinn_subsonic.src.*`. No numerical expression, algorithm, hyperparameter or
scientific default was changed.

## Validation status

Validation was completed without running training or numerical solvers.

- Structure: all 14 required package and stable-asset directories exist.
- Names: zero non-`Fig_` PNG/PDF files under `pinn_subsonic/assets/`.
- Names: zero non-`Table_` CSV files under `pinn_subsonic/datas/`.
- Names: zero spaces in organized source, checkpoint, data or asset paths.
- Syntax: all 89 Python files under `pinn_subsonic/src/` compile.
- Imports: `pinn_subsonic`, `pinn_subsonic.src.models`,
  `pinn_subsonic.src.training` and `pinn_subsonic.src.plot` import
  successfully in an environment containing PyTorch and SciPy.
- CLI: all 67 wrappers whose sources use `argparse` return successfully for
  `--help`.
- Paths: zero `/lustre/...` or `/Users/...` paths remain in the organized
  Python tree.
- Shared paths: 13 wrappers intentionally reference sources under
  `scripts/dev/`; all 13 targets exist and are documented shared
  dependencies.
- Wrapper targets: all 80 generated compatibility wrappers resolve to an
  existing original script.
- Integrity: 2,959 source/destination pairs were checked with SHA-256; zero
  mismatches were found.
- Hard links: all 2,959 checked scientific data/asset destinations share an
  inode with their source.
- Manifest paths: zero missing sources and zero missing destinations.

The repository's default `/opt/anaconda3/bin/python3` currently resolves to a
zero-byte `python3.13` file. Validation therefore used the intact cached
Anaconda interpreter with `PYTHONHOME=/opt/anaconda3`; no environment file was
modified. The system Python was used independently for syntax compilation.

The article bundle contains 151 pre-existing files: 85 PNG/PDF figures, 30
CSV files and 36 metadata/log/text files. Since no manuscript LaTeX source was
found, membership in this bundle could not be cross-checked against
`\includegraphics` or table references.

The final `git status --short` remains intentionally dirty and reports
pre-existing modifications, deletions and unrelated untracked files across
the repository (`14 M`, `56 D`, `307 ??` at the final audit). No reset, stash,
checkout, deletion, move, staging, commit or push was performed by this
migration.

## Remaining limitations

- No operational PINN-subsonic YAML configuration was found. Current
  configurations remain encoded in Python defaults, CSV manifests and
  checkpoint metadata.
- The compatibility wrappers still depend on the original scripts. This
  avoids scientific duplication and preserves behavior, but the historical
  scripts must remain in place.
- Runtime input/output paths embedded in historical source scripts were not
  refactored. Known missing tracked legacy inputs are listed above.
- The redundant first-pass atlas hard links remain present because this task
  explicitly prohibited deletion.

## Legacy archive

On 2026-07-30, a second SHA-validated pass moved 1,873 historical files to
`archive/pinn_subsonic_legacy/`:

- 1,829 nonconforming historical figure names with byte-identical active
  `Fig_` equivalents;
- 34 additional files with byte-identical canonical equivalents;
- six self-contained bundles or bundle sidecars;
- four unresolved copies of the same historical training-convergence figure.

No file was overwritten or deleted. The archive manifest records identical
pre-move and post-move SHA-256 values for every entry.

## Asset provenance

The post-archive active set contains 3,999 PNG/PDF paths. Every active path
has exactly one row in `datas/atlas/Table_asset_provenance.csv`, and zero
active rows are `ORPHAN_UNRESOLVED`.

The provenance audit combines AST output detection, filename normalization,
dynamic naming rules, wrapper resolution, previous migration mappings, and
SHA-256 duplicate propagation. `ASSET_PROVENANCE.md` contains the aggregate
counts and generator inventory.

## Reproducible figures

Two canonical plot entry points were added without changing scientific
calculations:

- `src/plot/plot_spectral_modal_architecture.py`;
- `src/plot/plot_subsonic_runtime.py`.

The architecture PNG was reproduced pixel-for-pixel. The runtime plot was
reproduced from the retained benchmark table with matching dimensions and
data, but is explicitly classified `ORPHAN_REPRODUCIBLE` because typography
and layout are not pixel-identical.

## Unresolved historical assets

The four archived `Fig_supp_ci_anchor_training_convergence` copies remain
documented as `ORPHAN_UNRESOLVED` in the archive manifest. Repository and
bundle searches did not recover their generator, and inferring the exact
history inputs would require an unsupported choice.

## Compatibility wrappers

No compatibility symlink was required. Existing wrappers whose behavior still
depends on historical source paths continue to use those paths; the targets
were not archived. Four additional canonical wrappers expose retained
historical generators without duplicating their scientific logic.

## Shared dependencies

`SCRIPT_DEPENDENCIES.md` and
`datas/atlas/Table_script_dependencies.csv` are the authoritative static
dependency reports. Shared `classical_solver/subsonic/` modules and historical
scripts still executed by canonical wrappers remain outside the archive.

## Naming limitation

The final figure-name check passes with zero non-`Fig_` PNG/PDF files in the
active roots. The strict CSV-name check reports 739 operational raw outputs
without a `Table_` prefix. These names are part of retained generator
interfaces, so changing them would violate the no-pipeline-breakage rule.
Canonical organized tables in `pinn_subsonic/datas/` remain normalized as
`Table_*.csv`.
