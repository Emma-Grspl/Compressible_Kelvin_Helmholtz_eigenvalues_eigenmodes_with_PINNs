# Subsonic migration audit

## Scope and safety status

- Original worktree audited: `/Users/emma.grospellier/Thèse/These_PINN_KH_RT`.
- Target branch: `These_PINN_KH_RT_subsonic`.
- Target worktree: `/Users/emma.grospellier/Thèse/These_PINN_KH_RT_subsonic`.
- This phase creates documentation only. No source, checkpoint, asset or result was moved, copied, renamed or altered.
- The original worktree is dirty and remains authoritative for uncommitted/untracked research outputs.
- The clean target worktree starts from commit `94e63a95`; uncommitted original changes are intentionally absent.

## Inventory summary

- Unique source files detected: **12357**.
- Manifest rows: **12455**; the difference is caused by **98** intentional article-PNG copy rows.
- Python files mapped: **348**.
- Checkpoints detected: **1021**.
- Article/release rows marked: **401**.
- Shared dependencies marked `KEEP_SHARED`: **5**.
- Ambiguous `REVIEW` rows: **10**.
- Proposed active destination collisions still requiring resolution: **51**.

### By action

| Action | Count |
|---|---:|
| ARCHIVE | 2533 |
| COPY | 8956 |
| EXCLUDE | 206 |
| KEEP_SHARED | 5 |
| MOVE | 745 |
| REVIEW | 10 |

### By category

| Category | Count |
|---|---:|
| article_asset | 285 |
| checkpoint | 1021 |
| classic_subsonic | 737 |
| config | 106 |
| documentation | 63 |
| evaluation | 50 |
| gep | 8 |
| launcher | 116 |
| pinn_subsonic | 9857 |
| plotting | 119 |
| shared_dependency | 16 |
| training | 77 |

### Unique files by Git source status

| Status | Count |
|---|---:|
| ignored | 857 |
| tracked | 1382 |
| untracked | 10118 |

## Critical findings

- `pinn_subsonic/` contains a prior curated subsonic organization, 49-chart atlas material, wrappers, checkpoints and provenance tables, but it is **untracked** in the original worktree. It is evidence and a copy source, not something that can be `git mv`-ed.
- Several tracked `assets/pinn_subsonic/` files are deleted locally while newer untracked canonical/article trees exist. The migration must preserve HEAD files until supersession is explicitly validated.
- Tracked classical subsonic Python files and translated classical figures have local modifications. Those rows are `REVIEW`; silently taking either HEAD or local bytes would be unsafe.
- Some subsonic convergence runs live under `classic_supersonic/reproducibility/`. Only files whose path explicitly identifies a subsonic run are retained, with `COPY` and a cross-location note.
- Purely supersonic paths were removed from the manifest. Generic dense-GEP modules remain only when statically referenced by subsonic scripts.
- No complete LaTeX manuscript with `\includegraphics` was found. Article usage is inferred from `Table_asset_provenance.csv`, N340 manifests, release directories and article build scripts.
- The existing `Table_script_dependencies.csv` includes virtual-environment entries; only subsonic and repository-local records were used.

## Checkpoints

- Detected **1021** subsonic checkpoint rows across active, curated, smoke and legacy trees.
- Article/reproduction priority is assigned to N340 atlas, fixed-Mach, Stage 0, anchors-only and seed-sensitivity checkpoints when supported by path or provenance.
- Smoke and legacy checkpoints are proposed for `ARCHIVE`, not deletion.
- Checkpoint files must remain byte-identical; later migration requires size and SHA-256 verification.

## Article assets

- Canonical sources remain under `assets/classic_subsonic/` or `assets/pinn_subsonic/`.
- PNGs identified by curated provenance, N340 manifests and article directories receive a second explicit manifest row under `assets/article/`.
- These copies are intentional. They do not replace canonical assets and preserve subdirectories to avoid basename collisions.

## Shared dependencies

- Core model, residual, sampling and trainer modules currently live in `src/` and are imported directly by active entry points.
- Dense GEP support under `classical_solver/gep/dense_gep_notebook_style.py` is shared with other regimes and remains `KEEP_SHARED` pending a standalone vendoring decision.
- Wrappers and imports must only be updated after these modules are copied or exposed through a stable package boundary.

## Main risks

1. Uncommitted tracked changes in the original are not inherited by the new worktree.
2. Untracked curated files can be lost if moved instead of copied.
3. Proposed destinations still have collision groups; these must be resolved before any move.
4. Dynamic `Path(...)`, `runpy`, shell and Slurm references escape complete static analysis.
5. Historical checkpoints may embed absolute paths; checkpoint metadata must not be rewritten.
6. The existing curated tree contains compatibility duplicates; checksums/provenance must determine canonical versus archive copies.

## Proposed target tree

```text
README.md
PROJECT_STRUCTURE.md
requirements.txt
assets/article/
assets/classic_subsonic/{csv,png,pdf}/
assets/pinn_subsonic/{csv,png,pdf}/
code/src/{models,physics,data,configs}/
code/src/scripts/{classical,training,gep,evaluation,utils}/
code/src/launch/slurm/
code/plots/scripts/{classic_subsonic,pinn_subsonic}/
models_saved/{fixed_mach,atlas/N340,anchors_only/N340,seed_sensitivity/N340,archive}/
archive/  # justified extra directory for non-checkpoint historical material
docs/
tests/
```

## Migration plan requiring approval

1. Resolve `REVIEW`, beginning with modified tracked classical files and multi-purpose scripts.
2. Resolve every destination collision using experiment/chart/seed-preserving paths.
3. Create the target tree in the new worktree only.
4. Use `git mv` only for tracked, unmodified files present in the new worktree.
5. Copy untracked/gitignored files from the original; verify size and SHA-256; never move them.
6. Preserve canonical article assets and create only intentional PNG copies in `assets/article/`.
7. Rename figures, CSVs and scripts after collision checks, then repair imports/paths without changing scientific logic.
8. Run only compile/import/CLI/path checks and produce final experiment documentation.

## Files requiring review

- `assets/classic_subsonic/modes/03_subsonic_three_modes_reim_ampphase_m050.png`: tracked file modified in original; human decision required before migration
- `assets/classic_subsonic/plots/01_subsonic_isolines_blumen_vs_classique.png`: tracked file modified in original; human decision required before migration
- `assets/classic_subsonic/plots/02_subsonic_error_map_classique.png`: tracked file modified in original; human decision required before migration
- `assets/classic_subsonic/plots/subsonic_hybrid_vs_blumen.png`: tracked file modified in original; human decision required before migration
- `classical_solver/subsonic/hybrid_subsonic_scan.py`: tracked file modified in original; human decision required before migration
- `classical_solver/subsonic/mstab17_subsonic_solver.py`: tracked file modified in original; human decision required before migration
- `classical_solver/subsonic/plot_subsonic_error_map.py`: tracked file modified in original; human decision required before migration
- `classical_solver/subsonic/reconstruct_blumen_subsonic_shooting.py`: tracked file modified in original; human decision required before migration
- `protocole_pinn_subsonic_alpha_mach_sweep.md`: tracked file modified in original; human decision required before migration
- `scripts/build_presentation_plots.py`: tracked file modified in original; human decision required before migration

## Destination collision examples

- `code/plots/scripts/pinn_subsonic/make_longwave_mapping_audit.py` receives 4 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `assets/pinn_subsonic/csv/README.md` receives 3 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `assets/pinn_subsonic/csv/atlas_catalog_49_charts.tsv` receives 3 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `assets/pinn_subsonic/csv/stage0_report.txt` receives 3 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/make_blumen_isoline_assets.py` receives 3 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/make_mode_pdfs.py` receives 3 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/make_numerical_audits.py` receives 3 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/make_representative_mode_figure.py` receives 3 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/src/scripts/utils/__init__.py` receives 3 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `assets/article/misc/Fig_03_sparse_spectral_supervision.png` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `assets/article/misc/Fig_atlas_overlap_consistency.png` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `assets/article/misc/Fig_mode_error_vs_alpha.png` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `assets/article/misc/Fig_sparse_ci_anchor_map_Mach_alpha.png` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `assets/article/misc/Fig_supp_modal_overlap_defect.png` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `assets/pinn_subsonic/png/ci/Fig_ci_curve_hybrid8_vs_physics_only.png` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `assets/pinn_subsonic/png/ci/Fig_ci_supervision_needed_barplot.png` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `assets/pinn_subsonic/png/modes/Fig_mode_error_vs_alpha.png` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/build_ci_and_atlas_assets.py` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/check_assets_v2.py` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/inventory_missing_inputs.py` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/make_blumen_classical_pinn_overlay.py` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/make_direct_modal_heatmaps_20.py` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/plot_blumen_isolines_2x2.py` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/plot_blumen_pinn_gep_2x2.py` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/plot_ci_isolines_local_fix.py` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/plot_exact_mode_M050_alpha0500_local.py` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/plot_kh_subsonic_ci_error_heatmap.py` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/plot_kh_subsonic_ci_supervision_budget_summary.py` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/plot_kh_subsonic_ci_supervision_vs_physics.py` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- `code/plots/scripts/pinn_subsonic/plot_kh_subsonic_ci_supervision_vs_physics_article.py` receives 2 proposed sources; preserve experiment/chart/seed subdirectories before migration.
- … 21 additional active collision groups are identified in the manifest.
## Explicitly excluded from this phase

- No training, GEP campaign or classical sweep was run.
- No checkpoint, CSV numerical value, image or PDF was modified.
- No source migration, renaming, import repair, requirements generation or final README restructuring was performed.
- No commit or push was made.
