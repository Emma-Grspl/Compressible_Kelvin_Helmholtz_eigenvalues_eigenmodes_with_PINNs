# Subsonic pre-migration control report

No file migration, rename, scientific-file replacement, commit, or push was performed in this phase.

## Review resolution

| Path | Difference | Recommendation | Manual decision |
|---|---|---|---|
| `assets/classic_subsonic/modes/03_subsonic_three_modes_reim_ampphase_m050.png` | binary_generated_figure | **KEEP_LOCAL** | true |
| `assets/classic_subsonic/plots/01_subsonic_isolines_blumen_vs_classique.png` | binary_generated_figure | **KEEP_LOCAL** | true |
| `assets/classic_subsonic/plots/02_subsonic_error_map_classique.png` | binary_generated_figure | **KEEP_LOCAL** | true |
| `assets/classic_subsonic/plots/subsonic_hybrid_vs_blumen.png` | binary_generated_figure | **KEEP_LOCAL** | true |
| `classical_solver/subsonic/hybrid_subsonic_scan.py` | plot_labels_translation_only | **KEEP_LOCAL** | true |
| `classical_solver/subsonic/mstab17_subsonic_solver.py` | runtime_dependency_import_refactor | **KEEP_LOCAL** | true |
| `classical_solver/subsonic/plot_subsonic_error_map.py` | plot_labels_translation_only | **KEEP_LOCAL** | true |
| `classical_solver/subsonic/reconstruct_blumen_subsonic_shooting.py` | plot_uncertainty_and_translation | **KEEP_LOCAL** | true |
| `protocole_pinn_subsonic_alpha_mach_sweep.md` | documentation_extension | **KEEP_LOCAL** | true |
| `scripts/build_presentation_plots.py` | plot_title_translation_only | **KEEP_LOCAL** | true |

All ten local versions are recommended because the differences are publication-language updates, headless-runtime compatibility, audited uncertainty visualization, or current protocol documentation. No local file was substituted during this audit.

## Destination collisions

The collision table contains **51 groups** and **112 source rows**. Twelve byte-identical groups retain one canonical source and document redundant copies; the 39 groups with different bytes preserve their scientific hierarchy. After refining the manifest, there are **0 active destination collisions**. The additional generic checkpoint-name collision detected during final validation was removed by preserving each full experiment path under `models_saved/supporting/subsonic_general/`.

## Scope by unique source file

| Layer | Files | Estimated size |
|---|---:|---:|
| CORE | 318 | 0.217 GiB (233,117,621 bytes) |
| SUPPORTING | 1,889 | 0.639 GiB (685,640,244 bytes) |
| ARCHIVE | 9,786 | 6.577 GiB (7,062,198,771 bytes) |
| EXCLUDE | 365 | 0.409 GiB (439,660,236 bytes) |

Total unique source files: **12,358**. The manifest has 12,456 rows because 98 candidate publication-copy mappings are retained as explicit, non-active audit rows.

## Checkpoints

- Detected locally: **1,021**.
- Unique SHA-256 hashes: **872**.
- Used by the current article result set: **49**.
- Production: **50**; supporting: **464**; historical archive: **358**; byte-identical duplicate rows: **149**.

The N340 article builders expect 49 checkpoints below `assets/pinn_subsonic/anchor_budget_runs/N340/joint`, but that directory is absent locally. The 49 curated `model_state.pt` files under `assets/pinn_subsonic/checkpoints/joint_atlas_v2` are the present local article set; no missing Jean-Zay path was fabricated. The additional production checkpoint is the frozen fixed-Mach reference and is not marked as directly used by the manuscript.

## Article figures

- Current Main Text figures: **1**.
- Current Supplementary figures: **0**.
- Current manuscript source: `article.md`. It references only `assets/article/pinn_spectral_workflow.svg`.
- No active LaTeX source with `\includegraphics` was found. The 98 PNGs remain candidates or release assets and are not proposed for automatic copying into `assets/article/`.

## Supersonic exclusion

The content scan records 1,336 trigger occurrences. The only exclusively supersonic file identified is `scripts/plot_blumen_ci_overlay_all_points.py`; the refined manifest classifies it as **EXCLUDE**. Exclusively supersonic files still active after refinement: **0**.

## Audit status

This phase changes documentation only. No `git mv`, bulk copy, rename, commit, push, solver execution, or scientific refactor was performed.
