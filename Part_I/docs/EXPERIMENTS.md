# Subsonic experiment inventory

The paths below are canonical migration destinations. Detailed input/output
tables and checkpoint aliases are available in the CSV inventories in this
directory.

| Experiment | Scientific purpose | Anchor budget | Main script | Checkpoints / seeds | Validation and outputs | Usage | Status |
|---|---|---:|---|---|---|---|---|
| Fixed-Mach physics-only | Physics-only branch-identification baseline at fixed Mach | A0 | `code/src/scripts/training/fixed_mach/physics_only/train_kh_subsonic_pinn.py` | Production fixed-Mach reference; see checkpoint inventory | Classical/PINN `c_i` and modal diagnostics | Baseline | Preserved |
| Fixed-Mach Stage 0 | Sparse scalar branch localization | A4 | `code/src/scripts/training/fixed_mach/physics_only/train_kh_subsonic_ci_stage0_anchor_lock.py` | Four scalar anchors; seed metadata in migrated configs | Anchor prediction and spectral tables | Main methodology | Preserved |
| Fixed-Mach Stage 1 modal | Physics-constrained modal reconstruction after Stage 0 | A4 | `code/src/scripts/utils/run_kh_subsonic_M05_alpha010_080_hybrid_stage1_mode_from_ci_stage0.sh` | Stage 1 checkpoints in production/supporting inventories | Pressure and primitive-field modal diagnostics | Main methodology | Preserved; historical shell name retained |
| Atlas N96 | Low sparse-eigenvalue budget comparison | N96 | `code/src/scripts/training/atlas/direct_pinn/train_atlas_chart_joint_ci_mode.py` | Per-chart/seed paths in checkpoint inventory | Anchor-budget tables and figures | Ablation | Preserved |
| Atlas N224 | Intermediate budget comparison | N224 | Same atlas trainer | Per-chart/seed inventory | Anchor-budget tables and figures | Ablation | Preserved |
| Atlas N340 | Production 49-chart atlas | N340 | Same atlas trainer | `models_saved/production/atlas/N340/<chart>/model_state.pt`; primary seed documented per chart | Spectral/modal atlas and scalar-guided GEP eigenpair selection | Main pipeline | Production |
| Atlas N520 | High-budget comparison | N520 | Same atlas trainer | Per-chart/seed inventory | Anchor-budget tables and figures | Ablation | Preserved |
| Atlas N640 | High-budget comparison | N640 | Same atlas trainer | Per-chart/seed inventory | Anchor-budget tables and figures | Ablation | Preserved |
| Atlas N705 | Highest audited budget comparison | N705 | Same atlas trainer | Per-chart/seed inventory | Anchor-budget tables and figures | Ablation | Preserved |
| Anchors-only N340 | Separate the scalar-anchor effect from physics coupling | N340 | Atlas anchors-only scripts under `code/src/scripts/training/atlas/anchors_only/` | `models_saved/supporting/anchors_only_N340/` | Holdout and chart-level comparisons | Supplementary/supporting | Preserved |
| Atlas seed sensitivity | Quantify chart-to-chart and seed variability | N340 | Scripts under `code/src/scripts/training/atlas/seed_sensitivity/` | `models_saved/supporting/seed_sensitivity/` | Seed summary tables/figures | Supplementary/supporting | Preserved |
| Canonical validation 372 | Evaluate the canonical validation set | N340 pipeline | Evaluation scripts under `code/src/scripts/evaluation/` | Production atlas checkpoints | 372-point tables | Validation | Preserved |
| Holdout 384 | Off-grid/generalization comparison | N340 | `code/src/scripts/evaluation/evaluate_subsonic_atlas_offgrid.py` | Production and anchors-only checkpoints | Holdout tables and figures | Supplementary/supporting | Preserved |
| Overlap 195 | Check chart-overlap consistency | N340 | `code/src/scripts/utils/build_N340_overlap_asset.py` | Production atlas | Overlap tables and figures | Supplementary/supporting | Preserved |
| Modal validation 20 | Compare reconstructed modes at representative points | N340 | Plot/evaluation scripts containing `mode_profiles_20` or `modal` | Production atlas | Pressure and primitive-field modal errors | Main/supporting | Preserved |
| GEP resolution | Resolve the eigenpair after neural scalar branch localization | N340 | `code/src/scripts/gep/selection/solve_dense_gep_notebook_style.py` and related GEP scripts | Production atlas scalar outputs | Selected GEP eigenvalue and eigenvector tables | Main pipeline | Preserved |
| Dense audit 1000 | Dense spectral audit of the parametric result | N340 | Evaluation scripts under `code/src/scripts/evaluation/` | Production atlas | Dense pointwise metrics | Supporting | Preserved |
| Near-neutral | Test continuation and GEP resolution where `c_i` is small | N340 | `code/src/scripts/evaluation/audit_subsonic_near_neutral_gep_resolution.py` | Production/supporting checkpoints | Near-neutral cuts and audits | Supplementary/supporting | Preserved |
| Runtime benchmark | Compare computational costs without changing physics | N340 | `code/src/scripts/utils/utils_benchmark_subsonic_runtime.py` | Production atlas | Runtime tables and figures | Supplementary/supporting | Preserved |

## Interpretation constraints

Sparse scalar eigenvalues are used for branch localization. Physics constrains
the spectral-modal representation. The GEP is diagonalized independently, and
the scalar neural estimate is used only afterward to select an eigenpair. No
experiment in this inventory should be described as showing that direct neural
prediction replaces the GEP or that physics universally improves `c_i`
interpolation.
