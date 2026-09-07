# Part II final-cleanup move plan

## Proven directory contracts

- **Main 1**: `UNRESOLVED: plot script requires an explicit --reference file; identify the release reference CSV before replacing.`
- **Main 2**: `assets/article/classical_supersonic/source_data/supersonic_blumen_positive_pointwise_values.csv (declared directory is not read by the script).`
- **Main 3**: `results/blumen_cr_validation/results/phase11_final_stratified_cr_validation.csv (declared assets directory is output, not input).`
- **Main 6**: `UNRESOLVED: generator reads assets/pinn_supersonic/... rather than experiments/atlas_12charts.`
- **Main 7**: `UNRESOLVED: generator reads assets/pinn_supersonic/... rather than experiments/anchor_budget.`
- **Main 8**: `UNRESOLVED: generator reads assets/pinn_supersonic/... rather than experiments/anchor_budget.`
- **Main 9**: `UNRESOLVED: generator reads assets/pinn_supersonic/... rather than experiments/pinn_direct.`
- **Main 10**: `results/matched_branch_localization/budget_matched/budget_matched_T401/analysis/phase7_paired_T401.csv;results/matched_branch_localization/budget_matched/budget_matched_T401/analysis/phase7_summary.json`
- **Main 11**: `results/failure_basin/analysis/phase8_basin_all_points.csv;results/failure_basin/analysis/phase8_nearest_pinn_seed.csv`
- **Main 12**: `assets/pinn_supersonic/png/modal_reconstruction/p3-supersonic-results (NO_SCRIPT_FOUND; retain directory pending a generator).`
- **Main 13**: `assets/pinn_supersonic/png/modal_reconstruction/p3-supersonic-results (NO_SCRIPT_FOUND; retain directory pending a generator).`
- **Main 14**: `UNRESOLVED: generator reads assets/pinn_supersonic/... rather than experiments/computational_cost.`
- **Supplementary S5**: `results/physics_residual_audit/independent/physics_residual_coordinate_summary.csv;results/physics_residual_audit/stage_comparison/stage2_vs_stage3_paired_coordinates.csv`
- **Supplementary S6**: `Six N76 per-seed metric JSON files plus the selected V64 shooting_validation_64.csv for each run; make the selected candidate explicit after confirming which candidate exists.`
- **Supplementary S7a**: `results/routing_audit/results/routing_boundary_audit/routing_boundary_results.csv;results/routing_audit/results/routing_boundary_audit/routing_boundaries.csv;results/routing_audit/results/routing_boundary_audit/routing_summary_by_eps.csv;results/routing_audit/results/routing_boundary_audit/routing_summary.json`
- **Supplementary S8**: `results/threshold_sensitivity/threshold_scale_sensitivity.csv;results/threshold_sensitivity/threshold_2D_sensitivity.csv;results/threshold_sensitivity/per_query_threshold_margin.csv`

## Explicitly protected legacy publication tables

- `experiments/classical_convergence/legacy_outputs/convergence_tables/table_matching_location_summary.csv`: proposed later destination `article/tables/` or `assets/article_sources/` after checksum-preserving manifest update.
- `experiments/modal_reconstruction/legacy_outputs/convergence_tables/table_modal_convergence_errors.csv`: proposed later destination `article/tables/` or `assets/article_sources/` after checksum-preserving manifest update.

## Proposed interface convergence (not performed)

- Move the two direct public plot paths to `code/plots/generators/`, then add publication wrappers under `code/plots/article/`.
- Move canonical `configs/` to `code/configs/` only after updating consumers listed by a path audit.
- Keep supported portable launchers under `code/slurm/`; classify all Jean-Zay-only and GEP launchers as experiment/provenance rather than public.
- Move legacy GEP implementation and launchers to `provenance/legacy_gep/` only after manifest and import closure proves no final consumer.

## Canonical-config consumers to update before a future `configs/` move

- `production/classical/dense_supersonic_campaign_config.json`: active preparation/evaluation scripts `prepare_build_classical_supersonic_assets.py`, `prepare_lowM_full_branches.py`, `freeze_dense_supersonic_results.py`, `run_blumen_true_positive_isoline_campaign.py`, `run_dense_supersonic_convergence_audit.py`, plus the two `code/slurm/pinn_direct/` launchers.
- `validation/pointwise/cases.yaml`: `code/scripts/evaluation/test_pointwise_convergence_campaign.py`.
- `validation/plotting_ci.yaml` and `validation/reference_v2.yaml`: `code/scripts/data_preparation/prepare_build_release_manifest.py`; `plotting_ci.yaml` is also used by `test_showcase_assets.py`.
- `validation/witness_M150_a01625.yaml`: the witness audit/calibration/probe/evaluation scripts under `code/scripts/`.
- `experiments/fixed_mach/S4M4.json`: `code/scripts/evaluation/smoke_test_local_supersonic_kappa_q_logamp.py`.

## HPC assessment

`scripts/hpc/HPC_MANIFEST.csv` lists no portable, self-contained production launcher. The retained families are source-available but require external inputs, private checkpoints, or missing configurations. `legacy_gep` is explicitly historical. Any later public launcher promotion must begin with the `code/slurm/` family, add memory/path checks, and prove that it does not invoke GEP material.

## Delete candidates (do not delete in this phase)

- `.bak`, `*.log`, `*.out`, `*.err`, and `*JOB_ID*.txt` rows marked not directly consumed in `DIRECTORY_SOURCE_DATA_AUDIT.csv`.
- Historical/legacy files are candidates only after a final content and dependency closure.

## Unresolved items

- Main 1, 6--9, and 14 directory contracts are mismatched with their declared scripts and require explicit source-file identification.
- Main 12--13 have `NO_SCRIPT_FOUND`; retain their source directory until a generator is recovered or the assets are declared final-only.
- Supplementary S6 chooses the first existing shooting candidate at runtime; the contract must record the selected candidate per seed.

## Contract-resolution update

All 10 previously ambiguous rows are resolved in `MANIFEST_CONTRACT_RESOLUTION.csv`. No scientific data move is proposed for this phase. The Main 8 raw builder remains historical and non-runnable against the current tree; its validated canonical PNG is therefore the correct current publication contract.
