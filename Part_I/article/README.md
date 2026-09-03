# Article figures

`assets/` contains exactly the curated figure set for Part I: nine Main figures
and eight Supplementary figures. Main figures use `Fig01_...` through
`Fig09_...`; Supplementary figures use section-aware labels such as
`FigS4_1_...`.

Each row in `FIGURE_MANIFEST.csv` identifies:

- the article label and final filename;
- the canonical validated project asset;
- the corresponding entrypoint under `scripts/figures/`;
- the processed data used by the original scientific plotting code;
- the SHA-256 hash of the curated PNG.

The entrypoints publish the validated canonical assets byte-for-byte. They are
publication wrappers, not the upstream scientific plotting calculations. This
ensures that rebuilding the article package does not retrain a neural model,
rerun Riccati shooting, or recompute a dense GEP. The docstring of each
entrypoint names the original scientific generator when available.

## Figure map

`FIGURE_MANIFEST.csv` remains the machine-readable source of truth. This table
is the corresponding human-readable map of every curated figure.

| Paper label | Final asset | Plotting entrypoint | Main source data |
| --- | --- | --- | --- |
| Fig. 1 | `Fig01_subsonic_isolines_blumen_vs_classical.png` | `scripts/figures/plot_Fig01_subsonic_isolines.py` | `assets/classic_subsonic/csv/data/Table_subsonic_hybrid_growth_map.csv` |
| Fig. 2 | `Fig02_spectral_modal_architecture_subsonic.png` | `scripts/figures/plot_Fig02_spectral_modal_architecture.py` | `docs/EXPERIMENTS.md` |
| Fig. 3 | `Fig03_atlas_49_charts_Mach_alpha.png` | `scripts/figures/plot_Fig03_atlas_49_charts.py` | `assets/pinn_subsonic/csv/article/N340/Table_N340_training_plan_for_seams.csv` |
| Fig. 4 | `Fig04_fixed_Mach_four_anchor_vs_physics_only.png` | `scripts/figures/plot_Fig04_fixed_Mach_four_anchor.py` | `archive/repo_cleanup_2026-04-24/plot_presentation/subsonic_pinn/ci_supervision_vs_physics` |
| Fig. 5 | `Fig05_anchor_budget_comparison.png` | `scripts/figures/plot_Fig05_anchor_budget_comparison.py` | `assets/pinn_subsonic/csv/article/N340/Table_anchor_budget_comparison.csv` |
| Fig. 6 | `Fig06_atlas_routing_interface_consistency_N340.png` | `scripts/figures/plot_Fig06_routing_interface_consistency.py` | `assets/pinn_subsonic/csv/article/N340/Table_atlas_overlap_consistency_N340.csv` |
| Fig. 7 | `Fig07_holdout_anchors_only_vs_physics_N340.png` | `scripts/figures/plot_Fig07_holdout_anchors_vs_physics.py` | `assets/pinn_subsonic/csv/article_supplementary/N340/source_data/Table_Holdout384_pointwise_physics_vs_anchors_only.csv` |
| Fig. 8 | `Fig08_representative_mode_M05_a05_N340.png` | `scripts/figures/plot_Fig08_representative_mode.py` | `assets/pinn_subsonic/csv/article/N340/Table_Data_representative_mode_M05_a05_N340.csv` |
| Fig. 9 | `Fig09_near_neutral_growth_rate_cuts_N340.png` | `scripts/figures/plot_Fig09_near_neutral_growth_rate_cuts.py` | `results/complementary_audits/curated/section4_results/Fig_near_neutral_growth_rate_cuts_N340_data.csv` |
| Fig. S1.1 | `FigS1_1_classical_subsonic_mode_four_fields_a0p500_M0p500.png` | `scripts/figures/plot_FigS1_1_classical_mode.py` | `classical_solver/subsonic` |
| Fig. S1.2 | `FigS1_2_subsonic_error_map_classical.png` | `scripts/figures/plot_FigS1_2_classical_error_map.py` | `assets/classic_subsonic/csv/data/Table_subsonic_blumen_shooting_metrics_by_point.csv` |
| Fig. S1.3 | `FigS1_3_classical_subsonic_convergence.png` | `scripts/figures/plot_FigS1_3_classical_convergence.py` | `assets/classic_subsonic/csv/article/convergence/tables/Table_classical_subsonic_convergence.csv` |
| Fig. S3.1 | `FigS3_1_sparse_ci_anchor_map_Mach_alpha_N340.png` | `scripts/figures/plot_FigS3_1_sparse_anchor_map.py` | `assets/pinn_subsonic/csv/anchor_budget_runs/N340/Table_anchors.csv` |
| Fig. S4.1 | `FigS4_1_GEP_N_convergence.png` | `scripts/figures/plot_FigS4_1_GEP_convergence.py` | `assets/pinn_subsonic/csv/article/results_pinn/release_final/tables/Table_GEP_N_convergence.csv` |
| Fig. S5.1 | `FigS5_1_selected_GEP_error_heatmap_N340.png` | `scripts/figures/plot_FigS5_1_selected_GEP_error_map.py` | `results/complementary_audits/curated/map_1000_ci_checks.csv` |
| Fig. S6.1 | `FigS6_1_direct_neural_vs_selected_GEP_modal_error_ecdf_20_N340.png` | `scripts/figures/plot_FigS6_1_modal_error_ecdf.py` | `assets/pinn_subsonic/csv/article/N340/Table_paired_modal_validation_20_N340.csv` |
| Fig. S8.1 | `FigS8_1_near_neutral_error_scaling_N340.png` | `scripts/figures/plot_FigS8_1_near_neutral_error_scaling.py` | `results/complementary_audits/curated/validation_pointwise_canonical.csv` |

## Old-to-final filename mapping

| Original semantic filename | Final article filename |
|---|---|
| `01_subsonic_isolines_blumen_vs_classique.png` | `Fig01_subsonic_isolines_blumen_vs_classical.png` |
| `Fig_spectral_modal_architecture_subsonic.png` | `Fig02_spectral_modal_architecture_subsonic.png` |
| `Fig_atlas_49_charts_Mach_alpha.png` | `Fig03_atlas_49_charts_Mach_alpha.png` |
| `ci_curve_hybrid4_vs_physics_only.png` | `Fig04_fixed_Mach_four_anchor_vs_physics_only.png` |
| `Fig_anchor_budget_comparison.png` | `Fig05_anchor_budget_comparison.png` |
| `Fig_atlas_overlap_consistency_N340.png` | `Fig06_atlas_routing_interface_consistency_N340.png` |
| `SuppFig_holdout_anchors_only_vs_physics_N340.png` | `Fig07_holdout_anchors_only_vs_physics_N340.png` |
| `Fig_representative_mode_M05_a05_N340.png` | `Fig08_representative_mode_M05_a05_N340.png` |
| `Fig_near_neutral_growth_rate_cuts_N340.png` | `Fig09_near_neutral_growth_rate_cuts_N340.png` |
| `classical_subsonic_mode_four_fields_a0p500_M0p500.png` | `FigS1_1_classical_subsonic_mode_four_fields_a0p500_M0p500.png` |
| `02_subsonic_error_map_classique.png` | `FigS1_2_subsonic_error_map_classical.png` |
| `Fig_classical_subsonic_convergence.png` | `FigS1_3_classical_subsonic_convergence.png` |
| `Fig_sparse_ci_anchor_map_Mach_alpha_N340.png` | `FigS3_1_sparse_ci_anchor_map_Mach_alpha_N340.png` |
| `SuppFig07_GEP_N_convergence.png` | `FigS4_1_GEP_N_convergence.png` |
| `Fig_ci_error_heatmap_PINN_GEP_vs_classical_N340.png` | `FigS5_1_selected_GEP_error_heatmap_N340.png` |
| `Fig_paired_modal_error_ecdf_20_N340.png` | `FigS6_1_direct_neural_vs_selected_GEP_modal_error_ecdf_20_N340.png` |
| `Fig_near_neutral_error_scaling_N340.png` | `FigS8_1_near_neutral_error_scaling_N340.png` |
