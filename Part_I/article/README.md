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

The entrypoints publish the validated canonical assets byte-for-byte. This
ensures that rebuilding the article package does not retrain a neural model,
rerun Riccati shooting, or recompute a dense GEP. The docstring of each
entrypoint also names the original scientific generator when available.

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
