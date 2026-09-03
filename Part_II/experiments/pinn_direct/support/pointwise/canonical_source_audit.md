# Pointwise canonical source audit

## Subsonic
- Path: `assets/classic_subsonic/data/subsonic_hybrid_growth_map.csv`
- Columns: `alpha, Mach, ci, omega_i, source, primary_ci, primary_omega_i, primary_mismatch, primary_success, secondary_ci, secondary_omega_i, secondary_stage1_mismatch, secondary_stage2_mismatch`
- Rows: 225; points per source Mach: {0.0: 15, 0.0699999999999999: 15, 0.1399999999999999: 15, 0.2099999999999999: 15, 0.2799999999999999: 15, 0.35: 15, 0.4199999999999999: 15, 0.4899999999999999: 15, 0.5599999999999999: 15, 0.6299999999999999: 15, 0.7: 15, 0.7699999999999999: 15, 0.8399999999999999: 15, 0.91: 15, 0.98: 15}
- Spectral status: `primary_success` plus optional mstab17 secondary diagnostics.
- Modal fields: not stored in this CSV; p, rho, u and v are reconstructible with Mstab17SubsonicSolver.
- Provenance: output of `classical_solver/subsonic/hybrid_subsonic_scan.py`.
- Limitation: requested Mach values are not all table nodes. Linear interpolation is applied only in Mach; alpha is never interpolated.

## Supersonic
- Spectral path: `assets/classic_supersonic/csv/pinn_direct/spectral/table_supersonic_primary_modal_spectral_133pts.csv`
- Spectral columns: `point_id, Mach, alpha, cr, ci, omega_i, validation_status, has_exported_modal_fields, reference_tier, spectral_ok, modal_ok, pinning_use, best_status, best_stage1_mismatch, best_stage2_mismatch, trusted_spectral, trusted_modal, valid_spectral_candidate, valid_modal_candidate, decision, source_kind, source_short, source, modal_fields_file, modes_pdf, blumen_ci_overlay_pdf, validation_note, reference_role, selected_setting, cr_range, ci_range, max_stage1, max_edge_frac, max_center_jump, max_modal_rel_l2, max_core_rel_l2_amp_ge_5pct, max_tail_rel_l2_amp_0p1pct_to_2pct, note, stage1_mismatch, cr_box_min, cr_box_max, ci_box_min, ci_box_max, status, exception, ln_p_start_right_exact_600, stage2_mismatch_exact_600, y_limit_600, amplitude_status_600, ln_p_start_right_exact_1200, stage2_mismatch_exact_1200, y_limit_1200, amplitude_status_1200, p_edge_frac, p_center_jump, p_adjacent_jump, p_left_right_energy_ratio, rho_edge_frac, rho_center_jump, rho_adjacent_jump, rho_left_right_energy_ratio, u_edge_frac, u_center_jump, u_adjacent_jump, u_left_right_energy_ratio, v_edge_frac, v_center_jump, v_adjacent_jump, v_left_right_energy_ratio, all_finite, max_adjacent_jump, max_left_right_energy_ratio, p_ylimit_rel_l2, rho_ylimit_rel_l2, u_ylimit_rel_l2, v_ylimit_rel_l2, max_ylimit_rel_l2, reject_reasons, campaign_target_status, source_file, touches_ci_upper_0p12, touches_cr_lower_0, boundary_flag, tail_polish_available, tail_polish_version, tail_polish_note`
- Rows per Mach: {1.1: 2, 1.2: 6, 1.25: 1, 1.3: 9, 1.33: 1, 1.4: 11, 1.5: 8, 1.6: 2, 1.7: 4, 1.8: 49, 1.9: 40}
- Modal path: `experiments/modal_reconstruction/support/modal/supersonic_reference_v2_modal_raw.parquet`
- Unique modal points per Mach: {1.1: 9, 1.2: 14, 1.25: 11, 1.3: 13, 1.33: 9, 1.4: 14, 1.5: 12, 1.6: 6, 1.7: 6, 1.8: 49, 1.9: 40}
- Spectral/modal status: the two accepted canonical statuses are modal-spectral validated and core-modal validated/tail-sensitive.
- Modal fields: p, rho, u and v real/imaginary arrays are present in the canonical Parquet.
- Provenance: current 133-point primary spectral table and current v2 raw modal Parquet; UNFROZEN archives are not used.

## Selection coverage
- Subsonic rank-1 coverage: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
- Supersonic rank-1 coverage: [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
- Some Mach values have fewer than three admissible canonical candidates; no placeholder candidate is invented.
