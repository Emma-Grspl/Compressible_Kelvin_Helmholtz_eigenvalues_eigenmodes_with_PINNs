# Large local files excluded from public Git

The source supersonic workspace contains the files below. They were not copied into Part II when their size is at or above the GitHub 100 MiB limit. Checkpoints are also intentionally excluded as a local binary bank. No source files were deleted.

| Original path | Size (MiB) | Source status | Public decision |
| --- | ---: | --- | --- |
| `experiments/modal_reconstruction/legacy_outputs/rebuilt_aggregates_latest/table_all_modal_fields_candidates_normalized.csv` | 538.4 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/legacy_outputs/final_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1/table_supersonic_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1_modal_fields.csv` | 506.5 | tracked | excluded: >=100 MiB GitHub limit |
| `assets/classic_supersonic/csv/modal_reconstruction/source/table_supersonic_sparse_PINN_reference_v2_FINAL_modal_fields_tail_polished_v1.csv` | 506.4 | untracked/local | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/legacy_outputs/final_sparse_PINN_reference_v2_smallM_M18M19/table_supersonic_sparse_PINN_reference_v2_modal_fields.csv` | 479.2 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/legacy_outputs/final_sparse_PINN_reference_v2_CONFIRMED/table_supersonic_sparse_PINN_reference_v2_CONFIRMED_modal_fields.csv` | 479.2 | tracked | excluded: >=100 MiB GitHub limit |
| `assets/classic_supersonic/csv/modal_reconstruction/source/table_supersonic_sparse_PINN_reference_v2_FINAL_modal_fields_raw_confirmed.csv` | 479.1 | untracked/local | excluded: >=100 MiB GitHub limit |
| `experiments/generic_shooting/legacy_outputs/reports/table_KH_shoot_candidates_vs_FINAL23_all.csv` | 378.2 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/legacy_outputs/final_sparse_PINN_reference/table_supersonic_sparse_PINN_reference_modal_fields.csv` | 332.9 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/supersonic_general/support/classic_supersonic/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS.tar.gz` | 297.8 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/support/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/classical_supersonic_final_modes_long.csv.gz` | 288.0 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/generic_shooting/legacy_outputs/reports/table_KH_shoot_candidates_vs_FINAL23_keep_or_review.csv` | 279.3 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/legacy_outputs/rebuilt_aggregates_latest/table_supersonic_reference_core_local_modal_fields_REBUILT.csv` | 276.8 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/supersonic_general/support/classic_supersonic/dense_kappa_q_campaign_v1_EXTENDED_ASSETS.tar.gz` | 275.9 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/support/dense_kappa_q_campaign_v1_FINAL_FULL_BRANCH_ASSETS/classical_supersonic_final_modal_reference.csv.gz` | 275.4 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/support/dense_kappa_q_campaign_v1_EXTENDED_ASSETS/classical_supersonic_extended_modes_long.csv.gz` | 266.8 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/legacy_outputs/merged_modal_fields_46pts/table_supersonic_sparse_46pts_modal_fields_MERGED.csv` | 233.6 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/legacy_outputs/refined_near_valid_branch/table_refined_near_valid_modal_fields.csv` | 175.5 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/legacy_outputs/QUARANTINE_INVALID_M180_M190_20260708_121200/table_supersonic_modal_fields_p_rho_u_v_50pts.csv` | 148.1 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/legacy_outputs/curated_strict/table_supersonic_reference_core_local_modal_fields_CURATED.csv` | 142.6 | tracked | excluded: >=100 MiB GitHub limit |
| `assets/classic_supersonic/csv/modal_reconstruction/final_50pts/table_supersonic_modal_fields_p_rho_u_v_50pts.csv` | 136.6 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/legacy_outputs/final_44pts_safe_20260708_121115/table_supersonic_modal_fields_p_rho_u_v_44pts_filtered.csv` | 127.4 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/supersonic_general/support/classic_supersonic/dense_kappa_q_campaign_v1_FINAL_FREEZE.tar.gz` | 124.5 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/modal_reconstruction/legacy_outputs/final_44pts_validated_only/table_supersonic_modal_fields_p_rho_u_v_44pts_validated_only.csv` | 110.6 | tracked | excluded: >=100 MiB GitHub limit |
| `experiments/generic_shooting/legacy_outputs/reports/table_KH_shoot_candidates_vs_FINAL23_reject.csv` | 98.9 | tracked | copied: >50 MiB, below GitHub limit |
| `experiments/modal_reconstruction/legacy_outputs/campaign_smallM_low_high_alpha_scan/table_campaign_smallM_modal_fields_reconstructed.csv` | 93.7 | tracked | copied: >50 MiB, below GitHub limit |
| `experiments/modal_reconstruction/support/modal/supersonic_reference_v2_modal_raw.parquet` | 67.1 | untracked/local | copied: >50 MiB, below GitHub limit |
| `experiments/modal_reconstruction/support/modal/supersonic_reference_v2_modal_tail_polished.parquet` | 67.0 | untracked/local | copied: >50 MiB, below GitHub limit |
| `experiments/modal_reconstruction/legacy_outputs/shooting/table_supersonic_reference_core_local_modal_fields_079656f8aa.csv` | 57.5 | tracked | copied: >50 MiB, below GitHub limit |

All `models_saved/` checkpoint binaries are retained only in the original local workspace and documented by `models_saved/README.md`.
