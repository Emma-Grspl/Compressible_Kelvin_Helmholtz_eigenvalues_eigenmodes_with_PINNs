# Supersonic sparse modal/spectral reference map

Frozen sparse reference map for supersonic KH PINN experiments.

File:
- `supersonic_sparse_modal_spectral_valid_M110_M170_alpha010_0275_46pts.csv`

Content:
- 46 points
- Mach range effectively covered: 1.10 <= M <= 1.70
- alpha range effectively covered: 0.10 <= alpha <= 0.275
- all retained points satisfy:
  - ci > 0
  - best_status = validated
  - best_spectral_success = True
  - best_mode_success = True

Important:
- The requested target domain was 1.10 <= M <= 1.95 and 0.0 <= alpha <= 0.4.
- In the available files, points for M=1.8 and M=1.9 exist only as review/extension points, without explicit modal validation.
- Therefore they are not included in this frozen validated map.

Use:
- canonical sparse supersonic spectral/modal anchors for PINN training/diagnostics.
