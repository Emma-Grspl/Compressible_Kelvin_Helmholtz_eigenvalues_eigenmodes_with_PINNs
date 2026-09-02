# Supersonic single-case first-order p/q PINN

- alpha=0.1
- Mach=1.6
- cr=0.299805
- ci=0.029118
- trained outputs: Re p, Im p, Re q, Im q with q approx p_y
- c is fixed from the classical spectral point
- no classical modal field is used in the training loss
- classical pressure/modes are used only for post-training diagnostics

## Diagnostics

- alpha: 0.1
- Mach: 1.6
- cr: 0.299805
- ci: 0.029118
- p_rel: 0.9361989163061064
- q_rel: 0.9115815110966434
- p_y_rel: 0.9115815110966434
- rho_rel: 0.9361989163061065
- u_rel: 0.8989238079111919
- v_rel: 0.9256385010771788
- gamma_rel: 0.8680056846768073
- align_scale_real: 0.7032985836966702
- align_scale_imag: 0.28097201394109617
- amp_mask_frac: 0.05

## Sparse modal anchor configuration

- anchor_points: 9
- anchor_ymax: 12.0
- w_anchor_p: 10.0
- w_anchor_q: 0.0
- best_epoch: 5972
- best_loss: 0.001480981475252101
