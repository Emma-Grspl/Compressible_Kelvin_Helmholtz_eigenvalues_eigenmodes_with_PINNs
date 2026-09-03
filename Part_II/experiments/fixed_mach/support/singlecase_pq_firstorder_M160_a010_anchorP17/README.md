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
- p_rel: 0.924763689065705
- q_rel: 0.9032160756204074
- p_y_rel: 0.9032160756204074
- rho_rel: 0.924763689065705
- u_rel: 0.8939310052459042
- v_rel: 0.9154838924460936
- gamma_rel: 0.846932084989334
- align_scale_real: 0.7635514654377727
- align_scale_imag: 0.2853867527989787
- amp_mask_frac: 0.05

## Sparse modal anchor configuration

- anchor_points: 17
- anchor_ymax: 12.0
- w_anchor_p: 10.0
- w_anchor_q: 0.0
- best_epoch: 5998
- best_loss: 0.0012735102045987486
