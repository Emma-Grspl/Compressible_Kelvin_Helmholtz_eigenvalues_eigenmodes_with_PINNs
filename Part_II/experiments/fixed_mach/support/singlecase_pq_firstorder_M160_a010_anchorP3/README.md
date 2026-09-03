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
- p_rel: 0.8917108948054498
- q_rel: 0.8928236422075058
- p_y_rel: 0.8928236422075058
- rho_rel: 0.8917108948054496
- u_rel: 0.9110091218153417
- v_rel: 0.9066238813666831
- gamma_rel: 1.4865365457283204
- align_scale_real: 0.9257341670435022
- align_scale_imag: 0.26535313993020404
- amp_mask_frac: 0.05

## Sparse modal anchor configuration

- anchor_points: 3
- anchor_ymax: 12.0
- w_anchor_p: 10.0
- w_anchor_q: 0.0
- best_epoch: 5984
- best_loss: 0.0011769504471019648
