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
- p_rel: 0.9589043724576749
- q_rel: 0.9264192346676353
- p_y_rel: 0.9264192346676353
- rho_rel: 0.9589043724576749
- u_rel: 0.9276200594136452
- v_rel: 0.9462522230341313
- gamma_rel: 0.8231621952905469
- align_scale_real: 0.5532702943054126
- align_scale_imag: 0.2456902894736804
- amp_mask_frac: 0.05

## Sparse modal anchor configuration

- anchor_points: 5
- anchor_ymax: 12.0
- w_anchor_p: 10.0
- w_anchor_q: 0.0
- best_epoch: 5949
- best_loss: 0.00221456642248064
