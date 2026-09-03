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
- p_rel: 0.9896412210296998
- q_rel: 0.9698551840398061
- p_y_rel: 0.9698551840398061
- rho_rel: 0.9896412210296999
- u_rel: 0.9932179457559995
- v_rel: 0.9693581618345627
- gamma_rel: 8.8561180247988
- align_scale_real: 0.3880898264907077
- align_scale_imag: 0.5759394712540499
- amp_mask_frac: 0.05
