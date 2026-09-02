# Validated local supersonic PINN: M=1.10

## Selected configuration

- Configuration: S4M4
- Spectral anchors: 4
- Modal anchors: 4
- Modal anchor alphas: 0.115000, 0.190000, 0.265000, 0.320000
- Dense modal audit: 53 alpha values
- Off-anchor audit: 49 alpha values

## Spectral metrics

- cr MAE: 5.685869644583e-03
- cr maximum absolute error: 5.073651490110e-02
- ci MAE: 7.987231573727e-03
- ci maximum absolute error: 2.499251520077e-02
- omega_i MAE: 1.545534338192e-03

## Off-anchor modal metrics

- kappa NRMSE: 4.875823608023e-02
- q NRMSE: 4.116543247163e-02
- log-amplitude RMSE: 8.514285421861e-02
- amplitude relative L2: 2.496298567585e-02
- phase RMSE: 8.138883656277e-02
- pressure relative L2: 6.390047418842e-02
- pressure overlap error: 1.356858605467e-03

## Paired result against S4

- Improved off-anchor modes: 49/49
- Degraded off-anchor modes: 0/49
- Equal off-anchor modes: 0/49

Conclusion: S4M4 is validated at M=1.10.
