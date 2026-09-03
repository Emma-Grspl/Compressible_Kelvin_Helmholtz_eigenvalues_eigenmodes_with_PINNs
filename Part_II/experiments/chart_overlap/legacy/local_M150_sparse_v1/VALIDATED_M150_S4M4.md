# Validated local supersonic PINN: M=1.50

## Selected configuration

- Configuration: S4M4
- Spectral anchors: 4
- Modal anchors: 4
- Modal anchor alphas: 0.070, 0.140, 0.205, 0.250
- Dense modal audit: 45 alpha values
- Off-anchor audit: 41 alpha values

## Spectral metrics

- cr MAE: 2.279619487562e-03
- cr maximum absolute error: 8.903751423116e-03
- ci MAE: 5.251006239949e-03
- ci maximum absolute error: 2.074032869722e-02
- omega_i MAE: 6.877192095961e-04

## Off-anchor modal metrics

- kappa NRMSE: 9.238689011688e-02
- q NRMSE: 2.636888239366e-02
- log-amplitude RMSE: 2.929699266405e-02
- amplitude relative L2: 1.652708541493e-02
- phase RMSE: 1.301840822951e-01
- pressure relative L2: 6.994379917004e-02
- pressure overlap error: 1.252251397882e-03

## Paired result against S4M2

- Improved off-anchor modes: 41/41
- Degraded off-anchor modes: 0/41

Conclusion: S4M4 is the retained fixed-Mach formulation.
