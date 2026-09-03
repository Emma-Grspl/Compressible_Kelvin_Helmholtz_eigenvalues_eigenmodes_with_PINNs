# Validated local supersonic PINN: M=1.90

## Selected configuration

- Configuration: S4M4
- Spectral anchors: 4
- Modal anchors: 4
- Modal anchor alphas: 0.065000, 0.115000, 0.165000, 0.200000
- Dense modal audit: 34 alpha values
- Off-anchor audit: 30 alpha values

## Spectral metrics

- cr MAE: 9.905143757737e-04
- cr maximum absolute error: 3.601867800504e-03
- ci MAE: 3.852938588348e-03
- ci maximum absolute error: 1.733498560423e-02
- omega_i MAE: 4.194552302429e-04

## Off-anchor modal metrics

- kappa NRMSE: 1.507277970499e-01
- q NRMSE: 1.829641318043e-02
- log-amplitude RMSE: 1.616699209146e-02
- amplitude relative L2: 8.899410542854e-03
- phase RMSE: 2.357498594753e-01
- pressure relative L2: 7.611260544608e-02
- pressure overlap error: 1.490732609395e-03

## Paired result against S4

- Improved off-anchor modes: 30/30
- Degraded off-anchor modes: 0/30
- Equal off-anchor modes: 0/30

Conclusion: S4M4 is validated at M=1.90.
