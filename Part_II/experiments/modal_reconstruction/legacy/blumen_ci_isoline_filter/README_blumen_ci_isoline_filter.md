# Blumen ci-isoline filtering of supersonic sparse PINN map

Input points: `assets/classic_supersonic/final_sparse_map/supersonic_sparse_modal_spectral_valid_M110_M170_alpha010_0275_46pts.csv`

Input Blumen ci digitization: `KH_RT_BLUMEN/supersonic/ci_datasets.csv`

Tolerances:
- tol_ci = 0.012
- tol_scaled = 1.0
- scale_M = 0.05
- scale_alpha = 0.025

A point is kept if at least one digitized positive-ci Blumen isoline satisfies:

`abs(ci_point - ci_isoline) <= tol_ci`

and

`scaled_distance((M,alpha), isoline) <= tol_scaled`.

Main output:
- `supersonic_sparse_map_ON_BLUMEN_CI_ISOLINES.csv`
- `supersonic_sparse_map_ALL_with_blumen_ci_isoline_distance.csv`
- `blumen_ci_isolines_overlay_sparse_points.pdf`
