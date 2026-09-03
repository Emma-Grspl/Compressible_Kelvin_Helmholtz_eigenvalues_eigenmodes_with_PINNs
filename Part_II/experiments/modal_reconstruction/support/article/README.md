# Classical supersonic article assets

This directory contains the compact set of figures and tables intended for the
article.

## Figures

- `figures/fig_ci_blumen_vs_classical_overlay.*`
  Digitized Blumen $c_i$ isolines and the 133-point classical shooting
  reference. Blumen points are displayed as scatter points only and are never
  connected by plotting lines.

- `figures/fig_modes_example_M140.*`
  Representative normalized $p$, $\rho$, $u$ and $v$ mode at
  $M=1.40$ and $\alpha=0.181250$.

- `figures/fig_validated_points_map.*`
  Coverage of the validated reference in the $(M,\alpha)$ plane.

## Blumen uncertainty

The current error-bar method is:

`conservative digitization/sampling proxy: one half of the median positive local coordinate spacing on each digitized curve`

These bars are a digitization/sampling uncertainty proxy, not a statistical
confidence interval. Replace the proxy with calibrated pixel-to-coordinate
uncertainties when explicit digitization calibration uncertainties are
available.

Median displayed coordinate uncertainties:

- Mach: `0.014522822`
- alpha: `0.0047521072`

Mach-coordinate calibration:

`already physical Mach`

## Reference scope

The primary reference contains 133 points:

- 44 jointly validated modal-spectral points with exported fields;
- 89 spectral and core-modal validated points with documented tail
  sensitivity.

Raw-confirmed modal fields are used for the representative mode.
