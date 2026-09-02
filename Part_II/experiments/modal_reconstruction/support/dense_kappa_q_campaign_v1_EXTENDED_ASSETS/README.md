# Extended classical supersonic assets

This package merges the sealed dense classical campaign with the validated
low-M/low-alpha extension.

- Unique spectral points: 820
- Modal PDF pages: 821
- Long-format modal rows: 3116820

Files:

- `classical_supersonic_extended_ci_map.pdf`: Mach-alpha map colored by ci,
  including positive and neutral digitized Blumen points.
- `classical_supersonic_extended_reference.csv`: one row per unique point,
  with Mach, alpha, cr, ci, omega_i and provenance.
- `classical_supersonic_extended_all_modes_p_rho_u_v.pdf`: one page per
  point, with real and imaginary parts of p, rho, u and v.
- `classical_supersonic_extended_modes_long.csv.gz`: one row per
  (point, y-coordinate), containing unnormalized p, rho, u and v real and
  imaginary parts, plus kappa and q.
- `classical_supersonic_extended_modes_index.csv`: compact lookup table
  linking each point to its NPZ source and mode index.
