# Compressible Kelvin-Helmholtz eigenvalues and eigenmodes with physics-informed neural networks

This repository is organized as two related studies of the compressible
Kelvin--Helmholtz eigenvalue problem.

## Part I -- Subsonic spectral branches

[Part_I](Part_I/) contains the subsonic study: a classical Riccati shooting
reference, sparse scalar spectral anchors, a piecewise-local neural atlas,
physics-informed modal constraints, and independent dense GEP evaluation with
scalar-guided eigenpair selection. The reported discrete result remains the
selected classical/GEP solution, not a replacement by the neural model.

## Part II -- Supersonic spectral branches

[Part_II](Part_II/) contains compressible supersonic Kelvin--Helmholtz
eigenvalues and eigenmodes. It includes classical Riccati shooting reference
calculations, external comparison with Blumen data, sparse complex-eigenvalue
anchors, a piecewise-local neural spectral atlas, physics-informed modal
constraints, PINN-informed local classical shooting, branch-localization
controls, independent physics and routing audits, and eigenmode
reconstruction.

The final high-accuracy supersonic eigenvalue/eigenmode is obtained by
classical Riccati shooting. The repository does not claim that the neural
atlas is a final eigensolver, that physics residuals improve branch recovery,
that chart continuity is global, or that PINN-informed shooting provides a
universal speedup.

## Layout

```text
.
├── Part_I/   # subsonic spectral branches
└── Part_II/  # supersonic spectral branches
```

Each part has its own article assets, scientific data, code, provenance, and
reproducibility documentation. Scientific assets are intentionally not mixed
between the two regimes.
