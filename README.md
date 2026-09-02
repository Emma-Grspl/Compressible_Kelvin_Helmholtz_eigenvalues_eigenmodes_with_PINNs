# Compressible Kelvin-Helmholtz eigenvalues and eigenmodes with physics-informed neural networks

This public repository currently contains **Part I only: subsonic spectral
branches** of the compressible Kelvin-Helmholtz eigenvalue problem.

Part I combines:

- a classical Riccati shooting reference;
- sparse scalar spectral anchors for branch localization;
- physics-informed, locally continuous neural charts;
- direct neural prediction of spectral and modal quantities;
- independent dense generalized eigenvalue problem (GEP) diagonalization;
- scalar-guided GEP eigenpair selection performed after diagonalization.

The neural estimate is not used to initialize or refine the GEP. It is used
only as a scalar selector among independently computed GEP eigenvalues. The
repository makes no claim that the direct neural model replaces a classical
eigensolver or provides a universal speedup.

## Repository layout

```text
.
|-- README.md
|-- .gitignore
`-- Part_I/
    |-- article/          # Curated manuscript figures and manifest
    |-- assets/           # Validated figures and processed scientific assets
    |-- code/             # Reproducible implementation and launchers
    |-- models_saved/     # Production checkpoints included in the public repo
    |-- results/          # Numerical audits and complementary validations
    |-- scripts/figures/  # One publishing entrypoint per article figure
    `-- docs/             # Experiment and reproducibility documentation
```

See [Part_I/README.md](Part_I/README.md) for the scientific workflow and
[Part_I/PROJECT_STRUCTURE.md](Part_I/PROJECT_STRUCTURE.md) for the detailed
directory conventions.
