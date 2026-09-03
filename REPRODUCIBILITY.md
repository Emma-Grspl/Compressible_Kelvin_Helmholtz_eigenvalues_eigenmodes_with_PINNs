# Repository-level reproducibility

Part I and Part II address different Kelvin--Helmholtz regimes and retain
separate scientific workflows and dependency requirements. This root document
is an entrypoint, not a one-command global reproduction workflow.

- [Part I reproducibility](Part_I/REPRODUCIBILITY.md) covers the subsonic
  Riccati reference, selected GEP evaluation, and published neural atlas.
- [Part II reproducibility](Part_II/REPRODUCIBILITY.md) covers supersonic
  Riccati-shooting references, publication-asset validation, and the limits of
  the publicly available neural workflow.

The root CI performs lightweight CPU integrity checks only. It does not train
models, recompute dense production GEP calculations, or run expensive
production shooting campaigns. Historical HPC execution metadata may remain in
provenance records and archived outputs; it is not a requirement for the CI
workflow.
