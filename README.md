# Part II — Supersonic spectral branches

## Overview

Part II extends the compressible Kelvin–Helmholtz study to the
**supersonic propagative regime**, where the unstable eigenvalue is genuinely
complex,

\[
c = c_r + i c_i,
\]

and both the real phase velocity \(c_r\) and the growth-rate component \(c_i\)
must be identified.

The supersonic problem is more difficult from a spectral point of view than the
subsonic regime considered in Part I. Branch localization becomes more
delicate, classical root finding can be sensitive to initialization in
difficult regions, and the neural model must represent a two-component complex
spectral quantity rather than only a growth-rate branch.

The final Part II workflow therefore combines:

- an independent classical **Riccati shooting reference**;
- validation against digitized **Blumen** data;
- sparse complex-eigenvalue supervision;
- a **piecewise-local neural spectral atlas**;
- physics-informed modal constraints;
- neural localization of the relevant classical branch;
- final **PINN-seeded Riccati shooting**;
- modal reconstruction from the converged classical solution;
- branch-localization, routing, multiseed, residual, threshold-sensitivity,
  failure-basin, and computational-cost controls.

The role of the neural model is deliberately limited.

The final high-accuracy supersonic eigenvalue and eigenmode are obtained from
the **classical Riccati shooting solver**. The PINN is used to organize and
localize the spectral search; it is not presented as the final eigensolver.

Part II contains **25 validated publication figures** together with their
machine-readable provenance.

---

## Scientific problem

The compressible Kelvin–Helmholtz instability is formulated as a
**non-self-adjoint eigenvalue problem** for the complex phase velocity

\[
c = c_r + i c_i.
\]

For the pressure perturbation \(\hat p(y)\), the governing second-order
equation is

\[
\hat p''
-
\frac{2U'}{U-c}\hat p'
-
\alpha^2
\left[
1-M^2(U-c)^2
\right]\hat p
=
0,
\]

where

- \(M\) is the Mach number,
- \(\alpha\) is the streamwise wavenumber,
- \(U(y)\) is the base-flow velocity profile,
- \(c_r\) controls the propagative component of the mode,
- \(c_i\) controls its temporal growth rate.

Unlike the subsonic branch studied in Part I, the supersonic regime requires
the simultaneous localization of both components of the complex eigenvalue.

The operator remains non-self-adjoint. A small differential residual, a smooth
neural prediction, or proximity to a known parameter point is therefore not by
itself sufficient to establish that a prediction belongs to the desired
classical spectral branch.

Independent classical validation remains essential.

---

## Numerical strategy

The final Part II methodology deliberately separates neural branch
localization from high-accuracy classical eigensolution.

### 1. Independent classical reference

A **Riccati shooting formulation** provides the classical supersonic spectral
reference.

The classical solver is used to:

- establish admissible complex roots;
- validate the target unstable branch;
- compare with digitized Blumen data;
- generate or validate sparse spectral supervision;
- quantify branch-localization failures;
- provide the final high-accuracy eigenvalue;
- reconstruct the associated eigenmode.

The classical solution remains the reference against which the neural and
hybrid workflows are evaluated.

---

### 2. Independent comparison with Blumen data

The supersonic branch is checked against external spectral information derived
from Blumen's published results.

These comparisons provide an independent physical validation of the classical
reference, including the propagative component \(c_r\) where appropriate.

The repository retains the corresponding digitized data, classical
calculations, uncertainty controls, and article figures.

The Blumen comparison is a validation target and should not be confused with
the neural branch-localization mechanism.

---

### 3. Sparse complex-eigenvalue supervision

The neural spectral model is trained from a sparse collection of complex
eigenvalue anchors.

Unlike Part I, where the retained scalar spectral quantity is primarily the
growth-rate branch \(c_i\), the supersonic atlas must represent the complex
spectral structure involving both

\[
c_r
\qquad\text{and}\qquad
c_i.
\]

The retained production study uses the sparse **`N76` anchor configuration**.

As in Part I, this anchor count should be interpreted within the tested
training, routing, and sampling protocol. It is not claimed to be a universal
minimum for the supersonic eigenvalue problem.

---

### 4. Piecewise-local neural spectral atlas

The supersonic spectral domain is represented through a
**piecewise-local atlas of neural charts** rather than by a single global
network.

The retained atlas contains **12 local charts**.

Each chart learns a locally coherent representation of the complex spectral
branch over a restricted parameter region.

Hard routing determines which chart is queried for a given parameter point.

This organization reduces the burden placed on a single network when
representing a difficult multi-branch, non-self-adjoint spectral landscape.

The atlas should therefore be interpreted as

> **a hard-routed collection of locally valid spectral models**

rather than as a globally continuous approximation of the complete
supersonic spectrum.

---

### 5. Physics-informed modal constraints

The spectral atlas is coupled to physics-informed modal information.

Differential residuals constrain the local predicted fields and provide an
additional physical-consistency test away from directly supervised spectral
anchors.

As in Part I, a low physics residual does not independently establish correct
branch membership.

Physics and spectral localization therefore play complementary but distinct
roles:

- sparse spectral information identifies the relevant branch;
- physics constrains the associated local field;
- classical shooting establishes the final admissible eigenpair.

---

### 6. PINN-seeded Riccati shooting

The central hybrid step in Part II is **PINN-seeded Riccati shooting**.

The neural atlas provides a local estimate of the relevant complex spectral
region. This information is then used to initialize or localize the subsequent
classical root search.

The Riccati shooting solver performs the actual nonlinear spectral solve.

The converged classical root — not the initial neural estimate — is retained as
the final high-accuracy result.

The final method is therefore better described as

> **neural branch localization followed by classical eigensolution**

than as a neural eigensolver.

---

### 7. Modal reconstruction

Once the complex eigenvalue has been established by the classical shooting
solver, the corresponding eigenmode is reconstructed from the converged
classical solution.

The retained modal workflow includes the pressure representation and the
derived physical perturbation fields required for article-level validation.

The final modal result is therefore attached to the converged classical
spectral root rather than to an unconstrained raw neural eigenvalue.

---

## Neural architecture

The neural component of Part II couples spectral and modal information over
local regions of the supersonic parameter domain.

The spectral component represents the local complex eigenvalue,

\[
(M,\alpha)
\longmapsto
(c_r,c_i),
\]

while the physics-informed component constrains the associated local modal
representation.

Training combines, depending on the retained experiment:

- sparse complex-eigenvalue supervision;
- local parameter sampling;
- differential physics residuals;
- boundary and far-field constraints;
- modal consistency terms;
- chart-local training and routing.

The network is therefore used primarily as a
**branch-localization model and physics-constrained local representation**.

It is not the final high-accuracy spectral solver.

---

## Main scientific results

### A classical supersonic reference can be validated independently

The retained supersonic shooting workflow constructs the target complex branch
without relying on a neural eigensolution.

Its spectral behaviour is compared with independent Blumen data and with
dedicated pointwise, convergence, matching-location, and far-field controls.

This classical layer remains the reference throughout Part II.

---

### Sparse neural supervision can localize the relevant complex branch

The sparse complex-anchor experiments show that a local neural representation
can provide useful information about the relevant spectral region over the
tested supersonic domain.

The retained production configuration uses the `N76` sparse-anchor set and a
12-chart piecewise-local atlas.

The purpose of this atlas is not to provide the final numerical root directly.
Its role is to narrow the relevant spectral region sufficiently to support
downstream classical solving.

---

### PINN-informed and generic shooting must be distinguished

Part II contains explicit controls comparing informed and non-informed
classical localization strategies.

These include:

- generic versus PINN-seeded initialization;
- budget-matched localization controls;
- failure-basin analysis;
- difficult-point shooting studies.

This distinction is important because a successful classical solve after
neural initialization does not imply that the neural prediction itself has the
accuracy of the converged classical root.

The quantity of interest is the behaviour of the
**combined localization-plus-shooting workflow**.

---

### Physics residuals do not by themselves establish branch recovery

Physics-informed residuals are used to constrain local modal fields and to
audit physical consistency.

Part II does **not** claim that a lower physics residual automatically produces
better spectral branch recovery.

Branch identification is assessed independently using the classical shooting
condition and retained spectral controls.

This prevents differential consistency from being conflated with spectral
admissibility.

---

### Routing is audited independently

Because the final neural representation is piecewise-local and hard-routed,
chart transitions are audited explicitly.

The repository preserves routing-boundary and inter-chart controls designed to
identify discontinuities or branch inconsistencies at chart interfaces.

The resulting claims are deliberately local: Part II does not assume or claim
global continuity of the complete neural atlas.

---

### Robustness is tested across seeds and thresholds

The retained reviewer-style experiments include multiseed and
threshold-sensitivity studies.

These controls are used to distinguish stable qualitative behaviour from
results that depend strongly on a particular initialization, chart boundary,
threshold, or training seed.

The repository preserves these experiments separately from the main
publication assets so that the evidential chain remains inspectable.

---

### The final solution remains classical

Regardless of the quality of the neural initialization, the final
high-accuracy eigenvalue is obtained through the Riccati shooting solver.

The neural atlas therefore does not replace the classical eigensolver.

This distinction is particularly important in difficult regions of the
supersonic spectrum, where small changes in initialization can alter which
classical basin is reached.

---

### Computational-cost results are protocol dependent

Part II contains dedicated computational-cost measurements, including the
retained **`COST500`** benchmark data.

These measurements compare the practical cost of the tested localization and
shooting strategies.

No **universal speedup claim** is made.

Any observed acceleration or overhead depends on the tested parameter
distribution, initialization strategy, solver tolerances, hardware, and
benchmark protocol.

---

## Interpretation and limitations

The Part II workflow should not be interpreted as a neural replacement for
classical spectral analysis.

The retained audits establish several important boundaries:

- the final high-accuracy eigenvalue is obtained by **Riccati shooting**;
- neural predictions provide branch localization rather than final
  eigensolutions;
- low physics residual does not establish correct spectral branch membership;
- the neural atlas is locally organized and hard-routed rather than globally
  continuous;
- sparse-anchor counts such as `N76` are protocol-dependent empirical choices,
  not universal optima;
- branch-localization performance must be separated from the accuracy of the
  converged classical root;
- difficult spectral regions can remain sensitive to initialization;
- computational-cost conclusions are benchmark-specific;
- PINN-informed shooting is not claimed to provide a universal speedup.

These limitations are part of the scientific interpretation and are preserved
explicitly through the article controls and experiment tree.

---

## Key retained datasets and configurations

Several naming conventions recur in the final Part II workflow.

### `N76`

The retained sparse complex-eigenvalue anchor configuration used by the
production spectral-atlas study.

### `T401`

The retained high-resolution shooting dataset used in the final spectral,
branch-localization, and modal validation workflows.

### `COST500`

The retained computational-cost benchmark used for article-level timing and
threshold-sensitivity controls.

These labels identify specific retained experiment configurations. They should
not be interpreted as universal numerical prescriptions outside the
corresponding protocols.

---

## Selected article assets

Part II contains **25 validated publication figures**, indexed by the
machine-readable
[`article/FIGURE_MANIFEST.csv`](article/FIGURE_MANIFEST.csv).

The following figures provide a compact visual summary of the Part II
workflow.

### Supersonic complex branch

![Supersonic branch tracking](assets/article_sources/Fig_supersonic_branch_tracking_cr_ci.png)

**Main 1 — Supersonic branch tracking in \(c_r\) and \(c_i\).**

The supersonic problem requires localization of both components of the complex
phase velocity,

\[
c = c_r + i c_i,
\]

rather than only the growth-rate component.

---

### Classical reference and Blumen comparison

![Blumen isolines and classical reference](assets/article_sources/Fig_supersonic_blumen_isolines.png)

**Main 2 — Blumen isolines and the classical supersonic reference.**

The Riccati shooting solution is compared with the external Blumen spectral
reference and remains the independent classical benchmark used throughout
Part II.

---

### Piecewise-local atlas

![Supersonic atlas geometry](assets/article_sources/Fig_supersonic_atlas_geometry.png)

**Main 6 — Piecewise-local atlas geometry.**

The supersonic parameter domain is represented through a hard-routed
collection of local neural charts rather than a single globally continuous
network.

---

### Sparse complex-anchor budget

![N76 sparse anchor budget](assets/article_sources/Fig_supersonic_anchor_budget_N76.png)

**Main 7 — `N76` sparse complex-anchor configuration.**

Sparse complex-eigenvalue anchors provide local branch information to the
neural atlas. `N76` denotes the retained production configuration under the
tested protocol and is not interpreted as a universal minimum.

---

### PINN-informed versus generic shooting

![PINN-informed versus generic shooting](assets/article_sources/Fig_T401_matched_seeded_vs_generic.png)

**Main 10 — Matched PINN-informed versus generic shooting control.**

This comparison isolates the role of neural branch localization from the
subsequent classical solve. The PINN supplies spectral information used to
localize or initialize the root search, while Riccati shooting remains the
actual eigensolver.

---

### Representative supersonic eigenmode

![Representative supersonic mode](assets/article_sources/Fig_supersonic_representative_mode_M140_a018.png)

**Main 4 — Representative mode at \(M=1.40,\alpha=0.18\).**

The retained modal reconstruction illustrates the eigenfunction associated
with the converged classical spectral solution.

---

### Computational cost and robustness

![Computational cost and robustness](assets/article_sources/Fig_supersonic_computational_cost_and_robustness.png)

**Main 14 — Computational-cost and robustness summary.**

The retained benchmark characterizes the practical cost of the tested
localization and shooting strategies. The corresponding conclusions are
protocol-dependent and are not presented as a universal speedup result.

---

More detailed controls — including branch-recovery diagnostics, T401
raw-versus-corrected analysis, failure-basin mapping, independent physics
residuals, multiseed ablations, routing discontinuities, inter-chart mismatch,
spectral and modal convergence, and `COST500` threshold sensitivity — are
retained in the full publication set and documented in
[`article/FIGURE_MANIFEST.csv`](article/FIGURE_MANIFEST.csv).

---

## Repository structure

```text
Part_II/
├── article/              # publication manifest and article-facing material
├── assets/               # validated scientific data and canonical assets
├── code/
│   ├── configs/          # canonical production/validation configurations
│   ├── plots/
│   │   ├── article/      # publication wrappers and validators
│   │   └── generators/   # scientific figure/table generators
│   ├── scripts/          # established executable workflow namespace
│   ├── slurm/            # supported public/HPC launchers
│   └── src/              # reusable scientific implementation
├── experiments/          # retained experiments and reviewer-style controls
├── provenance/           # historical code, migration and audit records
├── tests/                # lightweight CPU regression tests
├── PROJECT_STRUCTURE.md
├── README.md
├── REPRODUCIBILITY.md
└── requirements.txt
```

Unlike Part I, Part II retains `code/scripts/` as a separate executable
namespace.

This is intentional: it reflects the established organization of the
supersonic workflows and avoids restructuring working scientific entry points
solely for cosmetic symmetry with Part I.

---

## Active classical implementation

The final supersonic classical implementation is retained under

```text
code/src/classic_supersonic_reference/
```

and

```text
code/src/classical_solver/supersonic/
```

with final evaluation and shooting workflows under the established

```text
code/scripts/
```

namespace.

The active final method is **Riccati shooting**.

Historical generalized-eigenvalue implementations from earlier Part II
development are isolated under

```text
provenance/legacy_gep/
```

and are not part of the final computational workflow.

---

## Configurations

Canonical Part II configurations are stored under

```text
code/configs/
```

and indexed by

[`code/configs/CONFIG_MANIFEST.csv`](code/configs/CONFIG_MANIFEST.csv).

The public configuration tree separates production, validation, and retained
experimental configurations.

Historical configuration locations are not part of the active workflow.

---

## Experiments and reviewer controls

Scientific controls are grouped under

[`experiments/`](experiments/).

The retained experiment families include, where applicable:

- classical convergence and validation;
- PINN-seeded shooting;
- 12-chart atlas experiments;
- sparse-anchor studies;
- direct neural predictions;
- computational-cost benchmarks;
- Blumen \(c_r\) validation;
- failure-basin analysis;
- matched branch-localization controls;
- routing audits;
- multiseed robustness;
- threshold sensitivity;
- physics-residual audits;
- modal reconstruction.

Reviewer-style controls are kept close to the scientific experiment they
support rather than in a separate top-level script tree.

---

## Installation

From `Part_II/`, create and activate a Python environment appropriate for the
target CPU/GPU platform:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a workflow imports the local package namespace directly, follow the
environment setup documented in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

The public dependency list describes the scientific environment; exact
platform-specific or HPC environments are documented separately where
necessary.

---

## Reproducibility

The canonical reproduction instructions are given in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

They distinguish between:

- lightweight repository validation;
- classical shooting calculations;
- article-asset inspection;
- retained neural/atlas workflows;
- PINN-seeded shooting;
- modal reconstruction;
- computational-cost evaluation;
- more expensive training or HPC workflows.

Figure-specific provenance is documented by
[`article/FIGURE_MANIFEST.csv`](article/FIGURE_MANIFEST.csv),

while repository organization and active/historical boundaries are documented
in
[`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).

---

## Tests

From the repository root, run the lightweight Part II regression suite with

```bash
python -m pytest -q Part_II/tests
```

or from inside `Part_II/` with

```bash
python -m pytest -q tests
```

The current public Part II suite contains **4 lightweight regression tests**.

The article publication layer can additionally be checked from the repository
root with

```bash
python Part_II/code/plots/article/validate_article_figures.py
```

The current validated publication manifest contains **25 figures**.

---

## Publication provenance

The article manifest

[`article/FIGURE_MANIFEST.csv`](article/FIGURE_MANIFEST.csv)

is the machine-readable publication contract.

It records the mapping between publication assets and their retained scientific
dependencies.

The final dependency audit closes the publication graph with **zero missing
references**.

Historical implementations, obsolete plot paths, legacy GEP material, and
repository-migration records are retained under

[`provenance/`](provenance/)

rather than mixed with the active scientific workflow.

---

## Checkpoints and model state

Part II does not expose a dedicated root-level `models_saved/` directory.

The former checkpoint-policy documentation is retained under

```text
provenance/checkpoint_policy/
```

and model-dependent workflows should be followed through the exact paths
documented in the corresponding experiments and reproducibility instructions.

This prevents a documentation-only directory from being presented as a public
checkpoint release.

---

## Article assets

The curated Part II publication layer is indexed by
[`article/FIGURE_MANIFEST.csv`](article/FIGURE_MANIFEST.csv).

Canonical publication figures are stored under

[`assets/article_sources/`](assets/article_sources/).

The current manifest contains **25 validated publication figures** covering
the main scientific workflow and supplementary validation controls.

The figure manifest is the authoritative mapping between:

- article labels;
- canonical figure files;
- generating scripts;
- source data;
- integrity hashes;
- figure-specific notes.

---

## Citation

Formal citation metadata will be added once the associated publication
metadata is finalized.

Until then, when using the code, figures, or numerical results, please cite the
corresponding article or the repository URL.

---

## License

This repository is distributed under the **BSD 3-Clause License**.

See the repository-level [`LICENSE`](../LICENSE) file for details.
