# pinn_subsonic

Curated software/model layer for the validated subsonic Kelvin-Helmholtz PINN
atlas. The historical `scripts/`, `src/`, `model_saved/` and
`assets/pinn_subsonic/` trees remain authoritative and unchanged. This package
adds a human-readable, non-destructive organization layer.

## Scope

This package covers only the subsonic PINN:

- fixed-Mach and parametric `(alpha, Mach)` training;
- sparse scalar `c_i` supervision and physics-only baselines;
- joint spectral/modal chart training;
- seedGEP modal networks and atlas charts;
- spectral, modal, overlap, coverage and article diagnostics;
- associated checkpoints, figures and CSV tables.

It does not contain the supersonic PINN or either classical reference solver.
The subsonic classical solver remains a shared dependency under
`classical_solver/subsonic/`.

## Directory structure

```text
pinn_subsonic/
├── src/
│   ├── configs/       configuration notes; no synthetic YAML
│   ├── models/        copied core network and physics modules
│   ├── training/      copied trainers plus compatibility entry points
│   └── plot/          compatibility entry points for figures and audits
├── checkpoints/
│   ├── joint_atlas_v2/<chart_id>/
│   ├── experiments/<experiment>/
│   └── smoke/<experiment>/
├── assets/
│   ├── modes/
│   ├── ci/
│   └── atlas/
├── datas/
│   ├── modes/
│   ├── ci/
│   └── atlas/
├── MIGRATION_REPORT.md
└── migrate_subsonic_tree.py
```

The pre-existing curated directories `configs/`, `models/`, `scripts/`,
`slurm/` and `data/` are retained for compatibility. Nothing was moved or
deleted.

## Models

### Core model

`src/models/model_kh_subsonic_pinn.py` defines:

- `FourierEncoding`;
- `KHSubsonicFixedMachPINN`;
- `KHSubsonicMultiMachPINN`;
- compatibility helpers for loading fixed-Mach checkpoints.

The fixed-Mach model has a modal branch and a separate spectral `c_i` branch.
The multi-Mach model extends the input dependence to `(xi, alpha, Mach)`.

### Physics components

`src/models/model_physics_residual.py` contains the pressure and Riccati
residuals, compactified-coordinate mapping, boundary losses, normalization,
phase constraints, localization terms and physical shooting/matching losses.

### Joint atlas families

The current chart architecture is implemented in the historical joint-training
entry point and exposed through
`src/training/train_atlas_chart_joint_ci_mode.py`. It combines a trainable
`CiAtlasNet` with one of three modal families:

- `pq_legacy`;
- `pq_etaaware`;
- `pQscaled`.

These classes remain partly defined inside historical training scripts. Moving
them into standalone model modules would be a scientific refactor and is
therefore recorded as technical debt rather than performed here.

## Training entry points

The exhaustive source-to-entry-point list is in
`datas/atlas/Table_file_inventory.csv`. Main entry points are:

| Entry point | Objective | Main outputs |
|---|---|---|
| `train_fixed_mach_pinn.py` | Fixed-Mach spectral/modal PINN | checkpoint, history, audits |
| `train_parametric_alpha_mach_pinn.py` | Parametric `(alpha, Mach)` PINN | checkpoint and 2D audits |
| `train_parametric_ci_stage0_anchor_lock.py` | Sparse spectral Stage 0 | locked `c_i` checkpoint |
| `train_atlas_chart_joint_ci_mode.py` | Joint chart `c_i` and mode training | `model_best.pt`, `model_state.pt`, diagnostics |
| `train_atlas_modal_seeded_gep_pq.py` | Continuous seedGEP modal field | field checkpoint and profiles |
| `train_atlas_modal_seeded_gep_etaaware.py` | Eta-aware modal family | field checkpoint and profiles |
| `train_atlas_modal_seeded_gep_qscaled.py` | Scaled-derivative modal family | field checkpoint and profiles |
| `train_fixed_mach_ci_anchor_budget_ablation.py` | 4/8/16-anchor ablation | spectral comparison tables |
| `train_fixed_mach_ci_supervision_comparison.py` | Physics-only versus sparse `c_i` | checkpoints and comparison data |
| `train_parametric_pressure_pq_firstorder.py` | First-order pressure system | pressure/derivative checkpoint |
| `train_parametric_pqab_regularized_primitives.py` | Regularized primitive reconstruction | `p,q,A,B` checkpoint |
| `train_parametric_pq_hard_velocity_reconstruction.py` | Hard reconstruction of `u,v` | modal checkpoint |

Most files in `src/training/` are thin compatibility entry points. They execute
the original script with the same CLI and scientific implementation. The two
`training_core_*` modules and `sampling_subsonic.py` are actual copies with only
package-local imports changed.

No training configuration was converted into YAML because no operational
PINN-subsonic YAML was found. Parameters remain in CLI defaults, Python
dataclasses, CSV chart manifests and checkpoint metadata.

## Plot entry points

`src/plot/` exposes all identified subsonic plotting and article scripts.
Important groups are:

- `plot_kh_subsonic_ci_*`: `c_i`, supervision and spectral heatmaps;
- `plot_kh_subsonic_*mode*`: representative modes and field errors;
- `build_ci_and_atlas_assets.py`: primary spectral and atlas assets;
- `make_mode_pdfs.py` and `make_representative_mode_figure.py`: modal figures;
- `plot_03_plot_overlap_consistency.py`: chart overlap consistency;
- `plot_04_plot_near_neutral_results.py`: near-neutral diagnostics;
- `build_fullrect_publication_assets.py`: full-rectangle atlas assets;
- `plot_08_benchmark_subsonic_runtime.py`: runtime comparison.

The wrappers preserve each original script's inputs and outputs. Defaults may
still point to historical workspace paths relative to the repository.

## Checkpoints

`checkpoints/joint_atlas_v2/<chart_id>/` preserves 49 chart directories with:

- `model_best.pt`;
- `model_state.pt`;
- `joint_training_metadata.json`.

`checkpoints/experiments/` preserves non-smoke local experiments, including
Stage 0, Stage 1, Stage 1bis and the single-case physics baseline.

`checkpoints/smoke/` keeps smoke checkpoints separate from scientific runs.
All checkpoint files are hard links when source and destination share a
filesystem; checkpoint bytes and internal keys were not changed.

## Data

- `datas/modes/`: profile fields, modal errors, phase/amplitude metrics and
  modal validation tables.
- `datas/ci/`: `c_i`, growth rate, sparse-anchor and spectral comparison tables.
- `datas/atlas/`: chart histories, coverage, routing, global validation and the
  migration inventory.

The original relative source directory is retained below each category. CSV
files are named `Table_*`.

## Assets

`pinn_subsonic/assets/` is the complete organized figure archive:

- `modes/`: pressure, density, velocity, derivative and modal-error figures;
- `ci/`: `c_i`, growth-rate, spectral and anchor figures;
- `atlas/`: chart, coverage, overlap and global atlas figures.

The original relative source directory is retained below each category.
Figures are named `Fig_*`.

`assets/pinn_subsonic/` remains the stable selection layer:

- `article/`: the pre-existing manuscript-oriented bundle;
- `modes/`: principal modal figures;
- `ci/`: principal spectral figures;
- `atlas/`: principal atlas figures.

## Article assets

No LaTeX manuscript source containing `\includegraphics` was found in this
repository. Consequently, the existing explicit
`assets/pinn_subsonic/article/` bundle is treated as the manuscript selection,
but its usage cannot be independently verified from LaTeX.

The top-level article groups include method/architecture, sparse spectral
supervision, representative modes, modal heatmaps and supporting figures.
Historical audit and release material below `article/results_pinn/` is retained
unchanged and is not asserted to be manuscript content.

## Shared dependencies

The reorganized package intentionally depends on:

- `classical_solver.subsonic` for classical reference values and modes;
- the original `scripts/` and `scripts/dev/` entry points used by wrappers;
- PyTorch, NumPy, pandas, SciPy and Matplotlib;
- the pre-existing atlas manifests and checkpoint metadata.

The original files must remain available for wrapper execution.

## Naming conventions

- organized figures: `Fig_<subject>.<png|pdf>`;
- organized CSV files: `Table_<subject>.csv`;
- training entry points: `train_<scope>_<objective>.py`;
- plotting entry points: action-oriented `plot_`, `build_`, `make_`,
  `compare_` or `evaluate_` names;
- model modules: `model_<component>.py`;
- ASCII names with underscores and no spaces.

## Reproduction examples

Only lightweight inspection commands are shown:

```bash
python -c "import pinn_subsonic"
python -c "import pinn_subsonic.src.models"
python code/src/scripts/training/fixed_mach/physics_only/train_fixed_mach_pinn.py --help
python code/src/scripts/training/atlas/direct_pinn/train_atlas_chart_joint_ci_mode.py --help
python code/plots/scripts/pinn_subsonic/compatibility_wrapper/source_tree/pinn_subsonic/src/plot/build_ci_and_atlas_assets.py --help
python code/plots/scripts/pinn_subsonic/compatibility_wrapper/source_tree/pinn_subsonic/src/plot/plot_mode_pdfs.py --help
```

Figure generation should only be run when the referenced checkpoints and CSV
inputs are present. No command above starts training without additional user
action.

## Known ambiguities

- Direct classical modal-supervision experiments and explicitly broken scripts
  are left in place and classified `AMBIGUOUS`.
- Some figures/tables have generic names such as runtime, convergence or
  pointwise validation; they remain unclassified rather than guessed.
- No operational YAML exists for atlas geometry or training.
- Several model classes remain inline in training scripts.
- Some tracked legacy files under `assets/pinn_subsonic/mach_fixed/` are absent
  in the current worktree and could not be organized.
- A first non-destructive classification pass placed some chart profile links
  under the broad `atlas` category. Correct subject-specific links were added
  later; the redundant hard links were retained because deletion was forbidden.
- Four stable-asset name collisions contain different bytes. Descriptive
  `_from_modes` destinations preserve both variants without overwrite.

## Legacy archive

The second cleanup pass archives PINN-subsonic-only historical material under
`archive/pinn_subsonic_legacy/`. The operation is non-destructive: every move
is listed in `reports/Table_archive_manifest.csv`, and the SHA-256 before and
after each move must match.

The complete pre-move decision table remains
`datas/atlas/Table_legacy_archive_plan.csv`. Shared modules, canonical-wrapper
targets, and ambiguous experiments remain at their original paths.

## Asset provenance

`datas/atlas/Table_asset_provenance.csv` contains exactly one row for every
active PNG/PDF below `pinn_subsonic/assets/` and
`assets/pinn_subsonic/`. The table records the generator, function, output
expression, inputs, command, evidence, confidence, and SHA-256.

`ASSET_PROVENANCE.md` summarizes the active set. After archival, no active
asset has the status `ORPHAN_UNRESOLVED`.

## Reproducible figures

- `src/plot/plot_spectral_modal_architecture.py` is the canonical wrapper for
  the deterministic architecture diagram; its PNG reproduction is
  pixel-identical to the retained asset.
- `src/plot/plot_subsonic_runtime.py` reconstructs the runtime plot directly
  from `Table_subsonic_runtime.csv`; dimensions and plotted data match, while
  font and layout details are not claimed to be pixel-identical.
- Alpha-specific modal reconstructions are traced to
  `src/plot/plot_05_plot_ci4_vs_physics_modes.py`.

Reproduction evidence is recorded in
`datas/atlas/Table_asset_reproduction_checks.csv`.

## Unresolved historical assets

Four copies of `Fig_supp_ci_anchor_training_convergence` had no recoverable
generator and no unambiguous record of the exact history files used. They are
preserved under `archive/pinn_subsonic_legacy/figures/orphan_assets/` with
their original paths and hashes. They are not presented as active,
reproducible project figures.

## Compatibility wrappers

No symlink was added. Four canonical wrappers were added under `src/plot/`
for the architecture diagram, Blumen overlay, joint GEP convergence
diagnostic, and three-asset repair generator. Existing wrappers that execute
historical sources remain in place. Their source targets are classified
`KEEP_SHARED`.

## Shared dependency audit

`SCRIPT_DEPENDENCIES.md` and
`datas/atlas/Table_script_dependencies.csv` document package-local imports,
classical-solver dependencies, dynamic imports, references to `scripts/dev/`,
and paths that cannot yet be archived safely.

## Naming limitation

All 3,999 active PNG/PDF paths now use the `Fig_` prefix. The two active roots
also contain 739 operational CSV outputs whose historical filenames are read
or written directly by retained generators (`history.csv`,
`fields_vs_classic_*.csv`, and similar). They are not renamed in this pass
because doing so would break current wrapper behavior. Organized publication
tables under `pinn_subsonic/datas/` continue to use the `Table_` prefix.
