# Reproducibility -- Part II

Run the commands below from `Part_II/` after following the
[installation instructions](README.md#installation). The levels distinguish
lightweight inspection from the expensive classical and neural workflows used
to produce the supersonic study.

## Scope

Part II studies supersonic compressible Kelvin--Helmholtz spectral branches.
The final classical eigenvalue and eigenmode reference is obtained by
high-accuracy Riccati shooting. GEP material remains a diagnostic resource and
is not described here as the final classical method.

[`configs/`](configs/) is the canonical root for the normalized validation,
fixed-Mach smoke, and dense classical campaign configurations. Other campaign
snapshots remain under `code/configs/` pending later phases.

## Level 1 -- Validate publication figures

**Purpose.** Verify the 25 tracked curated PNG assets against their manifest
and canonical sources without regenerating scientific results.

```bash
python scripts/figures/validate_article_figures.py
```

**Inputs and outputs.** The command reads `article/FIGURE_MANIFEST.csv`,
`article/assets/`, and the recorded canonical assets. It reports the number of
validated figures.

**Hardware/cost.** CPU; lightweight.

## Level 2 -- Classical Riccati shooting calculations

**Purpose.** Inspect the public command interfaces for supersonic shooting and
branch reconstruction.

```bash
python code/scripts/shooting/solve_reconstruct_blumen_supersonic_shooting.py --help
python code/scripts/shooting/solve_continue_supersonic_neutral_boundary.py --help
```

**Inputs and outputs.** These entrypoints document their required spectral,
branch, and output arguments. Their production inputs depend on the selected
campaign and tracked asset tables.

**Hardware/cost.** CPU calculations can become expensive for continuation,
modal reconstruction, or dense branch campaigns. A complete production
reference calculation is not yet exposed as a single public command.

## Level 3 -- Evaluate available public neural workflow

**Purpose.** Inspect the available fixed-Mach checkpoint evaluation interface.

```bash
python code/scripts/evaluation/evaluate_pinn_supersonic_fixedM_checkpoint.py --help
```

**Limitation.** The Part II checkpoint bank is intentionally omitted from the
public repository. The command can be inspected, but checkpoint-based
evaluation requires separately obtained local binaries placed according to
[models_saved/README.md](models_saved/README.md).

**Hardware/cost.** A single evaluation is ordinarily CPU/GPU dependent on the
checkpoint and environment; no public checkpoint evaluation is guaranteed from
a fresh clone.

## Level 4 -- Production spectral evaluation

**Purpose.** Preserve the public evaluation and validation entrypoints used by
the supersonic workflow.

```bash
python code/scripts/evaluation/evaluate_validate_supersonic_branch_selector.py --help
python code/scripts/evaluation/evaluate_compare_KH_shoot_to_validated.py --help
```

**Limitation.** Not yet exposed as a single public entrypoint. These commands
may require campaign-specific inputs, results, or local checkpoints.

Campaign-specific paths are organized under [`experiments/`](experiments/).
This is an active workspace containing both run-specific inputs and derived
outputs, rather than the canonical publication-results directory. The
family-level dependencies are recorded in
[`experiments/EXPERIMENT_REGISTRY.csv`](experiments/EXPERIMENT_REGISTRY.csv);
validated publication-facing outputs are placed under `results/` where that
normalization is complete.

**Hardware/cost.** CPU for inspection; potentially expensive for actual
campaign evaluation.

## Level 5 -- Full training

**Purpose.** Retain the documented neural training implementations.

```bash
python code/scripts/training/train_global_supersonic_kappa_q_logamp.py --help
```

**Limitation.** Not yet exposed as a single public production entrypoint with
a compact public configuration and distributed checkpoint set. Do not infer a
production training command from archived launchers or experiment outputs.

**Hardware/cost.** GPU/HPC expected; expensive. Full training is not run in
CI.

## Historical execution metadata

Some retained provenance files, logs, and experiment outputs contain historical
Jean-Zay or local-workstation paths. They record execution history and are not
required to validate the public article package.
