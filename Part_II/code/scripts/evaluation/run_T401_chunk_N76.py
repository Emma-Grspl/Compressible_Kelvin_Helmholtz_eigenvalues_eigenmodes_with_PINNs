from __future__ import annotations

import os
from pathlib import Path

import scripts.benchmarks.benchmark_atlas2d_shooting_continuousM_Nvar as bench


chunk = int(
    os.environ["T401_CHUNK"]
)

if not 0 <= chunk <= 7:
    raise RuntimeError(
        f"Invalid T401_CHUNK={chunk}"
    )

repo = Path.cwd()

input_path = (
    repo
    / "assets/pinn_supersonic/"
      "atlas2d_v1_continuousM/N76/"
      "shooting_T401/chunks/"
    / f"T401_chunk_{chunk:02d}.csv"
).resolve()

output_root = (
    repo
    / "assets/pinn_supersonic/"
      "atlas2d_v1_continuousM/N76/"
      "shooting_T401/results/"
    / f"chunk_{chunk:02d}"
).resolve()

if not input_path.is_file():
    raise FileNotFoundError(
        input_path
    )

output_root.mkdir(
    parents=True,
    exist_ok=True,
)

# ------------------------------------------------------------
# Locate the two validated N76 path globals dynamically.
# This avoids depending on their variable names.
# ------------------------------------------------------------

path_globals = {
    name: value
    for name, value in vars(bench).items()
    if isinstance(value, Path)
}

input_matches = [
    name
    for name, value in path_globals.items()
    if str(value).endswith(
        "atlas2d_v1_continuousM/N76/"
        "validation/"
        "N76_validation_predictions_64.csv"
    )
]

output_matches = [
    name
    for name, value in path_globals.items()
    if str(value).endswith(
        "atlas2d_v1_continuousM/N76/"
        "shooting_validation"
    )
]

if len(input_matches) != 1:
    raise RuntimeError(
        "Could not uniquely identify N76 "
        f"input global: {input_matches}"
    )

if len(output_matches) != 1:
    raise RuntimeError(
        "Could not uniquely identify N76 "
        f"output global: {output_matches}"
    )

input_name = input_matches[0]
output_name = output_matches[0]

setattr(
    bench,
    input_name,
    input_path,
)

setattr(
    bench,
    output_name,
    output_root,
)

print("=" * 80)
print("T401 N76 SHOOTING WORKER")
print("=" * 80)
print("chunk        :", chunk)
print("input global :", input_name)
print("input        :", input_path)
print("output global:", output_name)
print("output       :", output_root)
print()

# argparse still sees arguments passed to this wrapper,
# e.g. --resume or --limit.
bench.main()
