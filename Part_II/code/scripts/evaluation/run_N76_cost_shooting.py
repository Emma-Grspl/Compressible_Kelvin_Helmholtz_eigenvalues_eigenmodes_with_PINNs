from __future__ import annotations

import os
from pathlib import Path

import scripts.benchmarks.benchmark_atlas2d_shooting_continuousM_Nvar as bench


repo = Path.cwd()

kind = os.environ["COST_KIND"].strip().lower()

if kind == "cost1":

    input_path = (
        repo
        / "assets/pinn_supersonic/csv/computational_cost/cost1/"
          "table_cost1_input.csv"
    ).resolve()

    output_root = (
        repo
        / "assets/pinn_supersonic/"
          "atlas2d_v1_continuousM/N76/"
          "runtime_benchmark/cost1/result"
    ).resolve()

elif kind == "cost500":

    chunk = int(
        os.environ["COST_CHUNK"]
    )

    if not 0 <= chunk <= 9:
        raise RuntimeError(
            f"Invalid COST_CHUNK={chunk}"
        )

    input_path = (
        repo
        / "assets/pinn_supersonic/"
          "atlas2d_v1_continuousM/N76/"
          "runtime_benchmark/cost500_chunks/"
        / f"cost500_chunk_{chunk:02d}.csv"
    ).resolve()

    output_root = (
        repo
        / "assets/pinn_supersonic/"
          "atlas2d_v1_continuousM/N76/"
          "runtime_benchmark/cost500_results/"
        / f"chunk_{chunk:02d}"
    ).resolve()

else:
    raise RuntimeError(
        f"Unknown COST_KIND={kind!r}"
    )


if not input_path.is_file():
    raise FileNotFoundError(
        input_path
    )

output_root.mkdir(
    parents=True,
    exist_ok=True,
)


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
        "Could not uniquely identify "
        f"input path global: {input_matches}"
    )

if len(output_matches) != 1:
    raise RuntimeError(
        "Could not uniquely identify "
        f"output path global: {output_matches}"
    )

setattr(
    bench,
    input_matches[0],
    input_path,
)

setattr(
    bench,
    output_matches[0],
    output_root,
)

print("=" * 80)
print("N76 RUNTIME BENCHMARK")
print("=" * 80)
print("kind  =", kind)
print("input =", input_path)
print("out   =", output_root)
print()

bench.main()
