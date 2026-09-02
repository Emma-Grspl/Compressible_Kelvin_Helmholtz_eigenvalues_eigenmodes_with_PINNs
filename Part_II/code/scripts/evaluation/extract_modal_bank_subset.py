#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mach",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-10,
    )

    args = parser.parse_args()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with np.load(
        args.input,
        allow_pickle=False,
    ) as source:
        data = {
            key: source[key]
            for key in source.files
        }

    Mach = np.asarray(
        data["Mach"],
        dtype=float,
    )

    selected_modes = np.flatnonzero(
        np.isclose(
            Mach,
            args.mach,
            rtol=0.0,
            atol=args.atol,
        )
    )

    if selected_modes.size == 0:
        raise RuntimeError(
            f"No mode found for Mach={args.mach}"
        )

    mode_ptr = np.asarray(
        data["mode_ptr"],
        dtype=np.int64,
    )

    point_indices = []

    new_ptr = [0]

    for mode_index in selected_modes:
        start = int(mode_ptr[mode_index])
        stop = int(mode_ptr[mode_index + 1])

        indices = np.arange(
            start,
            stop,
            dtype=np.int64,
        )

        point_indices.append(indices)

        new_ptr.append(
            new_ptr[-1] + len(indices)
        )

    point_indices_array = np.concatenate(
        point_indices
    )

    n_modes_total = len(Mach)
    n_points_total = int(mode_ptr[-1])

    output = {}

    for key, value in data.items():
        array = np.asarray(value)

        if key == "mode_ptr":
            output[key] = np.asarray(
                new_ptr,
                dtype=np.int64,
            )

        elif (
            array.ndim >= 1
            and len(array) == n_modes_total
        ):
            output[key] = array[selected_modes]

        elif (
            array.ndim >= 1
            and len(array) == n_points_total
        ):
            output[key] = array[
                point_indices_array
            ]

        else:
            output[key] = array

    np.savez_compressed(
        args.output,
        **output,
    )

    print("input       :", args.input)
    print("output      :", args.output)
    print("Mach        :", args.mach)
    print("modes       :", len(selected_modes))
    print("modal points:", len(point_indices_array))
    print(
        "alphas      :",
        output["alpha"].tolist(),
    )


if __name__ == "__main__":
    main()
