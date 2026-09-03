#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent

SOLVER = (
    PACKAGE
    / "src"
    / "classic_supersonic_reference"
    / "solver"
    / "mstab17_supersonic_solver.py"
)

SCRIPT_ROOTS = [
    PACKAGE / "scripts" / "build",
    PACKAGE / "scripts" / "campaigns",
    PACKAGE / "scripts" / "validation",
]

OUTPUT = (
    PACKAGE
    / "provenance"
    / "modal_reconstruction_path.md"
)

TARGET_METHODS = {
    "base_velocity",
    "base_velocity_derivative",
    "phase_speed",
    "asymptotic_gammas",
    "get_trajectories",
    "_interp_component",
    "solve",
    "plot_mode",
}

KEYWORDS = (
    "get_trajectories",
    "ln_p_start_right",
    "p_real",
    "p_imag",
    "rho_real",
    "rho_imag",
    "u_real",
    "u_imag",
    "v_real",
    "v_imag",
    "pressure",
    "density",
    "velocity",
    "modal_fields",
    "fields_output",
    "solve_ivp",
)


def source_segment(
    path: Path,
    node: ast.AST,
) -> str:
    lines = path.read_text().splitlines()

    start = max(int(node.lineno) - 1, 0)
    end = int(
        getattr(
            node,
            "end_lineno",
            node.lineno,
        )
    )

    return "\n".join(lines[start:end])


def selected_solver_definitions() -> list[tuple[str, int, str]]:
    source = SOLVER.read_text()
    tree = ast.parse(source)

    selected = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        if node.name not in TARGET_METHODS:
            continue

        selected.append(
            (
                node.name,
                int(node.lineno),
                source_segment(SOLVER, node),
            )
        )

    return sorted(
        selected,
        key=lambda item: item[1],
    )


def matching_line_numbers(
    path: Path,
) -> list[int]:
    lines = path.read_text().splitlines()
    matches = []

    for number, line in enumerate(lines, start=1):
        lowered = line.lower()

        if any(
            keyword.lower() in lowered
            for keyword in KEYWORDS
        ):
            matches.append(number)

    return matches


def merged_context_ranges(
    line_numbers: list[int],
    total_lines: int,
    radius: int = 6,
) -> list[tuple[int, int]]:
    ranges = []

    for number in line_numbers:
        start = max(1, number - radius)
        end = min(total_lines, number + radius)

        if not ranges or start > ranges[-1][1] + 1:
            ranges.append([start, end])
        else:
            ranges[-1][1] = max(
                ranges[-1][1],
                end,
            )

    return [
        (int(start), int(end))
        for start, end in ranges
    ]


def formatted_context(
    path: Path,
    start: int,
    end: int,
) -> str:
    lines = path.read_text().splitlines()

    selected = []

    for number in range(start, end + 1):
        selected.append(
            f"{number:04d}: {lines[number - 1]}"
        )

    return "\n".join(selected)


def main() -> None:
    if not SOLVER.exists():
        raise FileNotFoundError(SOLVER)

    output = [
        "# Modal reconstruction path",
        "",
        "## Selected solver definitions",
        "",
    ]

    definitions = selected_solver_definitions()

    for name, line, segment in definitions:
        output.extend(
            [
                f"### `{name}` — line {line}",
                "",
                "```python",
                segment,
                "```",
                "",
            ]
        )

    output.extend(
        [
            "## Existing field-export code",
            "",
        ]
    )

    matching_files = []

    for root in SCRIPT_ROOTS:
        for path in sorted(root.glob("*.py")):
            numbers = matching_line_numbers(path)

            if not numbers:
                continue

            matching_files.append(path)

            lines = path.read_text().splitlines()
            ranges = merged_context_ranges(
                numbers,
                len(lines),
            )

            output.extend(
                [
                    f"### `{path.relative_to(REPO)}`",
                    "",
                ]
            )

            for start, end in ranges:
                output.extend(
                    [
                        f"Lines {start}–{end}",
                        "",
                        "```python",
                        formatted_context(
                            path,
                            start,
                            end,
                        ),
                        "```",
                        "",
                    ]
                )

    if not matching_files:
        output.append(
            "No field-export implementation was detected."
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        "\n".join(output) + "\n"
    )

    print("Solver definitions:")
    for name, line, _ in definitions:
        print(f"  {name}: line {line}")

    print()
    print("Matching scripts:")
    for path in matching_files:
        print(
            " ",
            path.relative_to(REPO),
        )

    print()
    print("Wrote:", OUTPUT.relative_to(REPO))


if __name__ == "__main__":
    main()
