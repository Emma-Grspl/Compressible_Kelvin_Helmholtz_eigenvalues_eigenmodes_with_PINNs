#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]

SOLVER = (
    PACKAGE
    / "src"
    / "classic_supersonic_reference"
    / "solver"
    / "mstab17_supersonic_solver.py"
)

OUTPUT = PACKAGE / "provenance/solver_api_inventory.md"

SCRIPT_DIRS = [
    PACKAGE / "scripts/build",
    PACKAGE / "scripts/campaigns",
    PACKAGE / "scripts/validation",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def definition_header(
    source_lines: list[str],
    lineno: int,
) -> str:
    start = lineno - 1
    collected = []
    balance = 0

    for index in range(start, min(start + 30, len(source_lines))):
        line = source_lines[index].strip()
        collected.append(line)

        balance += line.count("(")
        balance -= line.count(")")

        if line.endswith(":") and balance <= 0:
            break

    return " ".join(collected)


def argparse_flags(tree: ast.AST) -> list[str]:
    flags = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function = node.func

        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "add_argument"
        ):
            continue

        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(
                argument.value,
                str,
            ):
                flags.append(argument.value)

    return sorted(set(flags))


def main() -> None:
    if not SOLVER.exists():
        raise FileNotFoundError(SOLVER)

    source = SOLVER.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)

    output = [
        "# Frozen solver API inventory",
        "",
        f"- Solver: `{SOLVER.relative_to(PACKAGE.parent)}`",
        f"- SHA-256: `{sha256(SOLVER)}`",
        "",
        "## Top-level functions",
        "",
    ]

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]

    if not functions:
        output.append("- None")
    else:
        for node in functions:
            output.append(
                f"- Line {node.lineno}: "
                f"`{definition_header(lines, node.lineno)}`"
            )

    output.extend([
        "",
        "## Classes and methods",
        "",
    ])

    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and not node.name.startswith("_")
    ]

    if not classes:
        output.append("- None")
    else:
        for class_node in classes:
            output.append(
                f"### `{class_node.name}` — line {class_node.lineno}"
            )
            output.append("")

            methods = [
                node
                for node in class_node.body
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                )
            ]

            for method in methods:
                output.append(
                    f"- Line {method.lineno}: "
                    f"`{definition_header(lines, method.lineno)}`"
                )

            output.append("")

    output.extend([
        "## Existing scripts referring to the solver",
        "",
    ])

    matched_scripts = 0

    for directory in SCRIPT_DIRS:
        for script in sorted(directory.glob("*.py")):
            text = script.read_text()

            if (
                "mstab17_supersonic_solver" not in text
                and "MSTAB" not in text
                and "SupersonicSolver" not in text
            ):
                continue

            matched_scripts += 1
            script_tree = ast.parse(text)
            flags = argparse_flags(script_tree)

            output.append(
                f"### `{script.relative_to(PACKAGE.parent)}`"
            )
            output.append("")

            if flags:
                output.append(
                    "- CLI arguments: "
                    + ", ".join(f"`{flag}`" for flag in flags)
                )
            else:
                output.append("- CLI arguments detected: none")

            output.append("")

    if matched_scripts == 0:
        output.append("- No solver-dependent script detected.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(output) + "\n")

    print(OUTPUT.read_text())
    print("Wrote:", OUTPUT.relative_to(PACKAGE.parent))


if __name__ == "__main__":
    main()
