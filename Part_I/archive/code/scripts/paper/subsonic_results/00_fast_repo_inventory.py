#!/usr/bin/env python3

from __future__ import annotations

import csv
import os
import re
from collections import Counter
from pathlib import Path


ROOT = Path.cwd().resolve()
OUT = ROOT / "assets/pinn_subsonic/paper_results_v1/data"
OUT.mkdir(parents=True, exist_ok=True)

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "paper_results_v1",
}

CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt"}

DATA_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".parquet",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".json",
}

FIGURE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".svg",
}

CONFIG_EXTENSIONS = {
    ".py",
    ".sh",
    ".slurm",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}

RESULT_TERMS = (
    "riccati",
    "anchor",
    "supervis",
    "single_case",
    "single-case",
    "fixed_mach",
    "fixed-mach",
    "atlas",
    "spectral",
    "modal",
    "eigenmode",
    "shooting",
    "classical",
    "gep",
    "neutral",
    "overlap",
    "runtime",
    "benchmark",
    "growth_rate",
    "growth-rate",
    "ci_",
    "_ci",
)

ANCHOR_TERMS = (
    "anchor",
    "supervis",
    "riccati",
    "hybrid",
    "reference_alpha",
    "reference-alpha",
    "sparse_ci",
    "sparse-ci",
)

CONFIG_SEARCH_TERMS = (
    "n_anchor",
    "num_anchor",
    "anchor_count",
    "anchor_alpha",
    "reference_alpha",
    "n_reference_alpha",
    "ci_supervision",
    "w_ci_supervision",
    "spectral_anchor",
    "sparse_ci",
)

ANCHOR_PATTERNS = (
    re.compile(
        r"(?i)(?:^|[/_.-])(0|4|8|16)[_-]?"
        r"(?:anchors?|anchorpoints?|anchor_points?|pts?|points?)"
        r"(?:$|[/_.-])"
    ),
    re.compile(
        r"(?i)(?:^|[/_.-])"
        r"(?:anchors?|anchorpoints?|anchor_points?)[_-]?"
        r"(0|4|8|16)(?:$|[/_.-])"
    ),
)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


checkpoints: list[str] = []
relevant_assets: list[dict] = []
anchor_named_files: list[str] = []
anchor_variants: list[dict] = []
config_hits: list[dict] = []

checkpoint_directory_counts: Counter[str] = Counter()
extension_counts: Counter[str] = Counter()

for directory, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [
        name for name in dirnames
        if name not in SKIP_DIRS
    ]

    directory_path = Path(directory)

    for filename in filenames:
        path = directory_path / filename

        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue

        relative_string = relative.as_posix()
        lower_path = relative_string.lower()
        extension = path.suffix.lower()

        extension_counts[extension or "<none>"] += 1

        if extension in CHECKPOINT_EXTENSIONS:
            checkpoints.append(relative_string)
            checkpoint_directory_counts[relative.parent.as_posix()] += 1

        if (
            extension in DATA_EXTENSIONS | FIGURE_EXTENSIONS
            and any(term in lower_path for term in RESULT_TERMS)
        ):
            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = -1

            relevant_assets.append(
                {
                    "path": relative_string,
                    "extension": extension,
                    "size_bytes": size_bytes,
                }
            )

        if any(term in lower_path for term in ANCHOR_TERMS):
            anchor_named_files.append(relative_string)

        detected_counts: set[int] = set()
        for pattern in ANCHOR_PATTERNS:
            detected_counts.update(
                int(value) for value in pattern.findall(relative_string)
            )

        for count in sorted(detected_counts):
            anchor_variants.append(
                {
                    "anchor_count": count,
                    "path": relative_string,
                }
            )

        if extension not in CONFIG_EXTENSIONS:
            continue

        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue

        if size_bytes > 2_000_000:
            continue

        try:
            with path.open(
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as handle:
                hit_count = 0

                for line_number, line in enumerate(handle, start=1):
                    lower_line = line.lower()

                    if not any(
                        term in lower_line
                        for term in CONFIG_SEARCH_TERMS
                    ):
                        continue

                    config_hits.append(
                        {
                            "path": relative_string,
                            "line": line_number,
                            "content": line.strip()[:1000],
                        }
                    )

                    hit_count += 1
                    if hit_count >= 100:
                        break

        except OSError:
            continue


checkpoints.sort()
relevant_assets.sort(key=lambda row: row["path"])
anchor_named_files = sorted(set(anchor_named_files))
anchor_variants.sort(
    key=lambda row: (row["anchor_count"], row["path"])
)
config_hits.sort(
    key=lambda row: (row["path"], row["line"])
)

checkpoint_directories = [
    {
        "directory": directory,
        "checkpoint_count": count,
    }
    for directory, count in checkpoint_directory_counts.most_common()
]

(OUT / "all_checkpoints.txt").write_text(
    "\n".join(checkpoints) + ("\n" if checkpoints else ""),
    encoding="utf-8",
)

(OUT / "anchor_named_files.txt").write_text(
    "\n".join(anchor_named_files)
    + ("\n" if anchor_named_files else ""),
    encoding="utf-8",
)

write_csv(
    OUT / "checkpoint_directories.csv",
    checkpoint_directories,
    ["directory", "checkpoint_count"],
)

write_csv(
    OUT / "existing_result_assets.csv",
    relevant_assets,
    ["path", "extension", "size_bytes"],
)

write_csv(
    OUT / "anchor_variants_from_names.csv",
    anchor_variants,
    ["anchor_count", "path"],
)

write_csv(
    OUT / "anchor_config_hits.csv",
    config_hits,
    ["path", "line", "content"],
)

summary_lines = [
    f"Repository: {ROOT}",
    f"Total checkpoints: {len(checkpoints)}",
    f"Directories containing checkpoints: {len(checkpoint_directories)}",
    f"Relevant existing data/figure assets: {len(relevant_assets)}",
    f"Files with anchor-related names: {len(anchor_named_files)}",
    f"Explicit 0/4/8/16 variants detected in names: {len(anchor_variants)}",
    f"Anchor-related configuration lines: {len(config_hits)}",
    "",
    "Top checkpoint directories:",
]

for row in checkpoint_directories[:40]:
    summary_lines.append(
        f"  {row['checkpoint_count']:5d}  {row['directory']}"
    )

summary_lines.extend(
    [
        "",
        "Explicit anchor variants detected:",
    ]
)

if anchor_variants:
    for row in anchor_variants[:100]:
        summary_lines.append(
            f"  {row['anchor_count']:2d} anchors  {row['path']}"
        )
else:
    summary_lines.append(
        "  No explicit anchor count detected in filenames."
    )

summary = "\n".join(summary_lines) + "\n"

(OUT / "inventory_summary.txt").write_text(
    summary,
    encoding="utf-8",
)

print(summary)
print("Generated files:")
print(f"  {OUT / 'inventory_summary.txt'}")
print(f"  {OUT / 'checkpoint_directories.csv'}")
print(f"  {OUT / 'anchor_named_files.txt'}")
print(f"  {OUT / 'anchor_variants_from_names.csv'}")
print(f"  {OUT / 'anchor_config_hits.csv'}")
print(f"  {OUT / 'existing_result_assets.csv'}")
