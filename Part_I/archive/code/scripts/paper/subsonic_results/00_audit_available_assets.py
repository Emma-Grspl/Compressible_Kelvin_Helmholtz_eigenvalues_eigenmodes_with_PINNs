#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path.cwd().resolve()
OUT_DIR = ROOT / "assets/pinn_subsonic/paper_results_v1/data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_PARTS = {
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

RELEVANT_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".csv",
    ".tsv",
    ".parquet",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".txt",
    ".log",
    ".out",
    ".err",
    ".slurm",
    ".py",
    ".sh",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
}

TEXT_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".txt",
    ".log",
    ".out",
    ".err",
    ".slurm",
    ".py",
    ".sh",
}

KEYWORDS = (
    "anchor",
    "supervis",
    "riccati",
    "single_case",
    "single-case",
    "fixed_mach",
    "fixed-mach",
    "atlas",
    "modal",
    "eigenmode",
    "shoot",
    "classical",
    "gep",
    "neutral",
    "overlap",
    "runtime",
    "benchmark",
    "audit",
    "ci_",
    "growth_rate",
)

ANCHOR_CONTEXT_PATTERN = re.compile(
    r"(?i).{0,140}"
    r"(?:"
    r"n[_-]?anchors?"
    r"|num[_-]?anchors?"
    r"|anchor[_-]?count"
    r"|anchor[_-]?alpha"
    r"|reference[_-]?alpha"
    r"|ci[_-]?supervision"
    r"|w[_-]?ci[_-]?supervision"
    r"|spectral[_-]?anchor"
    r")"
    r".{0,220}"
)

ANCHOR_COUNT_PATTERNS = (
    re.compile(
        r"(?i)(?:n[_-]?anchors?|num[_-]?anchors?|anchor[_-]?count)"
        r"\s*[:=]\s*(0|4|8|16)\b"
    ),
    re.compile(r"(?i)\b(0|4|8|16)[_-]?(?:anchors?|anchorpoints?|pts?)\b"),
    re.compile(r"(?i)\b(?:anchors?|anchorpoints?)[_-]?(0|4|8|16)\b"),
)


def excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def category_for(extension: str) -> str:
    if extension in {".pt", ".pth", ".ckpt"}:
        return "checkpoint"
    if extension in {".csv", ".tsv", ".parquet"}:
        return "tabular_data"
    if extension in {".npy", ".npz", ".h5", ".hdf5"}:
        return "array_data"
    if extension in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return "figure_or_document"
    if extension in {".json", ".yaml", ".yml", ".toml", ".ini"}:
        return "configuration"
    if extension in {".py", ".sh", ".slurm"}:
        return "code"
    if extension in {".txt", ".log", ".out", ".err"}:
        return "log_or_text"
    return "other"


def safe_read_text(path: Path, max_bytes: int = 4_000_000) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


manifest_rows: list[dict[str, object]] = []
anchor_hit_rows: list[dict[str, object]] = []
anchor_variant_rows: list[dict[str, object]] = []

for path in ROOT.rglob("*"):
    if not path.is_file() or excluded(path):
        continue

    extension = path.suffix.lower()
    if extension not in RELEVANT_EXTENSIONS:
        continue

    rel_path = path.relative_to(ROOT)
    path_text = str(rel_path)
    lower_path = path_text.lower()

    try:
        stat = path.stat()
    except OSError:
        continue

    path_keywords = sorted({kw for kw in KEYWORDS if kw in lower_path})
    text = ""
    content_keywords: list[str] = []

    if extension in TEXT_EXTENSIONS:
        text = safe_read_text(path)
        lower_text = text.lower()
        content_keywords = sorted({kw for kw in KEYWORDS if kw in lower_text})

        for index, match in enumerate(ANCHOR_CONTEXT_PATTERN.finditer(text)):
            if index >= 20:
                break
            anchor_hit_rows.append(
                {
                    "path": path_text,
                    "extension": extension,
                    "snippet": " ".join(match.group(0).split()),
                }
            )

    combined_text = f"{path_text}\n{text}"

    detected_counts: set[int] = set()
    for pattern in ANCHOR_COUNT_PATTERNS:
        detected_counts.update(int(value) for value in pattern.findall(combined_text))

    for count in sorted(detected_counts):
        anchor_variant_rows.append(
            {
                "anchor_count": count,
                "path": path_text,
                "extension": extension,
                "source": "filename_or_text",
            }
        )

    manifest_rows.append(
        {
            "path": path_text,
            "category": category_for(extension),
            "extension": extension,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime
            ).astimezone().isoformat(timespec="seconds"),
            "path_keywords": ";".join(path_keywords),
            "content_keywords": ";".join(content_keywords),
        }
    )

manifest_rows.sort(key=lambda row: (str(row["category"]), str(row["path"])))
anchor_hit_rows.sort(key=lambda row: str(row["path"]))
anchor_variant_rows.sort(
    key=lambda row: (int(row["anchor_count"]), str(row["path"]))
)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


write_csv(
    OUT_DIR / "available_assets_manifest.csv",
    manifest_rows,
    [
        "path",
        "category",
        "extension",
        "size_bytes",
        "modified_at",
        "path_keywords",
        "content_keywords",
    ],
)

write_csv(
    OUT_DIR / "anchor_search_hits.csv",
    anchor_hit_rows,
    ["path", "extension", "snippet"],
)

write_csv(
    OUT_DIR / "anchor_variant_candidates.csv",
    anchor_variant_rows,
    ["anchor_count", "path", "extension", "source"],
)

category_counts = Counter(str(row["category"]) for row in manifest_rows)
keyword_counts = Counter()

for row in manifest_rows:
    terms = set(
        filter(
            None,
            (
                str(row["path_keywords"]).split(";")
                + str(row["content_keywords"]).split(";")
            ),
        )
    )
    keyword_counts.update(terms)

variant_counts = Counter(
    int(row["anchor_count"]) for row in anchor_variant_rows
)

summary_lines = [
    f"Repository: {ROOT}",
    f"Relevant files: {len(manifest_rows)}",
    "",
    "Files by category:",
]

for name, count in sorted(category_counts.items()):
    summary_lines.append(f"  {name}: {count}")

summary_lines.extend(["", "Keyword occurrence by file:"])

for name, count in sorted(keyword_counts.items(), key=lambda item: (-item[1], item[0])):
    summary_lines.append(f"  {name}: {count}")

summary_lines.extend(["", "Detected anchor-count candidates:"])

for count in (0, 4, 8, 16):
    summary_lines.append(f"  {count} anchors: {variant_counts.get(count, 0)} candidate file(s)")

if anchor_variant_rows:
    summary_lines.extend(["", "Candidate files:"])
    for row in anchor_variant_rows:
        summary_lines.append(
            f"  [{row['anchor_count']}] {row['path']}"
        )
else:
    summary_lines.extend(
        [
            "",
            "No explicit 0/4/8/16 anchor count was detected in filenames or text files.",
            "Checkpoint metadata inspection is therefore required.",
        ]
    )

summary_path = OUT_DIR / "audit_summary.txt"
summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print("\n".join(summary_lines))
print("")
print(f"Manifest: {OUT_DIR / 'available_assets_manifest.csv'}")
print(f"Anchor hits: {OUT_DIR / 'anchor_search_hits.csv'}")
print(f"Anchor candidates: {OUT_DIR / 'anchor_variant_candidates.csv'}")
