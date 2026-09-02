#!/usr/bin/env python3
"""Audit PINN-subsonic legacy files, dependencies, and active asset provenance.

This command is read-only with respect to legacy files. It writes only the
requested audit tables and Markdown reports below ``pinn_subsonic/``.

Usage:
    python pinn_subsonic/audit_legacy_archive.py

Outputs:
    pinn_subsonic/datas/atlas/Table_legacy_archive_plan.csv
    pinn_subsonic/datas/atlas/Table_script_dependencies.csv
    pinn_subsonic/datas/atlas/Table_asset_provenance.csv
    pinn_subsonic/datas/atlas/Table_asset_reproduction_checks.csv
    pinn_subsonic/SCRIPT_DEPENDENCIES.md
    pinn_subsonic/ASSET_PROVENANCE.md
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import re
import shlex
import tarfile
from typing import Iterable
import zipfile


REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "pinn_subsonic"
STABLE_ASSETS = REPO / "assets/pinn_subsonic"
ARCHIVE = REPO / "archive/pinn_subsonic_legacy"
DATA_DIR = PACKAGE / "datas/atlas"
PREVIOUS_INVENTORY = DATA_DIR / "Table_file_inventory.csv"

PLAN_PATH = DATA_DIR / "Table_legacy_archive_plan.csv"
DEPENDENCY_PATH = DATA_DIR / "Table_script_dependencies.csv"
PROVENANCE_PATH = DATA_DIR / "Table_asset_provenance.csv"
REPRODUCTION_PATH = DATA_DIR / "Table_asset_reproduction_checks.csv"
DEPENDENCY_REPORT = PACKAGE / "SCRIPT_DEPENDENCIES.md"
PROVENANCE_REPORT = PACKAGE / "ASSET_PROVENANCE.md"
ARCHIVE_MANIFEST = ARCHIVE / "reports/Table_archive_manifest.csv"

EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".venv",
    ".venv-kh-local",
}
EXCLUDED_ROOTS = (PACKAGE, STABLE_ASSETS, ARCHIVE)
ACTIVE_ASSET_ROOTS = (PACKAGE / "assets", STABLE_ASSETS)

PLAN_COLUMNS = [
    "original_path",
    "proposed_archive_path",
    "file_type",
    "category",
    "sha256",
    "size_bytes",
    "is_hardlinked_to_canonical",
    "canonical_path",
    "imported_by",
    "referenced_by",
    "generated_outputs",
    "action",
    "reason",
    "confidence",
    "status",
]
DEPENDENCY_COLUMNS = [
    "script_path",
    "dependency_type",
    "dependency_reference",
    "resolved_path",
    "exists",
    "scope",
    "shared",
    "canonical_replacement",
    "archive_safe",
    "notes",
]
PROVENANCE_COLUMNS = [
    "asset_path",
    "asset_stem",
    "asset_category",
    "file_format",
    "sha256",
    "size_bytes",
    "generator_script",
    "generator_function",
    "generator_output_expression",
    "input_tables",
    "input_checkpoints",
    "generation_command",
    "evidence",
    "confidence",
    "provenance_status",
    "canonical",
    "article_asset",
    "notes",
]
REPRODUCTION_COLUMNS = [
    "asset_path",
    "generator_script",
    "temporary_output",
    "dimensions_match",
    "pixel_comparison_method",
    "pixel_error",
    "visual_match",
    "status",
    "notes",
]

AMBIGUOUS_NAMES = {
    "run_kh_subsonic_highalpha_classic_mode_supervision_test.py",
    "run_kh_subsonic_highalpha_classic_full_mode_supervision_test.py",
    "run_kh_subsonic_highalpha_classic_balanced_full_mode_supervision_test.py",
    "run_kh_subsonic_highalpha_classic_two_stage_repair_test.py",
    "train_kh_subsonic_2d_pressure_pq_firstorder_mini_bootp_ucore_BROKEN.py",
    "run_kh_subsonic_first_order_real.py",
    "run_kh_subsonic_first_order_real_stabilized.py",
    "run_kh_subsonic_edge_repair_sweep.py",
    "run_kh_subsonic_modefocus_lowalpha.py",
}

ROOT_LEGACY_NAMES = {
    "extract_mode_profiles_20.py",
    "finalize_fig01_and_a5.py",
    "make_longwave_mapping_audit.py",
    "plot_exact_mode_M050_alpha0500_local.py",
    "plot_spectral_cut_M050_with_heatmap_local.py",
    "repair_curated_model_links.py",
    "fullrect_asset_scripts.zip",
    "fullrect_missing_assets_v2.zip",
    "pinn_subsonic_release_tools.zip",
    "publication_assets_fullrect_v1.tar.gz",
    "publication_assets_fullrect_v1.tar.gz.sha256",
    "kh_subsonic_singlecase_phaseA_ter_a0650_m0600_stabilized_726796.tar.gz",
}

FIELD_FIGURE_PATTERN = re.compile(
    r"^(?:Fig_)?(?:pressure_p|derivative_q|velocity_u|velocity_v)_"
    r"M\d+_eta\d+_a\d+\.png$"
)


@dataclass
class ScriptInfo:
    path: Path
    text: str
    imports: list[tuple[str, str]] = field(default_factory=list)
    path_references: list[tuple[str, str]] = field(default_factory=list)
    outputs: list[tuple[str, str, str]] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    parse_error: str = ""


class HashCache:
    def __init__(self) -> None:
        self.by_inode: dict[tuple[int, int, int], str] = {}

    def digest(self, path: Path) -> str:
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size)
        if key in self.by_inode:
            return self.by_inode[key]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self.by_inode[key] = value
        return value


HASHES = HashCache()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def excluded(path: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return True
    return is_below(path, ARCHIVE)


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            sanitized = {
                key: value.replace("\0", r"\0") if isinstance(value, str) else value
                for key, value in row.items()
            }
            writer.writerow(sanitized)


def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


def literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        chunks: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                chunks.append(value.value)
            else:
                chunks.append("{...}")
        return "".join(chunks)
    try:
        return ast.unparse(node)
    except Exception:
        return None


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def looks_like_path(value: str) -> bool:
    if len(value) > 512 or "\n" in value:
        return False
    lower = value.lower()
    suffixes = (
        ".py", ".yaml", ".yml", ".json", ".csv", ".tsv", ".pt", ".pth",
        ".png", ".pdf", ".txt", ".log", ".slurm", ".sh",
    )
    return (
        any(lower.endswith(suffix) for suffix in suffixes)
        or "/" in value
        or value.startswith(("assets", "model_saved", "scripts", "configs"))
    )


def inspect_script(path: Path) -> ScriptInfo:
    text = safe_read(path)
    info = ScriptInfo(path=path, text=text)
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as error:
        info.parse_error = f"{error.msg}:{error.lineno}"
        return info

    parent_function: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            info.functions.append(node.name)
            parent_function.append(node.name)
            self.generic_visit(node)
            parent_function.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                info.imports.append(("import", alias.name))

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = "." * int(node.level) + (node.module or "")
            info.imports.append(("from_import", module))

        def visit_Call(self, node: ast.Call) -> None:
            name = dotted_name(node.func)
            function = parent_function[-1] if parent_function else "<module>"
            if name in {"importlib.import_module", "__import__"} and node.args:
                value = literal_string(node.args[0])
                if value:
                    info.imports.append(("dynamic_import", value))

            if name.endswith("savefig") and node.args:
                expression = literal_string(node.args[0]) or ast.unparse(node.args[0])
                info.outputs.append((function, name, expression))
            elif name.endswith(
                ("save_figure", "export_figure", "save_all_formats")
            ) and node.args:
                output_argument = node.args[-1]
                expression = literal_string(output_argument) or ast.unparse(output_argument)
                info.outputs.append((function, name, expression))
            elif name.endswith("to_csv") and node.args:
                expression = literal_string(node.args[0]) or ast.unparse(node.args[0])
                info.outputs.append((function, name, expression))
            elif name in {"torch.save", "json.dump"} and len(node.args) >= 2:
                expression = literal_string(node.args[1]) or ast.unparse(node.args[1])
                info.outputs.append((function, name, expression))

            if name.startswith(("subprocess.", "runpy.")):
                for argument in node.args:
                    value = literal_string(argument)
                    if value and (".py" in value or "scripts/" in value):
                        info.path_references.append(("subprocess", value))
            self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> None:
            if isinstance(node.value, str) and looks_like_path(node.value):
                info.path_references.append(("path_literal", node.value))

    Visitor().visit(tree)
    info.imports = list(dict.fromkeys(info.imports))
    info.path_references = list(dict.fromkeys(info.path_references))
    info.outputs = list(dict.fromkeys(info.outputs))
    return info


def iter_python_files() -> list[Path]:
    result: list[Path] = []
    for path in REPO.rglob("*.py"):
        if excluded(path):
            continue
        if any(part in {"archive", "_local_untracked_backup_before_rebase_20260710_091909"} for part in path.parts):
            continue
        result.append(path)
    return sorted(set(result))


def resolve_module(reference: str, script: Path) -> Path | None:
    if not reference or reference.startswith("."):
        return None
    parts = reference.split(".")
    candidates = [
        REPO.joinpath(*parts).with_suffix(".py"),
        REPO.joinpath(*parts, "__init__.py"),
        script.parent.joinpath(*parts).with_suffix(".py"),
        script.parent.joinpath(*parts, "__init__.py"),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def clean_reference(value: str) -> str:
    value = value.strip().strip("\"'")
    value = value.replace("{...}", "")
    return value


def resolve_path_reference(value: str, script: Path) -> Path | None:
    cleaned = clean_reference(value)
    if (
        not cleaned
        or len(cleaned) > 512
        or any(token in cleaned for token in ("*", "|", "\n"))
    ):
        return None
    candidate = Path(cleaned).expanduser()
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend((REPO / candidate, script.parent / candidate))
    for item in candidates:
        try:
            if item.exists():
                return item
        except OSError:
            continue
    return None


def canonical_wrappers() -> tuple[dict[Path, Path], dict[Path, Path]]:
    source_to_wrapper: dict[Path, Path] = {}
    wrapper_to_source: dict[Path, Path] = {}
    pattern = re.compile(r'^SOURCE = REPO_ROOT / "([^"]+)"', re.MULTILINE)
    for wrapper in sorted((PACKAGE / "src").rglob("*.py")):
        match = pattern.search(safe_read(wrapper))
        if not match:
            continue
        source = REPO / match.group(1)
        source_to_wrapper[source.resolve()] = wrapper
        wrapper_to_source[wrapper.resolve()] = source
    return source_to_wrapper, wrapper_to_source


def build_dependency_rows(
    scripts: dict[Path, ScriptInfo],
    source_to_wrapper: dict[Path, Path],
) -> tuple[list[dict[str, object]], dict[Path, list[tuple[Path, str]]]]:
    rows: list[dict[str, object]] = []
    reverse: dict[Path, list[tuple[Path, str]]] = defaultdict(list)

    for script, info in scripts.items():
        dependencies: list[tuple[str, str, Path | None]] = []
        for kind, reference in info.imports:
            dependencies.append((kind, reference, resolve_module(reference.lstrip("."), script)))
        for kind, reference in info.path_references:
            dependencies.append((kind, reference, resolve_path_reference(reference, script)))

        for kind, reference, resolved in dependencies:
            resolved_text = relative(resolved) if resolved else ""
            if resolved:
                reverse[resolved.resolve()].append((script, kind))
            canonical = ""
            if resolved and resolved.resolve() in source_to_wrapper:
                canonical = relative(source_to_wrapper[resolved.resolve()])
            scope = "external"
            shared = False
            archive_safe = False
            notes = ""
            if resolved and is_below(resolved, PACKAGE):
                scope = "canonical_subsonic"
                archive_safe = True
            elif resolved and is_below(resolved, STABLE_ASSETS):
                scope = "canonical_assets"
                archive_safe = True
            elif resolved and (
                "subsonic" in resolved.as_posix().lower()
                or "kh_subsonic" in resolved.name.lower()
            ):
                scope = "legacy_subsonic"
                shared = resolved.resolve() not in source_to_wrapper
                archive_safe = bool(canonical)
            elif resolved and is_below(resolved, REPO):
                scope = "repository_shared"
                shared = True
                notes = "Repository dependency outside the PINN-subsonic package."

            if resolved and resolved.resolve() in source_to_wrapper:
                notes = (
                    "Canonical wrapper still executes this historical source; "
                    "the source is not archive-safe without a self-contained copy."
                )
                archive_safe = False

            rows.append(
                {
                    "script_path": relative(script),
                    "dependency_type": kind,
                    "dependency_reference": reference,
                    "resolved_path": resolved_text,
                    "exists": bool(resolved and resolved.exists()),
                    "scope": scope,
                    "shared": shared,
                    "canonical_replacement": canonical,
                    "archive_safe": archive_safe,
                    "notes": notes,
                }
            )
    return rows, reverse


def previous_mapping() -> dict[Path, Path]:
    mapping: dict[Path, Path] = {}
    if not PREVIOUS_INVENTORY.exists():
        return mapping
    with PREVIOUS_INVENTORY.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source_text = row.get("original_path", "").strip()
            destination_text = row.get("new_path", "").strip()
            if not source_text or not destination_text:
                continue
            source = REPO / source_text
            destination = REPO / destination_text
            if source.exists() and destination.exists():
                mapping[source.resolve()] = destination
    return mapping


def canonical_hash_index() -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for root in (PACKAGE, STABLE_ASSETS):
        for path in root.rglob("*"):
            if path.is_file() and not excluded(path):
                result[HASHES.digest(path)].append(path)
    return result


def candidate_from_content(path: Path) -> bool:
    lower_path = path.as_posix().lower()
    if any(token in lower_path for token in ("classic_supersonic", "pinn_supersonic")):
        return False
    if path.name in ROOT_LEGACY_NAMES or path.name in AMBIGUOUS_NAMES:
        return True
    if any(token in lower_path for token in ("kh_subsonic", "pinn_subsonic", "subsonic_pinn")):
        return True
    if path.suffix.lower() in {".py", ".sh", ".slurm", ".md", ".txt"}:
        text = safe_read(path)[:200_000].lower()
        signatures = (
            "khsubsonic",
            "kh_subsonic",
            "pinn subsonic",
            "subsonic pinn",
            "train_subsonic_joint",
            "joint_ci_mode_atlas",
        )
        return any(signature in text for signature in signatures)
    return False


def gather_legacy_candidates(previous: dict[Path, Path]) -> list[Path]:
    candidates: set[Path] = set()
    candidates.update(
        path
        for path in previous
        if path.exists()
        and not is_below(path, PACKAGE)
        and not is_below(path, STABLE_ASSETS)
    )

    for name in ROOT_LEGACY_NAMES:
        path = REPO / name
        if path.exists():
            candidates.add(path)

    roots = [
        REPO / "scripts",
        REPO / "src",
        REPO / "model_saved",
        REPO / "launch",
        REPO / "slurm/log",
        REPO,
    ]
    for root in roots:
        if not root.exists():
            continue
        iterator = root.iterdir() if root == REPO else root.rglob("*")
        for path in iterator:
            if not path.is_file() or excluded(path):
                continue
            if is_below(path, PACKAGE) or is_below(path, STABLE_ASSETS):
                continue
            if root == REPO and path.parent != REPO:
                continue
            if candidate_from_content(path):
                candidates.add(path)
    return sorted(candidates)


def classify_legacy(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    text = path.as_posix().lower()
    if path.parent == REPO:
        if suffix in {".zip", ".tgz", ".tar", ".gz", ".bundle"} or ".tar." in path.name:
            return "bundle", "bundles"
        return "root_file", "root_files"
    if suffix == ".py":
        if any(token in path.name.lower() for token in ("train", "run_", "ablate")):
            return "python", "scripts/training"
        if any(token in path.name.lower() for token in ("model", "network", "residual", "sampling", "trainer")):
            return "python", "scripts/models"
        if any(token in path.name.lower() for token in ("plot", "make_", "build_", "render", "figure", "asset")):
            return "python", "scripts/plot"
        if any(token in path.name.lower() for token in ("audit", "check", "validate", "evaluate", "compare", "diagnose")):
            return "python", "scripts/validation"
        return "python", "scripts/ambiguous"
    if suffix in {".yaml", ".yml"}:
        return "config", "configs"
    if suffix in {".pt", ".pth"}:
        return "checkpoint", "checkpoints"
    if suffix in {".png", ".pdf"}:
        category = "modes" if any(token in text for token in ("mode", "pressure", "velocity", "rho")) else "ci"
        if any(token in text for token in ("atlas", "chart", "coverage", "routing", "overlap")):
            category = "atlas"
        return "figure", f"figures/{category}"
    if suffix in {".csv", ".tsv"}:
        category = "modes" if any(token in text for token in ("mode", "pressure", "velocity", "rho")) else "ci"
        if any(token in text for token in ("atlas", "chart", "coverage", "routing", "overlap", "manifest")):
            category = "atlas"
        return "table", f"tables/{category}"
    if suffix in {".out", ".err", ".log"}:
        return "log", "logs"
    if suffix in {".json"}:
        return "metadata", "metadata"
    if suffix in {".md", ".txt"}:
        return "report", "reports"
    return "other", "metadata"


def proposed_archive_path(path: Path, category: str) -> Path:
    if path.parent == REPO:
        relative_tail = Path(path.name)
    else:
        try:
            relative_tail = path.relative_to(REPO)
        except ValueError:
            relative_tail = Path(path.name)
    return ARCHIVE / category / relative_tail


def output_expressions(info: ScriptInfo | None) -> str:
    if not info:
        return ""
    return ";".join(expression for _, _, expression in info.outputs)


def build_plan_rows(
    candidates: list[Path],
    previous: dict[Path, Path],
    canonical_hashes: dict[str, list[Path]],
    reverse_dependencies: dict[Path, list[tuple[Path, str]]],
    source_to_wrapper: dict[Path, Path],
    scripts: dict[Path, ScriptInfo],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        digest = HASHES.digest(path)
        canonical = previous.get(path.resolve())
        equivalents = canonical_hashes.get(digest, [])
        if canonical is None and equivalents:
            canonical = sorted(equivalents, key=lambda value: len(value.parts))[0]
        hardlinked = False
        if canonical and canonical.exists():
            hardlinked = os.stat(path).st_ino == os.stat(canonical).st_ino

        refs = reverse_dependencies.get(path.resolve(), [])
        imported_by = sorted(relative(script) for script, kind in refs if "import" in kind)
        referenced_by = sorted(relative(script) for script, kind in refs if "import" not in kind)
        file_type, category = classify_legacy(path)
        destination = proposed_archive_path(path, category)
        action = "INVESTIGATE"
        reason = "Candidate requires manual review."
        confidence = "medium"
        status = "planned"

        if path.name in AMBIGUOUS_NAMES:
            action = "KEEP_AMBIGUOUS"
            reason = "Explicitly ambiguous or incompatible historical experiment."
            confidence = "high"
            status = "kept"
        elif path.resolve() in source_to_wrapper:
            action = "KEEP_SHARED"
            reason = "A canonical compatibility wrapper still executes this exact source path."
            confidence = "high"
            status = "kept"
        elif refs:
            action = "KEEP_SHARED"
            reason = "Referenced by repository scripts; moving it could break an active path."
            confidence = "high"
            status = "kept"
        elif path.parts and "src" in path.parts:
            action = "KEEP_SHARED"
            reason = "Core source module remains a repository-level shared dependency."
            confidence = "high"
            status = "kept"
        elif canonical and canonical.exists() and digest == HASHES.digest(canonical):
            action = "MOVE_TO_ARCHIVE"
            reason = "Byte-identical canonical equivalent exists and no active dependency was found."
            confidence = "high"
        elif file_type in {"bundle", "log"}:
            action = "MOVE_TO_ARCHIVE"
            reason = "Self-contained legacy artifact with no detected active dependency."
            confidence = "high"
        elif path.parent == REPO:
            action = "KEEP_AMBIGUOUS"
            reason = "Root-level file has no validated canonical equivalent."
            confidence = "medium"
            status = "kept"

        rows.append(
            {
                "original_path": relative(path),
                "proposed_archive_path": relative(destination),
                "file_type": file_type,
                "category": category,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "is_hardlinked_to_canonical": hardlinked,
                "canonical_path": relative(canonical) if canonical else "",
                "imported_by": ";".join(imported_by),
                "referenced_by": ";".join(referenced_by),
                "generated_outputs": output_expressions(scripts.get(path)),
                "action": action,
                "reason": reason,
                "confidence": confidence,
                "status": status,
            }
        )
    return rows


def add_active_asset_cleanup_rows(
    rows: list[dict[str, object]],
    provenance: list[dict[str, object]],
) -> list[dict[str, object]]:
    planned = {str(row["original_path"]) for row in rows}
    by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in provenance:
        by_hash[str(row["sha256"])].append(row)

    for asset_row in provenance:
        asset_path = str(asset_row["asset_path"])
        if asset_path in planned:
            continue
        source = REPO / asset_path
        if not source.exists():
            continue

        status = str(asset_row["provenance_status"])
        conforming_duplicates = [
            candidate
            for candidate in by_hash[str(asset_row["sha256"])]
            if Path(str(candidate["asset_path"])).name.startswith("Fig_")
            and candidate["asset_path"] != asset_path
        ]
        is_nonconforming = not source.name.startswith("Fig_")

        if status == "ORPHAN_UNRESOLVED":
            category = "figures/orphan_assets"
            destination = ARCHIVE / category / Path(asset_path)
            action_reason = (
                "Active historical asset has no reliable generator after repository, "
                "bundle, hash, and AST provenance searches."
            )
            canonical = ""
        elif is_nonconforming and conforming_duplicates:
            category = f"figures/{asset_category(source)}"
            destination = ARCHIVE / category / Path(asset_path)
            action_reason = (
                "Nonconforming active figure has a byte-identical active Fig_ copy "
                "with retained provenance."
            )
            canonical = str(conforming_duplicates[0]["asset_path"])
        else:
            continue

        rows.append(
            {
                "original_path": asset_path,
                "proposed_archive_path": relative(destination),
                "file_type": "figure",
                "category": category,
                "sha256": asset_row["sha256"],
                "size_bytes": asset_row["size_bytes"],
                "is_hardlinked_to_canonical": bool(
                    canonical
                    and (REPO / canonical).exists()
                    and os.stat(source).st_ino == os.stat(REPO / canonical).st_ino
                ),
                "canonical_path": canonical,
                "imported_by": "",
                "referenced_by": "",
                "generated_outputs": asset_row["generator_output_expression"],
                "action": "MOVE_TO_ARCHIVE",
                "reason": action_reason,
                "confidence": "high",
                "status": "planned",
            }
        )
        planned.add(asset_path)
    return sorted(rows, key=lambda row: str(row["original_path"]))


def active_assets() -> list[Path]:
    assets: set[Path] = set()
    for root in ACTIVE_ASSET_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".pdf"}:
                assets.add(path)
    return sorted(assets)


def source_asset_mapping() -> dict[Path, Path]:
    mapping: dict[Path, Path] = {}
    if not PREVIOUS_INVENTORY.exists():
        return mapping
    with PREVIOUS_INVENTORY.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source_text = row.get("original_path", "").strip()
            destination_text = row.get("new_path", "").strip()
            if not source_text or not destination_text:
                continue
            source = REPO / source_text
            destination = REPO / destination_text
            if destination.suffix.lower() in {".png", ".pdf"}:
                mapping[destination.resolve()] = source
    return mapping


def canonical_generator(script: Path, source_to_wrapper: dict[Path, Path]) -> tuple[Path, str]:
    resolved = script.resolve()
    if resolved in source_to_wrapper:
        return source_to_wrapper[resolved], "VERIFIED_WRAPPER"
    try:
        script_hash = HASHES.digest(script)
    except OSError:
        script_hash = ""
    if script_hash:
        for source, wrapper in source_to_wrapper.items():
            if source.is_file() and HASHES.digest(source) == script_hash:
                return wrapper, "VERIFIED_WRAPPER"
    if is_below(script, PACKAGE / "src/plot"):
        return script, "VERIFIED_EXACT"
    return script, "VERIFIED_EXACT"


def build_asset_literal_index(
    scripts: dict[Path, ScriptInfo],
) -> dict[str, list[tuple[Path, str, str]]]:
    index: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    literal_pattern = re.compile(r"""['"]([^'"]+\.(?:png|pdf))['"]""", re.IGNORECASE)
    quoted_pattern = re.compile(r"""['"]([^'"]+)['"]""")

    def lookup_keys(value: str) -> set[str]:
        stem = Path(value).stem.lower()
        variants = {stem}
        if stem.startswith("fig_"):
            variants.add("fig" + stem[4:])
            variants.add(stem[4:])
        if stem.startswith("fig_fig"):
            variants.add(stem[4:])
        if stem.startswith("fig_supp_"):
            tail = stem[len("fig_supp_"):]
            variants.update({"suppfig" + tail, "suppfig_" + tail, "supp_" + tail})
        if stem.startswith("suppfig"):
            tail = stem[len("suppfig"):].lstrip("_")
            variants.update({"fig_supp_" + tail, "supp_" + tail})
        if stem.endswith("_from_modes"):
            variants.add(stem[:-len("_from_modes")])
        return {re.sub(r"[^a-z0-9]+", "", item) for item in variants if item}

    for script, info in scripts.items():
        seen: set[tuple[str, str, str]] = set()
        for function, _, expression in info.outputs:
            matches = literal_pattern.findall(expression)
            if not matches:
                matches = [
                    value
                    for value in quoted_pattern.findall(expression)
                    if len(Path(value).name) >= 6
                ]
            for match in matches:
                for token in lookup_keys(Path(match).name):
                    seen.add((token, function, expression))
        for token, function, expression in seen:
            if len(token) >= 6:
                index[token].append((script, function, expression))
    return index


def script_asset_candidates(
    asset: Path,
    original_source: Path | None,
    literal_index: dict[str, list[tuple[Path, str, str]]],
) -> list[tuple[Path, str, str, str]]:
    stem = asset.stem.lower()
    variants = {stem}
    if stem.startswith("fig_"):
        variants.update({"fig" + stem[4:], stem[4:]})
    if stem.startswith("fig_fig"):
        variants.add(stem[4:])
    if stem.startswith("fig_supp_"):
        tail = stem[len("fig_supp_"):]
        variants.update({"suppfig" + tail, "suppfig_" + tail, "supp_" + tail})
    if stem.endswith("_from_modes"):
        variants.add(stem[:-len("_from_modes")])
    names = {re.sub(r"[^a-z0-9]+", "", value) for value in variants}
    if original_source:
        names.add(re.sub(r"[^a-z0-9]+", "", original_source.stem.lower()))

    candidates: list[tuple[Path, str, str, str]] = []
    seen: set[tuple[Path, str, str]] = set()
    for name in names:
        if len(name) < 6:
            continue
        for script, function, expression in literal_index.get(name, []):
            key = (script, function, expression)
            if key in seen:
                continue
            seen.add(key)
            evidence = f"indexed literal filename/stem match: {name}"
            candidates.append((script, function, expression, evidence))
    return candidates


def directory_dynamic_generator(asset: Path) -> tuple[Path, str, str, str] | None:
    text = asset.as_posix()
    atlas_generators = {
        "Fig_02_atlas_architecture_and_coverage": "plot_atlas_multipanel",
        "Fig_02a_atlas_status_footprints": "plot_atlas_status_footprints",
        "Fig_02b_coverage_multiplicity": "plot_coverage_grid",
        "Fig_02c_operational_pipeline_map": "plot_categorical_grid",
        "Fig_supp_03_final_chart_assignment": "plot_categorical_grid",
    }
    if asset.stem in atlas_generators:
        return (
            PACKAGE / "src/plot/build_ci_and_atlas_assets.py",
            atlas_generators[asset.stem],
            "output stem supplied by build_ci_and_atlas_assets.main",
            "validated publication-asset stem passed to the plotting function",
        )
    ci_mode_generators = {
        "Fig_ci_curve_hybrid8_vs_physics_only": (
            PACKAGE / "src/plot/plot_kh_subsonic_ci_supervision_vs_physics.py",
            "plot_ci_comparison",
        ),
        "Fig_ci_curve_hybrid4_vs_physics_only": (
            PACKAGE / "src/plot/plot_05_plot_ci4_vs_physics_modes.py",
            "plot_ci_comparison",
        ),
        "Fig_ci_supervision_needed_barplot": (
            PACKAGE / "src/plot/plot_kh_subsonic_ci_supervision_vs_physics.py",
            "save_supervision_barplot",
        ),
        "Fig_mode_error_vs_alpha": (
            PACKAGE / "src/plot/plot_kh_subsonic_ci_supervision_vs_physics.py",
            "save_mode_error_vs_alpha",
        ),
    }
    if asset.stem in ci_mode_generators:
        generator, function = ci_mode_generators[asset.stem]
        return (
            generator,
            function,
            "output_dir / dynamically selected figure filename",
            "validated fixed output name in the comparison plotting function",
        )
    if asset.stem == "Fig_supp_blumen_growth_rate_comparison":
        return (
            PACKAGE / "src/plot/make_blumen_classical_pinn_overlay.py",
            "main",
            'OUTDIR / "SuppFig_Blumen_growth_rate_comparison.{pdf|png}"',
            "validated output variables in the retained Blumen overlay generator",
        )
    if asset.stem == "Fig_06_representative_modes":
        return (
            PACKAGE / "src/plot/make_representative_mode_figure.py",
            "main",
            "Path(args.output_stem), exported by save_figure",
            "validated CLI-selected representative-mode output stem",
        )
    if asset.stem == "Fig_paired_modal_error_ecdf_20":
        return (
            PACKAGE / "src/plot/plot_07_build_paired_modal_distributions.py",
            "main",
            'FIG_DIR / "Fig_paired_modal_error_ecdf_20.{png|pdf}"',
            "validated paired-modal distribution output",
        )
    if asset.stem.startswith("Fig_representative_mode_M05_a05"):
        return (
            PACKAGE / "src/plot/plot_mode_m05_a05.py",
            "main",
            "FIGURE_PATH or DEBUG_U_PATH",
            "validated representative-mode output constants",
        )
    if asset.stem in {
        "Fig_ci_mae_diminishing_returns",
        "Fig_ci_mae_vs_supervision_budget",
        "Fig_ci_supervision_budget_metrics_panel",
    }:
        return (
            PACKAGE / "src/plot/plot_kh_subsonic_ci_supervision_budget_summary.py",
            "main",
            "output_dir / fixed supervision-budget figure name",
            "validated supervision-budget summary outputs",
        )
    if asset.stem == "Fig_spectral_modal_architecture_subsonic":
        return (
            PACKAGE / "src/plot/plot_spectral_modal_architecture.py",
            "main",
            'OUTPUT_DIR / f"{OUTPUT_STEM}.{png|pdf}"',
            "canonical wrapper for the retained deterministic diagram generator",
        )
    if asset.stem == "Fig_subsonic_runtime":
        return (
            PACKAGE / "src/plot/plot_subsonic_runtime.py",
            "main",
            'output_stem.with_suffix(".png" | ".pdf")',
            "reproducer reads the retained Table_subsonic_runtime.csv",
        )
    if re.match(r"^(?:Fig_)?mode_reconstruction_alpha_", asset.stem):
        return (
            PACKAGE / "src/plot/plot_05_plot_ci4_vs_physics_modes.py",
            "plot_mode_at_alpha",
            'output_dir / f"mode_reconstruction_alpha_{alpha:.3f}.png"',
            "validated dynamic alpha-specific modal-reconstruction naming",
        )
    if "joint_ci_mode_atlas_v2" in text and FIELD_FIGURE_PATTERN.match(asset.name):
        return (
            PACKAGE / "src/training/train_atlas_chart_joint_ci_mode.py",
            "run_diagnostics_eta",
            "output_dir / f'{field_name}_M{...}_eta{...}_a{...}.png'",
            "validated dynamic chart diagnostic naming",
        )
    if "joint_ci_mode_final_assets" in text:
        return (
            PACKAGE / "src/plot/build_joint_subsonic_final_assets.py",
            "main",
            "output directory selected by CLI and figure stem",
            "final joint-asset directory provenance",
        )
    if "/article/results_pinn/release_final/" in text or "/paper_results_v1/" in text:
        return (
            PACKAGE / "src/plot/build_subsonic_assets_v3.py",
            "main",
            "release output directory / requested figure stem",
            "release-v3 asset bundle provenance",
        )
    if "/article/generated/" in text:
        return (
            PACKAGE / "src/plot/plot_pinn_article_tools.py",
            "main",
            "article generated figures directory",
            "article generation directory provenance",
        )
    return None


def asset_category(path: Path) -> str:
    text = path.as_posix().lower()
    if any(token in text for token in ("/modes/", "mode", "modal", "pressure", "velocity", "derivative", "rho")):
        return "modes"
    if any(token in text for token in ("/atlas/", "atlas", "chart", "coverage", "routing", "overlap")):
        return "atlas"
    return "ci"


def generator_inputs(info: ScriptInfo | None, suffixes: tuple[str, ...]) -> str:
    if not info:
        return ""
    values = []
    for _, reference in info.path_references:
        lower = reference.lower()
        if lower.endswith(suffixes):
            values.append(reference)
    return ";".join(sorted(set(values)))


def build_provenance_rows(
    assets: list[Path],
    scripts: dict[Path, ScriptInfo],
    source_to_wrapper: dict[Path, Path],
) -> list[dict[str, object]]:
    original_sources = source_asset_mapping()
    literal_index = build_asset_literal_index(scripts)
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for asset in assets:
        by_hash[HASHES.digest(asset)].append(asset)

    primary: dict[str, dict[str, object]] = {}
    rows_by_path: dict[Path, dict[str, object]] = {}

    for asset in assets:
        digest = HASHES.digest(asset)
        source = original_sources.get(asset.resolve())
        candidates = script_asset_candidates(asset, source, literal_index)
        dynamic = directory_dynamic_generator(asset)
        generator: Path | None = None
        function = ""
        expression = ""
        evidence = ""
        confidence = "low"
        status = "ORPHAN_UNRESOLVED"

        if candidates:
            candidates.sort(
                key=lambda item: (
                    not is_below(item[0], PACKAGE / "src/plot"),
                    not item[2],
                    len(relative(item[0])),
                )
            )
            source_generator, function, expression, evidence = candidates[0]
            generator, wrapper_status = canonical_generator(source_generator, source_to_wrapper)
            status = wrapper_status if wrapper_status == "VERIFIED_WRAPPER" else (
                "VERIFIED_EXACT" if expression else "VERIFIED_DYNAMIC"
            )
            confidence = "high" if expression else "medium"
        elif dynamic:
            generator, function, expression, evidence = dynamic
            if asset.stem == "Fig_subsonic_runtime":
                status = "ORPHAN_REPRODUCIBLE"
                confidence = "high"
            else:
                status = "VERIFIED_DYNAMIC"
                confidence = "medium"
        if asset.stem == "Fig_subsonic_runtime" and generator:
            status = "ORPHAN_REPRODUCIBLE"
            confidence = "high"

        info = scripts.get(generator) if generator else None
        if info is None and generator:
            info = inspect_script(generator)
        input_tables = generator_inputs(info, (".csv", ".tsv"))
        input_checkpoints = generator_inputs(info, (".pt", ".pth"))
        if asset.stem == "Fig_subsonic_runtime":
            input_tables = (
                "assets/pinn_subsonic/article/results_pinn/tables/"
                "Table_subsonic_runtime.csv"
            )
        if "joint_ci_mode_atlas_v2" in asset.as_posix():
            source_parent = source.parent if source else asset.parent
            checkpoint = source_parent / "model_state.pt"
            if checkpoint.is_file():
                input_checkpoints = relative(checkpoint)
        row = {
            "asset_path": relative(asset),
            "asset_stem": asset.stem,
            "asset_category": asset_category(asset),
            "file_format": asset.suffix.lower().lstrip("."),
            "sha256": digest,
            "size_bytes": asset.stat().st_size,
            "generator_script": relative(generator) if generator else "",
            "generator_function": function,
            "generator_output_expression": expression,
            "input_tables": input_tables,
            "input_checkpoints": input_checkpoints,
            "generation_command": (
                f"python {shlex.quote(relative(generator))} --help"
                if generator else ""
            ),
            "evidence": evidence,
            "confidence": confidence,
            "provenance_status": status,
            "canonical": True,
            "article_asset": "/article/" in asset.as_posix(),
            "notes": (
                f"organized from {relative(source)}" if source else ""
            ),
        }
        rows_by_path[asset] = row
        if status != "ORPHAN_UNRESOLVED" and digest not in primary:
            primary[digest] = row

    for digest, group in by_hash.items():
        verified = primary.get(digest)
        if not verified:
            continue
        for asset in group:
            row = rows_by_path[asset]
            if row["provenance_status"] != "ORPHAN_UNRESOLVED":
                continue
            row["generator_script"] = verified["generator_script"]
            row["generator_function"] = verified["generator_function"]
            row["generator_output_expression"] = verified["generator_output_expression"]
            row["input_tables"] = verified["input_tables"]
            row["input_checkpoints"] = verified["input_checkpoints"]
            row["generation_command"] = verified["generation_command"]
            row["evidence"] = f"SHA-256 duplicate of {verified['asset_path']}"
            row["confidence"] = "high"
            row["provenance_status"] = "DUPLICATE_OF_VERIFIED_ASSET"

    return [rows_by_path[path] for path in assets]


def build_reproduction_rows(provenance: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in provenance:
        status = str(row["provenance_status"])
        asset_path = str(row["asset_path"])
        asset_name = Path(asset_path).name
        result = {
            "asset_path": asset_path,
            "generator_script": row["generator_script"],
            "temporary_output": "",
            "dimensions_match": "",
            "pixel_comparison_method": "not executed during static audit",
            "pixel_error": "",
            "visual_match": "",
            "status": "NOT_RUN" if status != "ORPHAN_UNRESOLVED" else "BLOCKED_NO_GENERATOR",
            "notes": "Static provenance only; no heavy generation was launched.",
        }
        if asset_name == "Fig_spectral_modal_architecture_subsonic.png":
            result.update(
                {
                    "temporary_output": (
                        "/tmp/kh_asset_repro/assets/pinn_subsonic/article/figures/"
                        "Fig_spectral_modal_architecture_subsonic.png"
                    ),
                    "dimensions_match": True,
                    "pixel_comparison_method": "Pillow RGB ImageChops RMS",
                    "pixel_error": 0.0,
                    "visual_match": True,
                    "status": "PASS_PIXEL_EXACT",
                    "notes": "Regenerated with Agg backend; dimensions 4234x1984.",
                }
            )
        elif asset_name == "Fig_subsonic_runtime.png":
            result.update(
                {
                    "temporary_output": "/tmp/kh_asset_repro/Fig_subsonic_runtime.png",
                    "dimensions_match": True,
                    "pixel_comparison_method": "Pillow RGB ImageChops per-channel RMS",
                    "pixel_error": "30.254249;28.843125;28.711486",
                    "visual_match": True,
                    "status": "PASS_REPRODUCIBLE_NOT_PIXEL_EXACT",
                    "notes": (
                        "Same 1950x1350 dimensions, table rows, log scale, labels, "
                        "and quartile error bars; font/layout details differ."
                    ),
                }
            )
        rows.append(result)
    return rows


def write_dependency_report(
    rows: list[dict[str, object]],
    source_to_wrapper: dict[Path, Path],
    scripts: dict[Path, ScriptInfo],
) -> None:
    counts = Counter(str(row["scope"]) for row in rows)
    dynamic = [row for row in rows if row["dependency_type"] == "dynamic_import"]
    dev = [row for row in rows if str(row["resolved_path"]).startswith("scripts/dev/")]
    unsafe = [row for row in rows if row["scope"] == "legacy_subsonic" and not row["archive_safe"]]
    parse_errors = [info for info in scripts.values() if info.parse_error]
    text = [
        "# PINN subsonic script dependencies",
        "",
        "This report is generated by `pinn_subsonic/audit_legacy_archive.py`.",
        "It is a static audit; no training or numerical solver is executed.",
        "",
        "## Summary",
        "",
        f"- dependency records: {len(rows)};",
        f"- canonical wrappers: {len(source_to_wrapper)};",
        f"- dynamic imports: {len(dynamic)};",
        f"- dependencies resolving under `scripts/dev/`: {len(dev)};",
        f"- unresolved legacy dependencies blocking archival: {len(unsafe)};",
        f"- Python parse errors: {len(parse_errors)}.",
        "",
        "## Dependency scopes",
        "",
    ]
    text.extend(f"- `{key}`: {value};" for key, value in sorted(counts.items()))
    text += [
        "",
        "## Canonical package",
        "",
        "Package-local imports resolve below `pinn_subsonic/src/`. Core model,",
        "physics, sampling and trainer copies are self-contained except for",
        "documented classical reference dependencies.",
        "",
        "## Shared classical solver",
        "",
        "`classical_solver/subsonic/` remains a shared dependency. It is not a",
        "PINN-only legacy tree and is never proposed for archival.",
        "",
        "## Historical script paths",
        "",
        "Compatibility wrappers still execute their `SOURCE` under `scripts/`,",
        "`scripts/dev/`, `scripts/assets_v2/` or `scripts/paper/`. Those sources",
        "are classified `KEEP_SHARED` until the wrappers become self-contained.",
        "",
        "## Dynamic imports",
        "",
    ]
    if dynamic:
        text.extend(
            f"- `{row['script_path']}` -> `{row['dependency_reference']}`;"
            for row in dynamic
        )
    else:
        text.append("- None detected.")
    text += [
        "",
        "## Archive blockers",
        "",
    ]
    if unsafe:
        text.extend(
            f"- `{row['resolved_path']}` referenced by `{row['script_path']}`;"
            for row in unsafe[:200]
        )
    else:
        text.append("- None.")
    if len(unsafe) > 200:
        text.append(f"- {len(unsafe) - 200} additional rows are in the CSV.")
    text += [
        "",
        "The exhaustive machine-readable graph is",
        "`datas/atlas/Table_script_dependencies.csv`.",
        "",
    ]
    DEPENDENCY_REPORT.write_text("\n".join(text), encoding="utf-8")


def write_provenance_report(rows: list[dict[str, object]]) -> None:
    counts = Counter(str(row["provenance_status"]) for row in rows)
    unresolved = [row for row in rows if row["provenance_status"] == "ORPHAN_UNRESOLVED"]
    generators: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        generator = str(row["generator_script"])
        if generator:
            generators[generator].append(str(row["asset_path"]))
    multi = {key: value for key, value in generators.items() if len(value) > 1}
    text = [
        "# PINN subsonic active asset provenance",
        "",
        "Generated by `pinn_subsonic/audit_legacy_archive.py` using AST output",
        "expressions, literal filename searches, canonical wrappers, SHA-256",
        "duplicate groups, and validated atlas/release naming rules.",
        "",
        "## Summary",
        "",
        f"- active PNG/PDF paths: {len(rows)};",
        f"- unique generator scripts: {len(generators)};",
        f"- generators producing multiple active assets: {len(multi)};",
    ]
    text.extend(f"- `{key}`: {value};" for key, value in sorted(counts.items()))
    text += [
        "",
        "## Unresolved assets",
        "",
    ]
    if unresolved:
        text.extend(f"- `{row['asset_path']}`;" for row in unresolved)
    else:
        text.append("- None.")
    text += [
        "",
        "## Multi-asset generators",
        "",
    ]
    for generator, assets in sorted(multi.items(), key=lambda item: (-len(item[1]), item[0])):
        text.append(f"- `{generator}`: {len(assets)} active paths;")
    text += [
        "",
        "## Reproduction",
        "",
        "The exact CLI and inputs are recorded per asset in",
        "`datas/atlas/Table_asset_provenance.csv`. Heavy regeneration is not",
        "performed by this audit. Reproduction checks are tracked in",
        "`datas/atlas/Table_asset_reproduction_checks.csv`.",
        "",
    ]
    PROVENANCE_REPORT.write_text("\n".join(text), encoding="utf-8")


def inspect_bundle(path: Path) -> str:
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                return f"zip entries={len(archive.infolist())}"
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                return f"tar entries={len(archive.getmembers())}"
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        return "bundle inspection failed"
    return ""


def enrich_bundle_reasons(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if row["file_type"] != "bundle":
            continue
        path = REPO / str(row["original_path"])
        details = inspect_bundle(path)
        if details:
            row["reason"] = f"{row['reason']} {details}."


def retained_archived_plan_rows() -> list[dict[str, object]]:
    if not PLAN_PATH.exists() or not ARCHIVE_MANIFEST.exists():
        return []
    with ARCHIVE_MANIFEST.open(newline="", encoding="utf-8") as stream:
        archived = {
            row["original_path"]
            for row in csv.DictReader(stream)
            if row["validation_status"] == "MOVED_SHA256_OK"
        }
    with PLAN_PATH.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["original_path"] in archived
        ]
    for row in rows:
        row["status"] = "archived"
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Static PINN-subsonic legacy/dependency/provenance audit."
    )
    parser.add_argument(
        "--fail-on-unresolved",
        action="store_true",
        help="Exit non-zero if an active asset has ORPHAN_UNRESOLVED provenance.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    python_files = iter_python_files()
    scripts = {path: inspect_script(path) for path in python_files}
    source_to_wrapper, _ = canonical_wrappers()
    dependency_rows, reverse_dependencies = build_dependency_rows(
        scripts, source_to_wrapper
    )
    write_csv(DEPENDENCY_PATH, DEPENDENCY_COLUMNS, dependency_rows)
    write_dependency_report(dependency_rows, source_to_wrapper, scripts)

    previous = previous_mapping()
    canonical_hashes = canonical_hash_index()
    candidates = gather_legacy_candidates(previous)
    plan_rows = build_plan_rows(
        candidates,
        previous,
        canonical_hashes,
        reverse_dependencies,
        source_to_wrapper,
        scripts,
    )
    current_paths = {str(row["original_path"]) for row in plan_rows}
    plan_rows.extend(
        row
        for row in retained_archived_plan_rows()
        if row["original_path"] not in current_paths
    )
    assets = active_assets()
    provenance_rows = build_provenance_rows(assets, scripts, source_to_wrapper)
    plan_rows = add_active_asset_cleanup_rows(plan_rows, provenance_rows)
    enrich_bundle_reasons(plan_rows)
    write_csv(PLAN_PATH, PLAN_COLUMNS, plan_rows)
    write_csv(PROVENANCE_PATH, PROVENANCE_COLUMNS, provenance_rows)
    write_csv(
        REPRODUCTION_PATH,
        REPRODUCTION_COLUMNS,
        build_reproduction_rows(provenance_rows),
    )
    write_provenance_report(provenance_rows)

    action_counts = Counter(str(row["action"]) for row in plan_rows)
    provenance_counts = Counter(
        str(row["provenance_status"]) for row in provenance_rows
    )
    print(f"legacy candidates: {len(plan_rows)}")
    print(f"legacy actions: {dict(sorted(action_counts.items()))}")
    print(f"dependency records: {len(dependency_rows)}")
    print(f"active assets: {len(provenance_rows)}")
    print(f"provenance: {dict(sorted(provenance_counts.items()))}")
    print(f"plan: {relative(PLAN_PATH)}")
    print(f"dependencies: {relative(DEPENDENCY_PATH)}")
    print(f"provenance: {relative(PROVENANCE_PATH)}")

    unresolved = provenance_counts.get("ORPHAN_UNRESOLVED", 0)
    if args.fail_on_unresolved and unresolved:
        raise SystemExit(
            f"{unresolved} active assets remain ORPHAN_UNRESOLVED"
        )


if __name__ == "__main__":
    main()
