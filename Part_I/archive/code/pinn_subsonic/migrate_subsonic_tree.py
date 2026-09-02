#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
import re
import shutil
import unicodedata


REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "pinn_subsonic"
SOURCE_ASSETS = REPO / "assets/pinn_subsonic"
INVENTORY = PACKAGE / "datas/atlas/Table_file_inventory.csv"

ROWS: list[dict[str, object]] = []
CONFLICTS: list[str] = []

CORE_COPIES = {
    "src/models/kh_subsonic_pinn.py": "pinn_subsonic/src/models/model_kh_subsonic_pinn.py",
    "src/physics/kh_subsonic_residual.py": "pinn_subsonic/src/models/model_physics_residual.py",
    "src/data/kh_subsonic_sampling.py": "pinn_subsonic/src/training/sampling_subsonic.py",
    "src/training/kh_subsonic_trainer.py": "pinn_subsonic/src/training/training_core_fixed_mach.py",
    "src/training/kh_subsonic_trainer_2d.py": "pinn_subsonic/src/training/training_core_parametric.py",
}

TRAINING_WRAPPERS = {
    "scripts/train_kh_subsonic_pinn.py": "train_fixed_mach_pinn.py",
    "scripts/train_kh_subsonic_pinn_2d.py": "train_parametric_alpha_mach_pinn.py",
    "scripts/train_kh_subsonic_ci_stage0_anchor_lock.py": "train_parametric_ci_stage0_anchor_lock.py",
    "scripts/train_kh_subsonic_2d_pressure_pq_firstorder_mini.py": "train_parametric_pressure_pq_firstorder.py",
    "scripts/train_kh_subsonic_2d_pressure_pq_firstorder_mini_bootp.py": "train_parametric_pressure_pq_bootstrap.py",
    "scripts/train_kh_subsonic_2d_pressure_pq_firstorder_mini_bootp_ucore_launcher.py": "train_parametric_pressure_pq_velocity_core.py",
    "scripts/train_kh_subsonic_2d_pressure_pq_firstorder_mini_bootp_velcore_launcher.py": "train_parametric_pressure_pq_full_velocity_core.py",
    "scripts/train_kh_subsonic_singlecase_pressure_pq_firstorder.py": "train_single_case_pressure_pq_firstorder.py",
    "scripts/train_subsonic_mini2d_pqAB_regularized_primitive.py": "train_parametric_pqab_regularized_primitives.py",
    "scripts/train_subsonic_mini2d_pqAB_regularized_primitive_core.py": "train_parametric_pqab_regularized_core.py",
    "scripts/train_subsonic_mini2d_pq_only_hard_reconstruct_uv.py": "train_parametric_pq_hard_velocity_reconstruction.py",
    "scripts/train_subsonic_mini2d_pq_only_hard_reconstruct_uv_core.py": "train_parametric_pq_hard_velocity_core.py",
    "scripts/dev/train_subsonic_joint_spectral_modal_chart.py": "train_atlas_chart_joint_ci_mode.py",
    "scripts/dev/train_subsonic_seedGEP_pq2d_continuous_M_alpha.py": "train_atlas_modal_seeded_gep_pq.py",
    "scripts/dev/train_subsonic_seedGEP_pq2d_continuous_M_alpha_etaaware.py": "train_atlas_modal_seeded_gep_etaaware.py",
    "scripts/dev/train_subsonic_seedGEP_pQscaled2d_continuous_M_alpha.py": "train_atlas_modal_seeded_gep_qscaled.py",
    "scripts/run_kh_subsonic_hybrid_alpha_sweep.py": "train_fixed_mach_hybrid_alpha_sweep.py",
    "scripts/run_kh_subsonic_ci_supervision_vs_physics.py": "train_fixed_mach_ci_supervision_comparison.py",
    "scripts/ablate_kh_subsonic_ci_supervision_budget.py": "train_fixed_mach_ci_anchor_budget_ablation.py",
    "scripts/run_kh_subsonic_pinn_2d_ci_mode_campaign.py": "train_parametric_ci_mode_campaign.py",
    "scripts/run_kh_subsonic_pinn_2d_band_M05_M07.py": "train_parametric_mach_band_M050_M070.py",
    "scripts/run_kh_subsonic_riccati_core1d.py": "train_fixed_mach_riccati_core.py",
    "scripts/run_kh_subsonic_riccati_core1d_windowed_sparse.py": "train_fixed_mach_riccati_sparse_windows.py",
}

PLOT_SOURCES = [
    "scripts/plot_kh_subsonic_ci_error_heatmap.py",
    "scripts/plot_kh_subsonic_ci_supervision_budget_summary.py",
    "scripts/plot_kh_subsonic_ci_supervision_vs_physics.py",
    "scripts/plot_kh_subsonic_fixed_mach_classic_vs_pinn_modes.py",
    "scripts/plot_kh_subsonic_fixed_mach_mode_field_error_heatmaps.py",
    "scripts/plot_kh_subsonic_mode_error_heatmaps_2d.py",
    "scripts/plot_kh_subsonic_mode_error_vs_alpha.py",
    "scripts/plot_kh_subsonic_pinn_results.py",
    "scripts/plot_kh_subsonic_pinn_results_2d.py",
    "scripts/plot_kh_subsonic_pinn_single_mode_like_thesis.py",
    "scripts/compare_kh_subsonic_ci_two_runs.py",
    "scripts/compare_kh_subsonic_classic_vs_pinn_mode.py",
    "scripts/compare_kh_subsonic_fixed_mach_modal_candidates.py",
    "scripts/compare_kh_subsonic_mode_amp_phase.py",
    "scripts/compare_kh_subsonic_mode_two_runs_amp_phase.py",
    "scripts/compare_kh_subsonic_pressure_mode.py",
    "scripts/evaluate_kh_subsonic_pinn_2d_ci_modes.py",
    "scripts/dev/build_fullrect_publication_assets.py",
    "scripts/dev/build_joint_subsonic_final_assets.py",
    "scripts/dev/build_subsonic_article_comparison_local.py",
    "scripts/dev/build_subsonic_assets_v3.py",
    "scripts/dev/joint_pinn_global_validation.py",
    "scripts/dev/plot_ci_isolines_local_fix.py",
    "scripts/dev/plot_exact_mode_M050_alpha0500_local.py",
    "scripts/dev/plot_mode_overlay_article.py",
    "scripts/dev/plot_spectral_cut_M050_with_heatmap_local.py",
]

PLOT_SOURCES += [
    str(path.relative_to(REPO))
    for root in (
        REPO / "scripts/assets_v2",
        REPO / "scripts/paper/subsonic_results",
    )
    for path in sorted(root.glob("*.py"))
    if path.name not in {"00_fast_repo_inventory.py", "00_audit_available_assets.py"}
]

AMBIGUOUS = [
    "scripts/run_kh_subsonic_highalpha_classic_mode_supervision_test.py",
    "scripts/run_kh_subsonic_highalpha_classic_full_mode_supervision_test.py",
    "scripts/run_kh_subsonic_highalpha_classic_balanced_full_mode_supervision_test.py",
    "scripts/run_kh_subsonic_highalpha_classic_two_stage_repair_test.py",
    "scripts/train_kh_subsonic_2d_pressure_pq_firstorder_mini_bootp_ucore_BROKEN.py",
    "scripts/run_kh_subsonic_first_order_real.py",
    "scripts/run_kh_subsonic_first_order_real_stabilized.py",
    "scripts/run_kh_subsonic_edge_repair_sweep.py",
    "scripts/run_kh_subsonic_modefocus_lowalpha.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ascii_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unnamed"


def record(
    source: Path,
    destination: Path | None,
    category: str,
    action: str,
    reason: str,
    status: str,
) -> None:
    ROWS.append(
        {
            "original_path": str(source.relative_to(REPO)) if source.is_relative_to(REPO) else str(source),
            "new_path": (
                str(destination.relative_to(REPO))
                if destination is not None and destination.is_relative_to(REPO)
                else str(destination or "")
            ),
            "category": category,
            "action": action,
            "sha256": sha256(source) if source.is_file() else "",
            "size_bytes": source.stat().st_size if source.is_file() else 0,
            "reason": reason,
            "status": status,
        }
    )


def conflict_destination(destination: Path, source: Path) -> Path:
    source_label = ascii_name(source.parent.name.lower())
    candidate = destination.with_name(f"{destination.stem}_from_{source_label}{destination.suffix}")
    if candidate.exists() and sha256(candidate) != sha256(source):
        candidate = destination.with_name(
            f"{destination.stem}_from_{source_label}_{sha256(source)[:8]}{destination.suffix}"
        )
    return candidate


def safe_link_or_copy(source: Path, destination: Path, category: str, reason: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) == sha256(destination):
            record(source, destination, category, "existing_identical", reason, "ok")
            return destination
        alternate = conflict_destination(destination, source)
        CONFLICTS.append(f"{destination.relative_to(REPO)} <- {source.relative_to(REPO)}")
        destination = alternate
        if destination.exists() and sha256(source) == sha256(destination):
            record(source, destination, category, "existing_identical_conflict_name", reason, "ok")
            return destination
    try:
        os.link(source, destination)
        action = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        action = "copy"
    record(source, destination, category, action, reason, "ok")
    return destination


def safe_write(destination: Path, content: str, category: str, reason: str, source: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()
    if destination.exists():
        if sha256(destination) == expected:
            record(source, destination, category, "existing_identical_generated", reason, "ok")
            return
        CONFLICTS.append(f"{destination.relative_to(REPO)} generated from {source.relative_to(REPO)}")
        record(source, destination, category, "conflict_not_overwritten", reason, "review")
        return
    destination.write_bytes(encoded)
    record(source, destination, category, "generated_wrapper", reason, "ok")


def wrapper_content(source: Path) -> str:
    relative = source.relative_to(REPO).as_posix()
    return f'''#!/usr/bin/env python3
"""Compatibility entry point for `{relative}`.

Scientific logic remains in the original script so both entry points stay aligned.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPO_ROOT / "{relative}"


def main() -> None:
    repository_directory = str(REPO_ROOT)
    if repository_directory not in sys.path:
        sys.path.insert(0, repository_directory)
    source_directory = str(SOURCE.parent)
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)
    runpy.run_path(str(SOURCE), run_name="__main__")


if __name__ == "__main__":
    main()
'''


def copy_core_modules() -> None:
    replacements = {
        "from src.data.kh_subsonic_sampling import": "from pinn_subsonic.src.training.sampling_subsonic import",
        "from src.models.kh_subsonic_pinn import": "from pinn_subsonic.src.models.model_kh_subsonic_pinn import",
        "from src.physics.kh_subsonic_residual import": "from pinn_subsonic.src.models.model_physics_residual import",
    }
    for source_value, destination_value in CORE_COPIES.items():
        source = REPO / source_value
        destination = REPO / destination_value
        content = source.read_text(encoding="utf-8")
        for old, new in replacements.items():
            content = content.replace(old, new)
        safe_write(
            destination,
            content,
            "MODEL_SUBSONIC" if "/models/" in destination_value else "TRAINING_SUBSONIC",
            "Core PINN subsonic module copied with package-local imports only.",
            source,
        )


def create_code_entrypoints() -> None:
    for source_value, filename in TRAINING_WRAPPERS.items():
        source = REPO / source_value
        if not source.exists():
            record(source, None, "TRAINING_SUBSONIC", "missing_source", "Expected training entry point.", "missing")
            continue
        safe_write(
            PACKAGE / "src/training" / filename,
            wrapper_content(source),
            "TRAINING_SUBSONIC",
            "Non-destructive wrapper preserving the original training implementation and CLI.",
            source,
        )
    used_names: set[str] = set()
    for source_value in PLOT_SOURCES:
        source = REPO / source_value
        if not source.exists():
            continue
        stem = ascii_name(source.stem.lower())
        if not stem.startswith(("plot_", "build_", "make_", "compare_", "evaluate_", "check_", "remake_", "select_", "finalize_", "extract_", "inventory_")):
            stem = f"plot_{stem}"
        filename = f"{stem}.py"
        if filename in used_names:
            filename = f"{stem}_from_{ascii_name(source.parent.name.lower())}.py"
        used_names.add(filename)
        safe_write(
            PACKAGE / "src/plot" / filename,
            wrapper_content(source),
            "PLOT_SUBSONIC",
            "Non-destructive wrapper preserving the original plotting implementation and CLI.",
            source,
        )


def classify_figure(path: Path) -> str | None:
    text = path.as_posix().lower()
    name = path.name.lower()
    atlas = ("atlas", "chart", "coverage", "routing", "overlap", "footprint", "pipeline", "mach_alpha")
    mode = ("mode", "modal", "pressure", "velocity", "derivative", "field", "rho_", "p_", "q_")
    ci = ("ci_", "_ci", "omega", "spectral", "growth", "blumen", "anchor")
    if any(token in name for token in mode):
        return "modes"
    if any(token in name for token in ci):
        return "ci"
    if any(token in text for token in atlas):
        return "atlas"
    return None


def classify_csv(path: Path) -> str | None:
    text = path.as_posix().lower()
    name = path.name.lower()
    atlas = (
        "atlas", "chart", "coverage", "routing", "overlap", "manifest", "global",
        "offgrid", "regional", "sampling", "history", "diagnostics_summary",
    )
    mode = (
        "mode", "modal", "field", "pressure", "velocity", "rho", "phase",
        "amplitude", "edge_band",
    )
    ci = ("ci_", "_ci", "omega", "spectral", "growth", "anchor", "blumen")
    if any(token in name for token in mode):
        return "modes"
    if any(token in name for token in ci):
        return "ci"
    if any(token in text for token in atlas):
        return "atlas"
    return None


def organized_destination(root: Path, category: str, source: Path, prefix: str) -> Path:
    relative = source.relative_to(SOURCE_ASSETS)
    parents = [ascii_name(part) for part in relative.parent.parts]
    filename = ascii_name(source.name)
    if not filename.startswith(prefix):
        filename = prefix + filename
    return root / category / Path(*parents) / filename


def organize_complete_assets_and_data() -> None:
    for source in sorted(SOURCE_ASSETS.rglob("*")):
        if not source.is_file() or ".DS_Store" in source.name:
            continue
        if source.suffix.lower() in {".png", ".pdf"}:
            category = classify_figure(source)
            if category is None:
                record(source, None, "AMBIGUOUS", "left_in_place", "Figure subject is not identifiable from path/name.", "review")
                continue
            destination = organized_destination(PACKAGE / "assets", category, source, "Fig_")
            safe_link_or_copy(source, destination, f"ASSET_{category.upper()}", "Complete organized PINN-subsonic figure archive.")
        elif source.suffix.lower() == ".csv":
            category = classify_csv(source)
            if category is None:
                record(source, None, "AMBIGUOUS", "left_in_place", "CSV content role is not identifiable from path/name.", "review")
                continue
            destination = organized_destination(PACKAGE / "datas", category, source, "Table_")
            safe_link_or_copy(source, destination, f"DATA_{category.upper()}", "Complete organized PINN-subsonic CSV archive.")


def organize_checkpoints() -> None:
    atlas_root = SOURCE_ASSETS / "joint_ci_mode_atlas_v2"
    for source in sorted(atlas_root.glob("*/*")):
        if source.is_file() and source.name in {"model_best.pt", "model_state.pt", "joint_training_metadata.json"}:
            chart = ascii_name(source.parent.name)
            destination = PACKAGE / "checkpoints/joint_atlas_v2" / chart / source.name
            safe_link_or_copy(source, destination, "CHECKPOINT_SUBSONIC", "Joint ci/mode atlas checkpoint with chart hierarchy.")

    model_saved = REPO / "model_saved"
    for experiment in sorted(model_saved.iterdir()) if model_saved.exists() else []:
        if not experiment.is_dir() or "subsonic" not in experiment.name.lower():
            continue
        family = "smoke" if experiment.name.startswith("_smoke") else "experiments"
        for source in sorted(experiment.rglob("*")):
            if not source.is_file() or source.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
                continue
            relative = source.relative_to(experiment)
            destination = (
                PACKAGE
                / "checkpoints"
                / family
                / ascii_name(experiment.name)
                / Path(*(ascii_name(part) for part in relative.parts))
            )
            safe_link_or_copy(source, destination, "CHECKPOINT_SUBSONIC", "Preserved local training experiment checkpoint hierarchy.")


def organize_stable_selected_assets() -> None:
    selections = {
        "modes": [
            SOURCE_ASSETS / "comparaisons_modales",
            SOURCE_ASSETS / "article/modes",
        ],
        "ci": [
            SOURCE_ASSETS / "comparaisons_ci",
            SOURCE_ASSETS / "article/ci",
        ],
        "atlas": [
            SOURCE_ASSETS / "article/atlas",
        ],
    }
    for category, roots in selections.items():
        for root in roots:
            if not root.exists():
                continue
            for source in sorted(root.glob("*")):
                if not source.is_file() or source.suffix.lower() not in {".png", ".pdf"}:
                    continue
                filename = ascii_name(source.name)
                if not filename.startswith("Fig_"):
                    filename = f"Fig_{category}_{filename}"
                destination = SOURCE_ASSETS / category / filename
                safe_link_or_copy(source, destination, f"ASSET_{category.upper()}", "Stable selected PINN-subsonic asset.")


def create_package_files() -> None:
    initializers = {
        PACKAGE / "__init__.py": '"""Organized subsonic Kelvin-Helmholtz PINN package."""\n',
        PACKAGE / "src/__init__.py": '"""Subsonic PINN source package."""\n',
        PACKAGE / "src/models/__init__.py": (
            '"""Subsonic PINN architectures and physics components."""\n'
            "from .model_kh_subsonic_pinn import KHSubsonicFixedMachPINN, KHSubsonicMultiMachPINN\n\n"
            '__all__ = ["KHSubsonicFixedMachPINN", "KHSubsonicMultiMachPINN"]\n'
        ),
        PACKAGE / "src/training/__init__.py": '"""Training modules and entry points for subsonic PINNs."""\n',
        PACKAGE / "src/plot/__init__.py": '"""Plotting entry points for subsonic PINN results."""\n',
        PACKAGE / "src/configs/README.md": (
            "# Subsonic configuration files\n\n"
            "No operational YAML specific to the PINN-subsonic atlas was found. "
            "Chart domains and training parameters remain encoded in CSV manifests, "
            "checkpoint metadata, and Python CLI defaults; they are not synthesized here.\n"
        ),
    }
    source = REPO / "pinn_subsonic/migrate_subsonic_tree.py"
    for destination, content in initializers.items():
        safe_write(destination, content, "CONFIG_SUBSONIC", "Required package structure.", source)


def record_ambiguous() -> None:
    for value in AMBIGUOUS:
        source = REPO / value
        if source.exists():
            record(
                source,
                None,
                "AMBIGUOUS",
                "left_in_place",
                "Historical, direct-modal-supervision, obsolete, or explicitly broken experiment.",
                "review",
            )


def write_inventory() -> None:
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "original_path", "new_path", "category", "action", "sha256",
        "size_bytes", "reason", "status",
    ]
    with INVENTORY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ROWS)


def main() -> None:
    create_package_files()
    copy_core_modules()
    create_code_entrypoints()
    organize_checkpoints()
    organize_complete_assets_and_data()
    organize_stable_selected_assets()
    record_ambiguous()
    write_inventory()
    counts: dict[str, int] = {}
    for row in ROWS:
        counts[str(row["action"])] = counts.get(str(row["action"]), 0) + 1
    print(f"inventory={INVENTORY.relative_to(REPO)} rows={len(ROWS)}")
    print("actions", counts)
    print(f"conflicts={len(CONFLICTS)}")


if __name__ == "__main__":
    main()
