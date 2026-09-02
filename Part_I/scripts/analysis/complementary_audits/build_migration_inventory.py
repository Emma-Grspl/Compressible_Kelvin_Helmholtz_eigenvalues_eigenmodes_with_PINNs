"""Build the file-level provenance inventory for the complementary audits.

This utility performs no scientific computation. It records SHA-256 hashes,
migration decisions, and destination verification for the former temporary
``complementary_phase`` working tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


PART_I_ROOT = Path(__file__).resolve().parents[3]

ASSET_TARGETS = {
    "assets/section3_pinn_spectral_atlas/Fig_spectral_modal_architecture_subsonic.png": "assets/complementary_audits/final_figures/Fig02_spectral_modal_architecture_subsonic.png",
    "assets/section3_pinn_spectral_atlas/Fig_spectral_modal_architecture_subsonic.pdf": "assets/complementary_audits/final_figures/Fig02_spectral_modal_architecture_subsonic.pdf",
    "assets/ci_curve_hybrid4_vs_physics_only.png": "assets/complementary_audits/final_figures/Fig04_fixed_Mach_four_anchor_vs_physics_only.png",
    "assets/Fig_anchor_budget_comparison.png": "assets/complementary_audits/final_figures/Fig05_anchor_budget_comparison.png",
    "assets/Fig_anchor_budget_comparison.pdf": "assets/complementary_audits/final_figures/Fig05_anchor_budget_comparison.pdf",
    "assets/Fig_atlas_overlap_consistency_N340.png": "assets/complementary_audits/final_figures/Fig06_atlas_routing_interface_consistency_N340.png",
    "assets/Fig_atlas_overlap_consistency_N340.pdf": "assets/complementary_audits/final_figures/Fig06_atlas_routing_interface_consistency_N340.pdf",
    "assets/SuppFig_holdout_anchors_only_vs_physics_N340.png": "assets/complementary_audits/final_figures/Fig07_holdout_anchors_only_vs_physics_N340.png",
    "assets/SuppFig_holdout_anchors_only_vs_physics_N340.pdf": "assets/complementary_audits/final_figures/Fig07_holdout_anchors_only_vs_physics_N340.pdf",
    "assets/Fig_representative_mode_M05_a05_N340.png": "assets/complementary_audits/final_figures/Fig08_representative_mode_M05_a05_N340.png",
    "assets/Fig_representative_mode_M05_a05_N340.pdf": "assets/complementary_audits/final_figures/Fig08_representative_mode_M05_a05_N340.pdf",
    "assets/section4_results/Fig_near_neutral_growth_rate_cuts_N340.png": "assets/complementary_audits/final_figures/Fig09_near_neutral_growth_rate_cuts_N340.png",
    "assets/section4_results/Fig_near_neutral_growth_rate_cuts_N340.pdf": "assets/complementary_audits/final_figures/Fig09_near_neutral_growth_rate_cuts_N340.pdf",
    "assets/SuppFig07_GEP_N_convergence_FIXED.png": "assets/complementary_audits/final_figures/FigS4_1_GEP_N_convergence.png",
    "assets/Fig_ci_error_heatmap_PINN_GEP_vs_classical_N340.png": "assets/complementary_audits/final_figures/FigS5_1_selected_GEP_error_heatmap_N340.png",
    "assets/Fig_ci_error_heatmap_PINN_GEP_vs_classical_N340.pdf": "assets/complementary_audits/final_figures/FigS5_1_selected_GEP_error_heatmap_N340.pdf",
    "assets/Fig_paired_modal_error_ecdf_20_N340.png": "assets/complementary_audits/final_figures/FigS6_1_direct_neural_vs_selected_GEP_modal_error_ecdf_20_N340.png",
    "assets/Fig_paired_modal_error_ecdf_20_N340.pdf": "assets/complementary_audits/final_figures/FigS6_1_direct_neural_vs_selected_GEP_modal_error_ecdf_20_N340.pdf",
    "assets/Fig_near_neutral_error_scaling_N340.png": "assets/complementary_audits/final_figures/FigS8_1_near_neutral_error_scaling_N340.png",
    "assets/Fig_near_neutral_error_scaling_N340.pdf": "assets/complementary_audits/final_figures/FigS8_1_near_neutral_error_scaling_N340.pdf",
}

SUPERSEDED_ASSETS = {
    "assets/Fig_anchor_budget_comparison_before_title_fix.png",
    "assets/Fig_spectral_modal_architecture_subsonic.png",
    "assets/Fig_spectral_modal_architecture_subsonic.pdf",
    "assets/SuppFig07_GEP_N_convergence.png",
    "assets/SuppFig07_GEP_N_convergence.pdf",
    "assets/SuppFig07_GEP_N_convergence_before_title_fix.png",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(relative: str) -> tuple[str, str, str]:
    path = Path(relative)
    name = path.name

    if relative.startswith(".venv-kh-subsonic/"):
        return "environment", "EXCLUDED_LOCAL_ENV", ""
    if name == ".DS_Store":
        return "metadata", "EXCLUDED_OS_METADATA", ""
    if relative.startswith("logs/") or path.suffix in {".log", ".out", ".err"}:
        return "log", "EXCLUDED_TRANSIENT_LOG", ""
    if "backup_before_final_wording" in path.parts or ".bak" in name:
        return "script_backup", "EXCLUDED_SUPERSEDED_BACKUP", ""
    if relative in SUPERSEDED_ASSETS:
        return "asset", "EXCLUDED_SUPERSEDED_ASSET", ""
    if relative in ASSET_TARGETS:
        return "asset", "MIGRATED_CANONICAL_ASSET", ASSET_TARGETS[relative]
    if relative.startswith("assets/") and path.suffix.lower() == ".csv":
        target = Path("results/complementary_audits/curated") / path.relative_to("assets")
        return "result", "MIGRATED_RESULT", target.as_posix()
    if relative == "assets/Table_S7_4_normalized_residuals.tex":
        return "documentation", "MIGRATED_DOCUMENT", "docs/complementary_audits/Table_S7_4_normalized_residuals.tex"
    if path.parts and path.parts[0].startswith("phase_"):
        target = Path("results/complementary_audits") / path
        return "result", "MIGRATED_RESULT", target.as_posix()
    if relative.startswith("scripts/") and path.suffix == ".py":
        target = Path("scripts/analysis/complementary_audits") / name
        return "script", "MIGRATED_ANALYSIS_SCRIPT", target.as_posix()
    if relative == "MANIFEST.txt":
        return "provenance", "MIGRATED_PROVENANCE", "provenance/complementary_audits/ORIGINAL_MANIFEST.txt"
    if relative == "requirements.txt":
        return "documentation", "MIGRATED_DOCUMENT", "docs/complementary_audits/requirements.txt"
    return "unknown", "REVIEW_NOT_MIGRATED", ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Path to the former complementary_phase directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PART_I_ROOT / "provenance" / "complementary_audits",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    duplicate_candidates: defaultdict[tuple[str, int], list[str]] = defaultdict(list)
    for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = file_path.relative_to(source).as_posix()
        category, decision, target_relative = classify(relative)
        source_hash = sha256(file_path)
        target_path = PART_I_ROOT / target_relative if target_relative else None
        target_exists = bool(target_path and target_path.is_file())
        target_hash = sha256(target_path) if target_exists else ""
        hash_verified = bool(target_exists and source_hash == target_hash)
        rows.append(
            {
                "source_path": f"complementary_phase/{relative}",
                "size_bytes": file_path.stat().st_size,
                "sha256": source_hash,
                "category": category,
                "decision": decision,
                "target_path": target_relative,
                "target_exists": target_exists,
                "target_sha256": target_hash,
                "hash_verified": hash_verified,
            }
        )
        if not relative.startswith(".venv-kh-subsonic/"):
            duplicate_candidates[(source_hash, file_path.stat().st_size)].append(relative)

    inventory_path = output_dir / "COMPLEMENTARY_PHASE_INVENTORY.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    duplicate_path = output_dir / "COMPLEMENTARY_DUPLICATE_GROUPS.csv"
    with duplicate_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["sha256", "size_bytes", "copy_count", "source_paths"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for (digest, size), paths in sorted(duplicate_candidates.items()):
            if len(paths) > 1:
                writer.writerow(
                    {
                        "sha256": digest,
                        "size_bytes": size,
                        "copy_count": len(paths),
                        "source_paths": ";".join(f"complementary_phase/{path}" for path in paths),
                    }
                )

    migrated = [row for row in rows if str(row["decision"]).startswith("MIGRATED_")]
    failed = [row for row in migrated if not row["hash_verified"]]
    review = [row for row in rows if row["decision"] == "REVIEW_NOT_MIGRATED"]
    print(f"inventory={inventory_path}")
    print(f"files={len(rows)} migrated={len(migrated)} failed={len(failed)} review={len(review)}")
    if failed or review:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
