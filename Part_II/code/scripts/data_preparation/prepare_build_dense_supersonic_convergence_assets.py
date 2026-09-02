#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve(repo: Path, path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo / path


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=False)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate dense supersonic convergence tasks and build the validation asset package."
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=Path("assets/classic_supersonic/dense_kappa_q_campaign_v1_FINAL_FREEZE"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            "classic_supersonic/reproducibility/results/"
            "dense_supersonic_convergence_audit_v1"
        ),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(
            "assets/classic_supersonic/"
            "dense_kappa_q_campaign_v1_CONVERGENCE_AUDIT"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def concat_task_csv(results_dir: Path, filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for task_dir in sorted((results_dir / "tasks").glob("task_*")):
        path = task_dir / filename
        if path.is_file() and path.stat().st_size:
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()




def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

def finite_max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.max()) if len(values) else math.nan


def finite_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.median()) if len(values) else math.nan


def spectral_errors(runs: pd.DataFrame) -> pd.DataFrame:
    numeric = ["cr", "ci", "omega_i", "residual_norm"]
    for column in numeric:
        runs[column] = pd.to_numeric(runs.get(column), errors="coerce")
    references = {
        "box": "box_L50",
        "integration": "strict_L40_y1",
        "matching": "strict_L40_y1",
    }
    membership = {
        "box": ["box_L20", "box_L30", "strict_L40_y1", "box_L50"],
        "integration": ["accuracy_coarse", "accuracy_nominal", "strict_L40_y1"],
        "matching": ["matching_y0p5", "strict_L40_y1", "matching_y1p5"],
    }
    rows: list[dict[str, Any]] = []
    keys = ["task_index", "role", "Mach", "alpha"]
    for _, group in runs.groupby(keys, sort=True):
        by_id = {str(row["audit_setting"]): row for _, row in group.iterrows()}
        for sweep, setting_ids in membership.items():
            reference = by_id.get(references[sweep])
            if reference is None:
                continue
            for setting_id in setting_ids:
                row = by_id.get(setting_id)
                if row is None:
                    continue
                dcr = float(row["cr"] - reference["cr"]) if np.isfinite(row["cr"]) and np.isfinite(reference["cr"]) else math.nan
                dci = float(row["ci"] - reference["ci"]) if np.isfinite(row["ci"]) and np.isfinite(reference["ci"]) else math.nan
                domega = float(row["omega_i"] - reference["omega_i"]) if np.isfinite(row["omega_i"]) and np.isfinite(reference["omega_i"]) else math.nan
                rows.append(
                    {
                        **{key: row[key] for key in keys},
                        "sweep": sweep,
                        "setting": setting_id,
                        "reference_setting": references[sweep],
                        "level": float(row.get("audit_level", math.nan)),
                        "accepted": bool(row.get("accepted", False)),
                        "residual_norm": float(row.get("residual_norm", math.nan)),
                        "abs_error_cr": abs(dcr),
                        "abs_error_ci": abs(dci),
                        "abs_error_omega_i": abs(domega),
                        "complex_c_error": math.hypot(dcr, dci) if np.isfinite(dcr) and np.isfinite(dci) else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def save_scatter_map(frame: pd.DataFrame, value: str, title: str, label: str, output: Path) -> None:
    values = pd.to_numeric(frame[value], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values) & (values > 0.0)
    if not np.any(finite):
        raise ValueError(f"No positive finite values for {value}")
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    scatter = ax.scatter(
        pd.to_numeric(frame.loc[finite, "alpha"]),
        pd.to_numeric(frame.loc[finite, "Mach"]),
        c=values[finite],
        s=18,
        norm=LogNorm(vmin=max(values[finite].min(), 1e-16), vmax=values[finite].max()),
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(label)
    ax.set_xlabel(r"Wavenumber $\alpha$")
    ax.set_ylabel(r"Mach number $M$")
    ax.set_title(title)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def aggregate_curve(errors: pd.DataFrame, sweep: str) -> pd.DataFrame:
    subset = errors[errors["sweep"].eq(sweep)].copy()
    return (
        subset.groupby(["setting", "level"], as_index=False)
        .agg(
            median_complex_error=("complex_c_error", "median"),
            max_complex_error=("complex_c_error", "max"),
            median_ci_error=("abs_error_ci", "median"),
            max_ci_error=("abs_error_ci", "max"),
        )
        .sort_values("level")
    )


def save_error_curve(table: pd.DataFrame, title: str, xlabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
    x = pd.to_numeric(table["level"], errors="coerce")
    median = np.maximum(pd.to_numeric(table["median_complex_error"], errors="coerce"), 1e-18)
    maximum = np.maximum(pd.to_numeric(table["max_complex_error"], errors="coerce"), 1e-18)
    ax.plot(x, median, marker="o", label=r"Median $|\Delta c|$")
    ax.plot(x, maximum, marker="s", label=r"Maximum $|\Delta c|$")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Complex eigenvalue error $|\Delta c|$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_matching(errors: pd.DataFrame, output: Path) -> None:
    table = aggregate_curve(errors, "matching")
    save_error_curve(table, "Sensitivity to Riccati matching location", r"Matching location $y_m$", output)


def save_modal_curve(modal_errors: pd.DataFrame, output: Path) -> None:
    subset = modal_errors[~as_bool(modal_errors["is_modal_reference"])].copy()
    order = {"modal_coarse": 0, "modal_nominal": 1}
    table = (
        subset.groupby("modal_setting", as_index=False)
        .agg(
            median_p_error=("p_rel_l2_amp_mask", "median"),
            max_p_error=("p_rel_l2_amp_mask", "max"),
        )
    )
    table["order"] = table["modal_setting"].map(order)
    table = table.sort_values("order")
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
    ax.plot(x, np.maximum(table["median_p_error"], 1e-18), marker="o", label="Median")
    ax.plot(x, np.maximum(table["max_p_error"], 1e-18), marker="s", label="Maximum")
    ax.set_yscale("log")
    ax.set_xticks(x, table["modal_setting"].tolist())
    ax.set_ylabel(r"Phase-aligned modal error $\|p-p_{ref}\|_2/\|p_{ref}\|_2$")
    ax.set_title("Modal reconstruction convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def add_summary_page(pdf: PdfPages, summary: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    status = summary["audit_status"]
    fig.text(0.5, 0.93, "Dense classical supersonic convergence audit", ha="center", fontsize=18, weight="bold")
    fig.text(0.5, 0.88, f"Audit status: {status}", ha="center", fontsize=16, weight="bold")
    lines = [
        f"Audit points: {summary['n_audit_points']} (17 Mach x 3 regimes)",
        f"Spectral reruns: {summary['n_spectral_runs']}",
        f"Modal reruns: {summary['n_modal_runs']}",
        "",
        f"Dense reference max residual: {summary['metrics']['dense_reference_residual_max']:.3e}",
        f"Audit rerun max residual: {summary['metrics']['spectral_run_residual_max']:.3e}",
        f"Box L40 vs L50 max |Delta c|: {summary['metrics']['box_L40_vs_L50_complex_error_max']:.3e}",
        f"Nominal vs strict integration max |Delta c|: {summary['metrics']['accuracy_nominal_vs_strict_complex_error_max']:.3e}",
        f"Matching-location max |Delta c|: {summary['metrics']['matching_location_complex_error_max']:.3e}",
        f"Nominal modal max relative L2: {summary['metrics']['modal_nominal_p_rel_l2_max']:.3e}",
        "",
        "PASS means every explicit criterion in convergence_summary.json is satisfied.",
        "The dense residual maps are internal consistency checks; the independent",
        "box/integration/matching/modal reruns provide the numerical convergence test.",
    ]
    fig.text(0.10, 0.80, "\n".join(lines), va="top", fontsize=11, family="monospace")
    failed = [name for name, item in summary["criteria"].items() if not item["passed"]]
    if failed:
        fig.text(0.10, 0.25, "Failed criteria:\n- " + "\n- ".join(failed), va="top", fontsize=11)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_curve_page(pdf: PdfPages, table: pd.DataFrame, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 5.8), constrained_layout=True)
    x = pd.to_numeric(table["level"], errors="coerce")
    ax.plot(x, np.maximum(table["median_complex_error"], 1e-18), marker="o", label=r"Median $|\Delta c|$")
    ax.plot(x, np.maximum(table["max_complex_error"], 1e-18), marker="s", label=r"Maximum $|\Delta c|$")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Complex eigenvalue error $|\Delta c|$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def basic_pdf_check(path: Path) -> int:
    data = path.read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-4096:]:
        raise ValueError(f"Invalid PDF structure: {path}")
    return len(re.findall(rb"/Type\s*/Page\b", data))


def write_manifest(root: Path) -> None:
    rows: list[dict[str, Any]] = []
    checksum_lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.csv", "SHA256SUMS.txt"}:
            continue
        relative = path.relative_to(root)
        digest = sha256_file(path)
        rows.append({"path": str(relative), "size_bytes": path.stat().st_size, "sha256": digest})
        checksum_lines.append(f"{digest}  {relative}\n")
    atomic_csv(root / "manifest.csv", pd.DataFrame(rows))
    atomic_text(root / "SHA256SUMS.txt", "".join(checksum_lines))


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    freeze_dir = resolve(repo, args.freeze_dir)
    results_dir = resolve(repo, args.results_dir)
    assets_dir = resolve(repo, args.assets_dir)
    if assets_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Assets already exist: {assets_dir}")
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True)
    tables_dir = assets_dir / "convergence_tables"
    plots_dir = assets_dir / "convergence_plots"
    raw_dir = assets_dir / "raw_audit"
    provenance_dir = assets_dir / "provenance"
    tables_dir.mkdir()
    plots_dir.mkdir()
    raw_dir.mkdir()
    provenance_dir.mkdir()

    points = pd.read_csv(results_dir / "audit_points.csv")
    config = json.loads((results_dir / "audit_config.json").read_text(encoding="utf-8"))
    done = sorted((results_dir / "tasks").glob("task_*/DONE.json"))
    if len(done) != len(points):
        raise RuntimeError(f"Expected {len(points)} completed tasks, found {len(done)}.")
    spectral_runs = concat_task_csv(results_dir, "spectral_runs.csv")
    modal_runs = concat_task_csv(results_dir, "modal_runs.csv")
    modal_error_frame = concat_task_csv(results_dir, "modal_errors.csv")
    expected_spectral = len(points) * len(config["spectral_settings"])
    expected_modal = len(points) * len(config["modal_settings"])
    if len(spectral_runs) != expected_spectral:
        raise RuntimeError(f"Expected {expected_spectral} spectral runs, found {len(spectral_runs)}.")
    if len(modal_runs) != expected_modal:
        raise RuntimeError(f"Expected {expected_modal} modal runs, found {len(modal_runs)}.")

    reference_path = freeze_dir / "classical_supersonic_maps/classical_supersonic_dense_reference.csv"
    reference = pd.read_csv(reference_path)
    errors = spectral_errors(spectral_runs)

    atomic_csv(tables_dir / "audit_points.csv", points)
    atomic_csv(tables_dir / "spectral_convergence_runs.csv", spectral_runs)
    atomic_csv(tables_dir / "spectral_convergence_errors.csv", errors)
    atomic_csv(tables_dir / "modal_convergence_runs.csv", modal_runs)
    atomic_csv(tables_dir / "modal_convergence_errors.csv", modal_error_frame)
    atomic_csv(tables_dir / "dense_internal_diagnostics.csv", reference)

    box = aggregate_curve(errors, "box")
    integration = aggregate_curve(errors, "integration")
    matching = aggregate_curve(errors, "matching")
    atomic_csv(tables_dir / "spectral_box_summary.csv", box)
    atomic_csv(tables_dir / "spectral_integration_summary.csv", integration)
    atomic_csv(tables_dir / "matching_location_summary.csv", matching)

    save_scatter_map(
        reference,
        "residual_norm",
        "Dense reference: Riccati matching residual",
        r"Residual norm $\sqrt{\Delta\kappa^2+\Delta q^2}$",
        plots_dir / "dense_solver_residual_map.pdf",
    )
    save_scatter_map(
        reference,
        "mode_gamma_mismatch_at_match",
        "Dense reference: reconstructed-mode matching residual",
        r"Mode matching residual",
        plots_dir / "dense_mode_matching_residual_map.pdf",
    )
    save_error_curve(
        box,
        "Spectral box convergence",
        r"Spectral half-extent $L$",
        plots_dir / "spectral_box_convergence.pdf",
    )
    save_error_curve(
        integration,
        "Spectral integration convergence",
        "Maximum integration step",
        plots_dir / "spectral_integration_convergence.pdf",
    )
    save_matching(errors, plots_dir / "matching_location_sensitivity.pdf")
    save_modal_curve(modal_error_frame, plots_dir / "modal_convergence.pdf")

    def error_for(sweep: str, setting: str) -> float:
        subset = errors[errors["sweep"].eq(sweep) & errors["setting"].eq(setting)]
        return finite_max(subset["complex_c_error"])

    thresholds = config["acceptance_thresholds"]
    metrics = {
        "dense_reference_residual_max": finite_max(reference["residual_norm"]),
        "spectral_run_residual_max": finite_max(spectral_runs["residual_norm"]),
        "box_L40_vs_L50_complex_error_max": error_for("box", "strict_L40_y1"),
        "accuracy_nominal_vs_strict_complex_error_max": error_for("integration", "accuracy_nominal"),
        "matching_location_complex_error_max": finite_max(
            errors[errors["sweep"].eq("matching") & ~errors["setting"].eq("strict_L40_y1")]["complex_c_error"]
        ),
        "modal_nominal_p_rel_l2_max": finite_max(
            modal_error_frame[modal_error_frame["modal_setting"].eq("modal_nominal")]["p_rel_l2_amp_mask"]
        ),
        "spectral_acceptance_fraction": float(as_bool(spectral_runs["accepted"]).mean()),
        "modal_success_fraction": float(as_bool(modal_runs["modal_success"]).mean()),
    }
    criteria: dict[str, dict[str, Any]] = {}
    for name, limit in thresholds.items():
        value = metrics[name]
        criteria[name] = {
            "value": value,
            "limit": float(limit),
            "operator": "<=",
            "passed": bool(np.isfinite(value) and value <= float(limit)),
        }
    criteria["all_spectral_reruns_accepted"] = {
        "value": metrics["spectral_acceptance_fraction"],
        "limit": 1.0,
        "operator": "==",
        "passed": bool(metrics["spectral_acceptance_fraction"] == 1.0),
    }
    criteria["all_modal_reruns_completed"] = {
        "value": metrics["modal_success_fraction"],
        "limit": 1.0,
        "operator": "==",
        "passed": bool(metrics["modal_success_fraction"] == 1.0),
    }
    audit_status = "PASS" if all(item["passed"] for item in criteria.values()) else "FAIL"
    summary = {
        "created_at": utc_now(),
        "audit_status": audit_status,
        "n_audit_points": int(len(points)),
        "n_spectral_runs": int(len(spectral_runs)),
        "n_modal_runs": int(len(modal_runs)),
        "metrics": metrics,
        "criteria": criteria,
        "scope": {
            "Mach_values": sorted(float(value) for value in points["Mach"].unique()),
            "roles": sorted(str(value) for value in points["role"].unique()),
            "note": (
                "Dense residual diagnostics cover all 770 retained points. Independent "
                "numerical convergence reruns cover low-alpha, peak-growth and near-neutral "
                "points for every Mach."
            ),
        },
    }
    atomic_json(tables_dir / "convergence_summary.json", summary)

    report_path = plots_dir / "convergence_audit_report.pdf"
    with PdfPages(report_path) as pdf:
        add_summary_page(pdf, summary)
        add_curve_page(pdf, box, "Spectral box convergence", r"Spectral half-extent $L$")
        add_curve_page(pdf, integration, "Spectral integration convergence", "Maximum integration step")
        add_curve_page(pdf, matching, "Matching-location sensitivity", r"Matching location $y_m$")
        fig, ax = plt.subplots(figsize=(8.27, 5.8), constrained_layout=True)
        subset = modal_error_frame[modal_error_frame["modal_setting"].isin(["modal_coarse", "modal_nominal"])]
        modal_summary = subset.groupby("modal_setting", as_index=False)["p_rel_l2_amp_mask"].agg(["median", "max"]).reset_index()
        x = np.arange(len(modal_summary))
        ax.plot(x, np.maximum(modal_summary["median"], 1e-18), marker="o", label="Median")
        ax.plot(x, np.maximum(modal_summary["max"], 1e-18), marker="s", label="Maximum")
        ax.set_yscale("log")
        ax.set_xticks(x, modal_summary["modal_setting"].tolist())
        ax.set_ylabel("Phase-aligned relative L2 mode error")
        ax.set_title("Modal reconstruction convergence")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    for pdf in sorted(plots_dir.glob("*.pdf")):
        pages = basic_pdf_check(pdf)
        print(f"Verified PDF: {pdf.name} ({pages} page markers)")

    for task_dir in sorted((results_dir / "tasks").glob("task_*")):
        target = raw_dir / task_dir.name
        target.mkdir()
        for name in ("DONE.json", "spectral_runs.csv", "modal_runs.csv", "modal_errors.csv", "modal_arrays.npz"):
            source = task_dir / name
            if source.is_file():
                shutil.copy2(source, target / name)
    shutil.copy2(results_dir / "audit_config.json", provenance_dir / "audit_config.json")
    shutil.copy2(results_dir / "audit_points.csv", provenance_dir / "audit_points.csv")
    for script_name in (
        "run_dense_supersonic_convergence_audit.py",
        "build_dense_supersonic_convergence_assets.py",
    ):
        source = repo / "classic_supersonic/scripts/validation" / script_name
        if source.is_file():
            shutil.copy2(source, provenance_dir / script_name)

    readme = f"""# Dense classical supersonic convergence audit\n\nStatus: **{audit_status}**\n\nThis package complements the frozen dense reference. It contains:\n\n- internal residual diagnostics for all 770 retained eigenpairs and modes;\n- independent spectral box, integration-accuracy and matching-location reruns;\n- modal-resolution reruns;\n- three audit points per Mach: low alpha, peak growth and near neutrality;\n- explicit criteria and numerical values in `convergence_tables/convergence_summary.json`.\n\nThe package is intentionally not made read-only so it can be extracted or replaced locally without permission errors.\n"""
    atomic_text(assets_dir / "README.md", readme)
    write_manifest(assets_dir)

    bundle = assets_dir.parent / f"{assets_dir.name}.tar.gz"
    checksum = bundle.with_suffix(bundle.suffix + ".sha256")
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(assets_dir, arcname=assets_dir.name)
    digest = sha256_file(bundle)
    atomic_text(checksum, f"{digest}  {bundle.name}\n")
    with tarfile.open(bundle, "r:gz") as archive:
        archive.getmembers()

    print(f"AUDIT STATUS: {audit_status}")
    print(f"Convergence assets: {assets_dir}")
    print(f"Transfer bundle: {bundle}")
    print(f"Checksum: {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
