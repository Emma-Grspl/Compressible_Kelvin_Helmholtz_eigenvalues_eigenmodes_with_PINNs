#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[3] if "classic_supersonic" in Path(__file__).parts else Path.cwd()
PACKAGE = REPO / "classic_supersonic"

SPECTRAL = PACKAGE / "data/spectral/supersonic_primary_modal_spectral_133pts.csv"
MODAL = PACKAGE / "data/modal/supersonic_reference_v2_modal_raw.parquet"
BLUMEN_POINTS = PACKAGE / "data/blumen/blumen_ci_digitized_points.csv"
BLUMEN_LEVELS = PACKAGE / "data/blumen/blumen_ci_curve_levels.csv"

EXPECTED_STATUS_COUNTS = {
    "modal_spectral_validated_with_exported_fields": 44,
    "validated_core_stable_tail_sensitive": 89,
}
EXPECTED_MACH_COUNTS = {
    1.10: 2, 1.20: 6, 1.25: 1, 1.30: 9, 1.33: 1,
    1.40: 11, 1.50: 8, 1.60: 2, 1.70: 4, 1.80: 49, 1.90: 40,
}
FIELDS = [("p", "Pressure"), ("rho", "Density"), ("u", "Streamwise velocity"), ("v", "Transverse velocity")]


def key(mach: object, alpha: object) -> tuple[float, float]:
    return round(float(mach), 6), round(float(alpha), 6)


def load_spectral() -> pd.DataFrame:
    frame = pd.read_csv(SPECTRAL)
    required = {"Mach", "alpha", "cr", "ci", "validation_status"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing spectral columns: {sorted(missing)}")
    for column in ["Mach", "alpha", "cr", "ci", "omega_i"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "omega_i" not in frame.columns:
        frame["omega_i"] = frame["alpha"] * frame["ci"]
    frame = (frame.dropna(subset=["Mach", "alpha", "cr", "ci"])
             .drop_duplicates(["Mach", "alpha"])
             .sort_values(["Mach", "alpha"]).reset_index(drop=True))
    if len(frame) != 133:
        raise RuntimeError(f"Expected 133 points; observed {len(frame)}")
    if frame["validation_status"].value_counts().to_dict() != EXPECTED_STATUS_COUNTS:
        raise RuntimeError("Unexpected validation-status counts")
    mach_counts = {round(float(m), 2): int(n) for m, n in frame.groupby("Mach").size().items()}
    if mach_counts != EXPECTED_MACH_COUNTS:
        raise RuntimeError(f"Unexpected Mach counts: {mach_counts}")
    frame["key"] = [key(m, a) for m, a in zip(frame["Mach"], frame["alpha"])]
    return frame


def load_modal(expected: set[tuple[float, float]]) -> pd.DataFrame:
    columns = [
        "Mach", "alpha", "y",
        "p_real", "p_imag", "rho_real", "rho_imag",
        "u_real", "u_imag", "v_real", "v_imag",
    ]
    frame = pd.read_parquet(MODAL, columns=columns)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Mach", "alpha", "y"])
    frame["key"] = [key(m, a) for m, a in zip(frame["Mach"], frame["alpha"])]
    frame = frame.loc[frame["key"].isin(expected)].copy()
    observed = set(frame["key"])
    if observed != expected:
        raise RuntimeError(f"Modal key mismatch; missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    if frame.duplicated(["Mach", "alpha", "y"], keep=False).any():
        raise RuntimeError("Duplicate (Mach, alpha, y) rows")
    return frame


def load_blumen() -> tuple[pd.DataFrame, str]:
    points = pd.read_csv(BLUMEN_POINTS)
    levels = pd.read_csv(BLUMEN_LEVELS)
    points["curve_id"] = pd.to_numeric(points["curve_id"], errors="coerce").astype("Int64")
    levels["curve_id"] = pd.to_numeric(levels["curve_id"], errors="coerce").astype("Int64")
    levels["ci_level"] = pd.to_numeric(levels["ci_level"], errors="coerce")
    include_column = next((c for c in ["include_in_overlay", "include_in_quantitative_comparison"] if c in levels.columns), None)
    if include_column is not None:
        include = levels[include_column].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
        levels = levels.loc[include].copy()
    merged = points.merge(levels[["curve_id", "curve_label", "family", "ci_level"]], on="curve_id", how="inner", validate="many_to_one")
    merged = merged.loc[~merged["family"].astype(str).eq("cr_special")].copy()
    x = pd.to_numeric(merged["Mach"], errors="coerce").to_numpy(float)
    alpha = pd.to_numeric(merged["alpha"], errors="coerce").to_numpy(float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise RuntimeError("No finite Blumen Mach values")
    if float(np.nanmin(finite)) < 1.0 and float(np.nanmax(finite)) <= 1.1:
        mach = x + 0.9
        calibration = "digitized x + 0.9"
    else:
        mach = x
        calibration = "already physical Mach"
    merged["Mach_physical"] = mach
    merged["alpha"] = alpha
    return merged.dropna(subset=["Mach_physical", "alpha"]), calibration


def build_overlay(spectral: pd.DataFrame, pdf: Path, png: Path) -> dict:
    blumen, calibration = load_blumen()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    curves = blumen[["curve_id", "curve_label", "ci_level"]].drop_duplicates().sort_values(["ci_level", "curve_id"], na_position="first")
    for _, curve in curves.iterrows():
        selected = blumen.loc[blumen["curve_id"].eq(curve["curve_id"])]
        ax.scatter(selected["Mach_physical"], selected["alpha"], marker=".", s=22, label=f"Blumen {curve['curve_label']}", zorder=1)
    norm = Normalize(vmin=float(spectral["ci"].min()), vmax=float(spectral["ci"].max()))
    exported = spectral.loc[spectral["validation_status"].eq("modal_spectral_validated_with_exported_fields")]
    tail = spectral.loc[spectral["validation_status"].eq("validated_core_stable_tail_sensitive")]
    plotted = ax.scatter(exported["Mach"], exported["alpha"], c=exported["ci"], cmap="magma", norm=norm, marker="o", s=48, edgecolors="black", linewidths=0.55, label="validated modal-spectral (44)", zorder=4)
    ax.scatter(tail["Mach"], tail["alpha"], c=tail["ci"], cmap="magma", norm=norm, marker="s", s=40, edgecolors="black", linewidths=0.45, label="core-modal validated, tail-sensitive (89)", zorder=3)
    fig.colorbar(plotted, ax=ax, pad=0.02, label=r"Reference $c_i$")
    ax.set_xlabel("Mach number")
    ax.set_ylabel(r"Wavenumber $\alpha$")
    ax.set_title("Blumen digitized $c_i$ isolines and 133-point classical supersonic reference")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=6.8, ncols=3, loc="best")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"blumen_mach_calibration": calibration, "blumen_points": len(blumen)}


def core_limit(group: pd.DataFrame) -> float:
    y = group["y"].to_numpy(float)
    full = float(np.nanmax(np.abs(y)))
    envelope = np.zeros(len(group), float)
    for field, _ in FIELDS:
        values = group[f"{field}_real"].to_numpy(float) + 1j * group[f"{field}_imag"].to_numpy(float)
        peak = float(np.nanmax(np.abs(values)))
        if np.isfinite(peak) and peak > 0:
            envelope = np.maximum(envelope, np.abs(values) / peak)
    mask = np.isfinite(envelope) & (envelope >= 1e-3)
    if not mask.any():
        return full
    return min(full, max(min(10.0, full), 1.08 * float(np.nanmax(np.abs(y[mask])))))


def plot_field(ax: plt.Axes, y: np.ndarray, values: np.ndarray, limit: float, title: str) -> None:
    peak = float(np.nanmax(np.abs(values)))
    if not np.isfinite(peak) or peak <= 0:
        raise RuntimeError(f"Invalid field normalization for {title}")
    values = values / peak
    mask = np.abs(y) <= limit
    ax.plot(y[mask], values.real[mask], label="real", linewidth=0.9)
    ax.plot(y[mask], values.imag[mask], label="imaginary", linewidth=0.9)
    ax.plot(y[mask], np.abs(values[mask]), "--", label="modulus", linewidth=0.9)
    ax.axhline(0.0, linewidth=0.6)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-1.06, 1.06)
    ax.set_title(title)
    ax.set_xlabel("y")
    ax.set_ylabel("normalized mode")
    ax.grid(True, alpha=0.2)


def build_modes(spectral: pd.DataFrame, modal: pd.DataFrame, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    groups = {k: g.sort_values("y") for k, g in modal.groupby("key", sort=False)}
    with PdfPages(pdf_path, metadata={"Title": "Classical supersonic 133-point modal atlas", "Author": "Emma Grospellier"}) as pdf:
        cover = plt.figure(figsize=(11.69, 8.27))
        cover.text(0.5, 0.84, "Classical supersonic modal-spectral reference", ha="center", fontsize=18)
        cover.text(0.5, 0.76, "133 points: 44 validated modal-spectral + 89 core-modal validated, tail-sensitive", ha="center", fontsize=11)
        cover.text(0.5, 0.15, "One page per point; p, rho, u and v are normalized independently.\nSolid: real and imaginary parts. Dashed: modulus.", ha="center", fontsize=9)
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)
        for index, row in spectral.iterrows():
            group = groups[row["key"]]
            y = group["y"].to_numpy(float)
            limit = core_limit(group)
            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
            for ax, (field, label) in zip(axes.flat, FIELDS):
                values = group[f"{field}_real"].to_numpy(float) + 1j * group[f"{field}_imag"].to_numpy(float)
                plot_field(ax, y, values, limit, f"{label} {field}")
            axes.flat[0].legend(frameon=False, ncols=3, loc="best")
            fig.suptitle(
                f"M={row['Mach']:.2f}, alpha={row['alpha']:.6f}, cr={row['cr']:.8f}, ci={row['ci']:.8f}, omega_i={row['omega_i']:.8e}\n"
                f"status={row['validation_status']}    point {index + 1}/133",
                fontsize=11,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def validate_pdf(path: Path, spectral: pd.DataFrame) -> dict:
    reader = PdfReader(str(path))
    if len(reader.pages) != 134:
        raise RuntimeError(f"Expected 134 pages; observed {len(reader.pages)}")
    text = "\n".join(page.extract_text() or "" for page in reader.pages[1:])
    pattern = re.compile(r"\bM\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*alpha\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.I)
    observed = {key(m, a) for m, a in pattern.findall(text)}
    expected = set(spectral["key"])
    if observed != expected:
        raise RuntimeError(f"PDF labels mismatch; missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    return {"pages": 134, "unique_point_labels": 133}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("/tmp/supersonic_primary133_build_report.json"))
    args = parser.parse_args()

    output = args.output_dir.resolve()
    overlay_pdf = output / "ci/blumen_ci_with_primary_133pts.pdf"
    overlay_png = output / "ci/blumen_ci_with_primary_133pts.png"
    modes_pdf = output / "modes/supersonic_primary_133pts_modes_p_rho_u_v.pdf"

    spectral = load_spectral()
    modal = load_modal(set(spectral["key"]))
    overlay_report = build_overlay(spectral, overlay_pdf, overlay_png)
    build_modes(spectral, modal, modes_pdf)
    pdf_report = validate_pdf(modes_pdf, spectral)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "point_count": 133,
        "status_counts": EXPECTED_STATUS_COUNTS,
        "mach_counts": {str(k): v for k, v in EXPECTED_MACH_COUNTS.items()},
        "overlay": overlay_report,
        "modal_pdf": pdf_report,
        "outputs": [str(overlay_pdf), str(overlay_png), str(modes_pdf)],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    print("Points: 133")
    print("Status counts:", EXPECTED_STATUS_COUNTS)
    print("Mach counts:", EXPECTED_MACH_COUNTS)
    print("Wrote:", overlay_pdf)
    print("Wrote:", overlay_png)
    print("Wrote:", modes_pdf)
    print("PDF pages: 134")
    print("PDF unique point labels: 133")
    print("Wrote:", args.report)
    print("PRIMARY133 ASSET BUILD: PASS")


if __name__ == "__main__":
    main()
