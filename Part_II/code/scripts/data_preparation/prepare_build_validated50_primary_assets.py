#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from pypdf import PdfReader


REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "classic_supersonic"

SPECTRAL = (
    PACKAGE
    / "data/spectral/"
    "supersonic_modal_spectral_validated_50pts.csv"
)

CANONICAL_DIR = (
    PACKAGE
    / "reproducibility/results/final_50pts/"
    "canonical_validated50"
)

MODAL = (
    CANONICAL_DIR
    / "supersonic_modal_fields_p_rho_u_v_validated50.parquet"
)

OVERLAY = (
    CANONICAL_DIR
    / "blumen_ci_isolines_overlay_validated50.png"
)

REPORT = (
    CANONICAL_DIR
    / "primary_assets_build_report.json"
)

FIELDS = [
    ("p", "Pressure p"),
    ("rho", "Density rho"),
    ("u", "Streamwise velocity u"),
    ("v", "Transverse velocity v"),
]

EXPECTED_MACH_COUNTS = {
    1.10: 2,
    1.20: 6,
    1.25: 1,
    1.30: 9,
    1.33: 1,
    1.40: 11,
    1.50: 8,
    1.60: 2,
    1.70: 4,
    1.80: 3,
    1.90: 3,
}


plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def point_key(
    mach: object,
    alpha: object,
) -> tuple[float, float]:
    return (
        round(float(mach), 6),
        round(float(alpha), 6),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_spectral() -> pd.DataFrame:
    if not SPECTRAL.is_file():
        raise FileNotFoundError(SPECTRAL)

    frame = pd.read_csv(SPECTRAL)

    required = {
        "Mach",
        "alpha",
        "cr",
        "ci",
    }

    missing = required - set(frame.columns)

    if missing:
        raise KeyError(
            f"Missing spectral columns: {sorted(missing)}"
        )

    for column in [
        "Mach",
        "alpha",
        "cr",
        "ci",
        "omega_i",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    if "omega_i" not in frame.columns:
        frame["omega_i"] = (
            frame["alpha"]
            * frame["ci"]
        )

    frame = (
        frame.dropna(
            subset=[
                "Mach",
                "alpha",
                "cr",
                "ci",
            ]
        )
        .drop_duplicates(
            [
                "Mach",
                "alpha",
            ]
        )
        .sort_values(
            [
                "Mach",
                "alpha",
            ]
        )
        .reset_index(drop=True)
    )

    if len(frame) != 50:
        raise RuntimeError(
            f"Expected 50 spectral points; observed {len(frame)}."
        )

    observed_counts = {
        round(float(mach), 2): int(count)
        for mach, count in (
            frame.groupby("Mach")
            .size()
            .items()
        )
    }

    if observed_counts != EXPECTED_MACH_COUNTS:
        raise RuntimeError(
            "Unexpected Mach distribution.\n"
            f"Observed: {observed_counts}\n"
            f"Expected: {EXPECTED_MACH_COUNTS}"
        )

    frame["key"] = [
        point_key(mach, alpha)
        for mach, alpha in zip(
            frame["Mach"],
            frame["alpha"],
        )
    ]

    return frame


def load_modal(
    expected_keys: set[tuple[float, float]],
) -> pd.DataFrame:
    if not MODAL.is_file():
        raise FileNotFoundError(MODAL)

    frame = pd.read_parquet(MODAL)

    required = {
        "Mach",
        "alpha",
        "y",
        "p_real",
        "p_imag",
        "rho_real",
        "rho_imag",
        "u_real",
        "u_imag",
        "v_real",
        "v_imag",
        "source_kind",
    }

    missing = required - set(frame.columns)

    if missing:
        raise KeyError(
            f"Missing modal columns: {sorted(missing)}"
        )

    numeric_columns = [
        "Mach",
        "alpha",
        "y",
        "p_real",
        "p_imag",
        "rho_real",
        "rho_imag",
        "u_real",
        "u_imag",
        "v_real",
        "v_imag",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=[
            "Mach",
            "alpha",
            "y",
        ]
    )

    frame["key"] = [
        point_key(mach, alpha)
        for mach, alpha in zip(
            frame["Mach"],
            frame["alpha"],
        )
    ]

    observed_keys = set(frame["key"])

    missing_keys = sorted(
        expected_keys - observed_keys
    )

    extra_keys = sorted(
        observed_keys - expected_keys
    )

    if missing_keys or extra_keys:
        raise RuntimeError(
            "Modal/spectral key mismatch.\n"
            f"Missing: {missing_keys}\n"
            f"Extra: {extra_keys}"
        )

    duplicated = frame.duplicated(
        [
            "Mach",
            "alpha",
            "y",
        ],
        keep=False,
    )

    if duplicated.any():
        raise RuntimeError(
            "Duplicate (Mach, alpha, y) rows "
            "exist in the canonical modal package."
        )

    return frame


def adaptive_limit(
    y: np.ndarray,
    values: np.ndarray,
) -> float:
    full_limit = float(
        np.nanmax(
            np.abs(y)
        )
    )

    if (
        not np.isfinite(full_limit)
        or full_limit <= 0.0
    ):
        return 1.0

    amplitude = np.abs(values)

    peak = float(
        np.nanmax(amplitude)
    )

    if (
        not np.isfinite(peak)
        or peak <= 0.0
    ):
        return full_limit

    normalized = amplitude / peak

    significant = (
        np.isfinite(normalized)
        & (normalized >= 1.0e-3)
    )

    if not significant.any():
        return full_limit

    significant_limit = float(
        np.nanmax(
            np.abs(
                y[significant]
            )
        )
    )

    lower_bound = min(
        10.0,
        full_limit,
    )

    return min(
        full_limit,
        max(
            lower_bound,
            1.08 * significant_limit,
        ),
    )


def plot_field(
    axis: plt.Axes,
    point_modal: pd.DataFrame,
    field: str,
    title: str,
) -> None:
    y = point_modal[
        "y"
    ].to_numpy(dtype=float)

    real = point_modal[
        f"{field}_real"
    ].to_numpy(dtype=float)

    imag = point_modal[
        f"{field}_imag"
    ].to_numpy(dtype=float)

    finite = (
        np.isfinite(y)
        & np.isfinite(real)
        & np.isfinite(imag)
    )

    y = y[finite]
    values = (
        real[finite]
        + 1j * imag[finite]
    )

    order = np.argsort(y)
    y = y[order]
    values = values[order]

    if len(y) == 0:
        raise RuntimeError(
            f"No finite values for field {field}."
        )

    peak = float(
        np.nanmax(
            np.abs(values)
        )
    )

    if (
        not np.isfinite(peak)
        or peak <= 0.0
    ):
        raise RuntimeError(
            f"Invalid normalization peak for field {field}."
        )

    normalized_real = (
        values.real / peak
    )

    normalized_amplitude = (
        np.abs(values) / peak
    )

    limit = adaptive_limit(
        y,
        values,
    )

    window = (
        np.abs(y) <= limit
    )

    axis.plot(
        y[window],
        normalized_real[window],
        label=f"Re({field}) / max|{field}|",
        linewidth=1.0,
    )

    axis.plot(
        y[window],
        normalized_amplitude[window],
        linestyle="--",
        label=f"|{field}| / max|{field}|",
        linewidth=1.0,
    )

    axis.axhline(
        0.0,
        linewidth=0.7,
    )

    axis.set_xlim(
        -limit,
        limit,
    )

    axis.set_ylim(
        -1.08,
        1.08,
    )

    axis.set_xlabel("y")
    axis.set_ylabel("normalized mode")
    axis.set_title(title)

    axis.grid(
        True,
        alpha=0.22,
    )

    axis.legend(
        frameon=False,
        loc="best",
    )


def build_modes_pdf(
    spectral: pd.DataFrame,
    modal: pd.DataFrame,
    output: Path,
) -> None:
    lookup = {
        row["key"]: row
        for _, row in spectral.iterrows()
    }

    groups = {
        key: group.sort_values("y")
        for key, group in modal.groupby(
            "key",
            sort=False,
        )
    }

    source_counts = (
        modal[
            [
                "key",
                "source_kind",
            ]
        ]
        .drop_duplicates()
        ["source_kind"]
        .value_counts()
        .to_dict()
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "Title": (
            "Classical supersonic validated "
            "50-point modal atlas"
        ),
        "Author": "Emma Grospellier",
        "Subject": (
            "44 canonical raw-confirmed base points "
            "and 6 exact-shooting high-Mach points"
        ),
        "Keywords": (
            "Kelvin-Helmholtz supersonic "
            "modal spectral reference"
        ),
    }

    with PdfPages(
        output,
        metadata=metadata,
    ) as pdf:
        figure = plt.figure(
            figsize=(11.69, 8.27),
        )

        figure.text(
            0.5,
            0.88,
            "Classical supersonic validated "
            "modal-spectral reference",
            ha="center",
            va="center",
            fontsize=18,
        )

        figure.text(
            0.5,
            0.81,
            "50 points: 44 canonical raw-confirmed "
            "base points + 6 exact-shooting "
            "high-Mach points",
            ha="center",
            va="center",
            fontsize=11,
        )

        counts_text = "\n".join(
            f"M={mach:.2f}: {count} points"
            for mach, count
            in EXPECTED_MACH_COUNTS.items()
        )

        sources_text = "\n".join(
            f"{source}: {count} points"
            for source, count
            in sorted(
                source_counts.items()
            )
        )

        figure.text(
            0.24,
            0.68,
            "Distribution by Mach\n\n"
            + counts_text,
            ha="left",
            va="top",
            fontsize=10,
        )

        figure.text(
            0.58,
            0.68,
            "Modal sources\n\n"
            + sources_text,
            ha="left",
            va="top",
            fontsize=10,
        )

        figure.text(
            0.5,
            0.14,
            "Each field is normalized by its own "
            "maximum amplitude. Solid: real part. "
            "Dashed: modulus.\n"
            "The displayed y-window retains all "
            "locations where the amplitude is at least "
            "1e-3 of the field maximum.",
            ha="center",
            va="center",
            fontsize=9,
        )

        pdf.savefig(
            figure,
            bbox_inches="tight",
        )

        plt.close(figure)

        total = len(spectral)

        for index, row in spectral.iterrows():
            key = row["key"]
            point_modal = groups[key]

            source_values = (
                point_modal[
                    "source_kind"
                ]
                .drop_duplicates()
                .tolist()
            )

            if len(source_values) != 1:
                raise RuntimeError(
                    f"Several modal sources for {key}: "
                    f"{source_values}"
                )

            source = source_values[0]

            figure, axes = plt.subplots(
                2,
                2,
                figsize=(11.69, 8.27),
                constrained_layout=True,
            )

            for axis, (field, title) in zip(
                axes.flat,
                FIELDS,
            ):
                plot_field(
                    axis,
                    point_modal,
                    field,
                    title,
                )

            figure.suptitle(
                (
                    f"M={row['Mach']:.2f}, "
                    f"alpha={row['alpha']:.6f}    "
                    f"cr={row['cr']:.8f}, "
                    f"ci={row['ci']:.8f}, "
                    f"omega_i={row['omega_i']:.8e}\n"
                    f"modal source: {source}    "
                    f"point {index + 1}/{total}"
                ),
                fontsize=12,
            )

            pdf.savefig(
                figure,
                bbox_inches="tight",
            )

            plt.close(figure)


def validate_pdf(
    pdf_path: Path,
    spectral: pd.DataFrame,
) -> dict:
    reader = PdfReader(
        str(pdf_path)
    )

    expected_pages = 51

    if len(reader.pages) != expected_pages:
        raise RuntimeError(
            f"Expected {expected_pages} pages; "
            f"observed {len(reader.pages)}."
        )

    text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages[1:]
    )

    pattern = re.compile(
        r"\bM\s*=\s*"
        r"([0-9]+(?:\.[0-9]+)?)"
        r"\s*,\s*"
        r"alpha\s*=\s*"
        r"([0-9]+(?:\.[0-9]+)?)",
        flags=re.IGNORECASE,
    )

    observed = {
        point_key(mach, alpha)
        for mach, alpha
        in pattern.findall(text)
    }

    expected = set(
        spectral["key"]
    )

    if observed != expected:
        raise RuntimeError(
            "PDF point-label validation failed.\n"
            f"Missing: {sorted(expected - observed)}\n"
            f"Extra: {sorted(observed - expected)}"
        )

    return {
        "pages": len(reader.pages),
        "unique_point_labels": len(observed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    arguments = parser.parse_args()

    output_root = (
        arguments.output_dir.resolve()
    )

    ci_directory = (
        output_root / "ci"
    )

    modes_directory = (
        output_root / "modes"
    )

    ci_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    modes_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        SPECTRAL,
        MODAL,
        OVERLAY,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    spectral = load_spectral()

    modal = load_modal(
        set(spectral["key"])
    )

    overlay_output = (
        ci_directory
        / "blumen_ci_isolines_overlay_validated50.png"
    )

    modes_output = (
        modes_directory
        / "supersonic_modes_p_rho_u_v_validated50.pdf"
    )

    shutil.copy2(
        OVERLAY,
        overlay_output,
    )

    build_modes_pdf(
        spectral,
        modal,
        modes_output,
    )

    pdf_validation = validate_pdf(
        modes_output,
        spectral,
    )

    with Image.open(
        overlay_output
    ) as image:
        overlay_dimensions = [
            image.width,
            image.height,
        ]

    source_counts = (
        modal[
            [
                "key",
                "source_kind",
            ]
        ]
        .drop_duplicates()
        ["source_kind"]
        .value_counts()
        .to_dict()
    )

    report = {
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),
        "spectral_selection": str(
            SPECTRAL.relative_to(REPO)
        ),
        "modal_source": str(
            MODAL.relative_to(REPO)
        ),
        "point_count": len(spectral),
        "source_counts": source_counts,
        "pdf_validation": pdf_validation,
        "overlay_dimensions": (
            overlay_dimensions
        ),
        "outputs": {
            "overlay": {
                "path": str(overlay_output),
                "sha256": sha256(
                    overlay_output
                ),
            },
            "modes_pdf": {
                "path": str(modes_output),
                "sha256": sha256(
                    modes_output
                ),
            },
        },
    }

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n"
    )

    print("Points:", len(spectral))
    print("Source counts:", source_counts)
    print("Wrote:", overlay_output)
    print("Wrote:", modes_output)
    print("PDF pages:", pdf_validation["pages"])
    print(
        "PDF unique labels:",
        pdf_validation[
            "unique_point_labels"
        ],
    )
    print("Wrote:", REPORT)
    print()
    print(
        "VALIDATED50 PRIMARY ASSETS BUILD: PASS"
    )


if __name__ == "__main__":
    main()
