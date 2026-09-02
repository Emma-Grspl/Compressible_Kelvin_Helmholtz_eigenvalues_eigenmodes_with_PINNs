#!/usr/bin/env python
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


SRC = Path("assets/classic_supersonic/final_sparse_PINN_reference_v2_CONFIRMED")
OUT = Path("assets/classic_supersonic/final_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1")
OUT.mkdir(parents=True, exist_ok=True)

SPEC_IN = SRC / "supersonic_sparse_PINN_reference_v2_CONFIRMED_spectral.csv"
FIELDS_IN = SRC / "supersonic_sparse_PINN_reference_v2_CONFIRMED_modal_fields.csv"

SPEC_OUT = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1_spectral.csv"
FIELDS_OUT = OUT / "supersonic_sparse_PINN_reference_v2_CONFIRMED_tail_polished_v1_modal_fields.csv"
AUDIT_OUT = OUT / "tail_polish_audit.csv"
PDF_OUT = OUT / "tail_polish_raw_vs_polished_left_tail_review.pdf"
SUMMARY_OUT = OUT / "summary_tail_polished_v1.json"
SHA_OUT = OUT / "SHA256SUMS.txt"

FIELDS = ["p", "rho", "u", "v"]

# On ne touche que la queue gauche faible.
JOIN_FRAC = 0.05       # y_join là où |p| atteint 5% du max, côté gauche
FIT_LOW = 0.003        # fit entre 0.3% et 8% du max
FIT_HIGH = 0.08
MIN_FIT_POINTS = 40


def norm_cols(df):
    if "Mach" not in df.columns and "M" in df.columns:
        df = df.rename(columns={"M": "Mach"})
    df["Mach"] = pd.to_numeric(df["Mach"], errors="coerce")
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    return df


def zfield(df, name):
    return (
        pd.to_numeric(df[f"{name}_real"], errors="coerce").to_numpy(float)
        + 1j * pd.to_numeric(df[f"{name}_imag"], errors="coerce").to_numpy(float)
    )


def set_zfield(df, name, z):
    df[f"{name}_real"] = np.real(z)
    df[f"{name}_imag"] = np.imag(z)


def fit_tail_lambda(y, z, peak):
    amp = np.abs(z)
    mask = (
        np.isfinite(y)
        & np.isfinite(np.real(z))
        & np.isfinite(np.imag(z))
        & (y < 0.0)
        & (amp >= FIT_LOW * peak)
        & (amp <= FIT_HIGH * peak)
    )

    if mask.sum() < MIN_FIT_POINTS:
        mask = (
            np.isfinite(y)
            & np.isfinite(np.real(z))
            & np.isfinite(np.imag(z))
            & (y < 0.0)
            & (amp >= 0.001 * peak)
            & (amp <= 0.12 * peak)
        )

    if mask.sum() < MIN_FIT_POINTS:
        return None

    yy = y[mask]
    zz = z[mask]
    aa = np.abs(zz)

    good = aa > 0
    yy = yy[good]
    zz = zz[good]
    aa = aa[good]

    if len(yy) < MIN_FIT_POINTS:
        return None

    log_amp = np.log(aa)
    phase = np.unwrap(np.angle(zz))

    slope_amp, intercept_amp = np.polyfit(yy, log_amp, 1)
    slope_phase, intercept_phase = np.polyfit(yy, phase, 1)

    pred = intercept_amp + slope_amp * yy
    denom = np.linalg.norm(log_amp - log_amp.mean())
    rel_resid = float(np.linalg.norm(log_amp - pred) / denom) if denom > 0 else 0.0

    lam = complex(float(slope_amp), float(slope_phase))

    if not np.isfinite(lam.real) or not np.isfinite(lam.imag):
        return None

    # Pour une queue gauche, l'amplitude doit croître quand y va vers 0.
    if lam.real <= 0:
        return None

    return lam, rel_resid, int(len(yy)), float(yy.min()), float(yy.max())


def polish_one_point(g):
    g = g.sort_values("y").copy()
    y = pd.to_numeric(g["y"], errors="coerce").to_numpy(float)

    zp = zfield(g, "p")
    amp = np.abs(zp)
    peak = float(np.nanmax(amp))

    audit = {
        "tail_polish_left_applied": False,
        "tail_polish_reason": "",
        "tail_polish_lambda_real": np.nan,
        "tail_polish_lambda_imag": np.nan,
        "tail_polish_fit_rel_resid": np.nan,
        "tail_polish_fit_n": 0,
        "tail_polish_y_join": np.nan,
        "tail_polish_n_replaced": 0,
    }

    if not np.isfinite(peak) or peak <= 0:
        audit["tail_polish_reason"] = "bad_peak"
        return g, audit

    left = (y < 0.0) & np.isfinite(y) & np.isfinite(amp)
    if not np.any(left):
        audit["tail_polish_reason"] = "no_left_side"
        return g, audit

    crossing = np.where(left & (amp >= JOIN_FRAC * peak))[0]
    if len(crossing) == 0:
        audit["tail_polish_reason"] = "no_join_crossing"
        return g, audit

    j = int(crossing[0])
    yj = float(y[j])

    if yj >= -1e-10:
        audit["tail_polish_reason"] = "join_too_close_to_center"
        return g, audit

    fit = fit_tail_lambda(y, zp, peak)
    if fit is None:
        audit["tail_polish_reason"] = "fit_failed"
        return g, audit

    lam, resid, nfit, yfit_min, yfit_max = fit

    replace = y < yj
    if replace.sum() < 5:
        audit["tail_polish_reason"] = "too_few_replaced"
        return g, audit

    for name in FIELDS:
        z = zfield(g, name)
        zj = z[j]
        z_new = z.copy()
        z_new[replace] = zj * np.exp(lam * (y[replace] - yj))
        set_zfield(g, name, z_new)

    audit.update({
        "tail_polish_left_applied": True,
        "tail_polish_reason": "ok",
        "tail_polish_lambda_real": float(lam.real),
        "tail_polish_lambda_imag": float(lam.imag),
        "tail_polish_fit_rel_resid": float(resid),
        "tail_polish_fit_n": int(nfit),
        "tail_polish_fit_y_min": float(yfit_min),
        "tail_polish_fit_y_max": float(yfit_max),
        "tail_polish_y_join": float(yj),
        "tail_polish_n_replaced": int(replace.sum()),
    })

    return g, audit


def point_match(df, M, a):
    return df[
        np.isclose(df["Mach"].astype(float), M, atol=1e-10)
        & np.isclose(df["alpha"].astype(float), a, atol=1e-10)
    ].copy()


def xlim_left_tail(y, z):
    amp = np.abs(z)
    peak = np.nanmax(amp)
    if not np.isfinite(peak) or peak <= 0:
        return np.nanmin(y), 0.0

    mask = (y < 0) & (amp >= 0.003 * peak)
    if not np.any(mask):
        mask = y < 0
    left = float(np.nanmin(y[mask]))
    return left, 0.0


def make_pdf(spec, raw_fields, polished_fields, audit):
    with PdfPages(PDF_OUT) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), dpi=160)
        ax = fig.add_subplot(111)
        ax.axis("off")

        lines = [
            "Tail polish v1 - raw vs polished left tail",
            "",
            f"Points: {len(spec)}",
            f"Rows raw: {len(raw_fields)}",
            f"Rows polished: {len(polished_fields)}",
            "",
            "Method:",
            "  Fit complex exponential p_tail(y)=A exp(lambda y) in weak left-tail overlap.",
            "  Replace y < y_join using same lambda for p, rho, u, v.",
            "  Keep core unchanged.",
            "",
            "This is an export-level tail regularization. Spectral cr/ci are unchanged.",
            "",
            "Applied counts:",
            str(audit['tail_polish_left_applied'].value_counts(dropna=False).to_dict()),
        ]
        ax.text(0.04, 0.96, "\n".join(lines), va="top", family="monospace", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for _, r in spec.sort_values(["Mach", "alpha"]).iterrows():
            M = float(r["Mach"])
            a = float(r["alpha"])

            raw = point_match(raw_fields, M, a).sort_values("y")
            pol = point_match(polished_fields, M, a).sort_values("y")

            if raw.empty or pol.empty:
                continue

            y = raw["y"].to_numpy(float)

            fig, axes = plt.subplots(4, 1, figsize=(11.69, 8.27), dpi=170)

            for ax, name in zip(axes, FIELDS):
                zr = zfield(raw, name)
                zp = zfield(pol, name)

                scale = np.nanmax(np.abs(zr))
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0

                ax.plot(y, np.real(zr) / scale, linewidth=0.55, label=f"raw Re({name})")
                ax.plot(y, np.real(zp) / scale, "--", linewidth=0.7, label=f"polished Re({name})")
                ax.plot(y, np.abs(zp) / scale, ":", linewidth=0.7, label=f"polished |{name}|")

                ax.axhline(0.0, color="black", linewidth=0.45, alpha=0.4)
                ax.set_xlim(*xlim_left_tail(y, zr))
                ax.set_ylim(-1.05, 1.08)
                ax.grid(True, alpha=0.22, linestyle=":")
                ax.set_title(name, fontsize=8)
                ax.set_xlabel("y", fontsize=8)
                ax.legend(fontsize=6, loc="upper left")

            ar = audit[
                np.isclose(audit["Mach"], M)
                & np.isclose(audit["alpha"], a)
            ]

            if len(ar):
                ar0 = ar.iloc[0]
                subtitle = (
                    f"applied={ar0['tail_polish_left_applied']}, "
                    f"lambda={ar0['tail_polish_lambda_real']:.3e}"
                    f"+{ar0['tail_polish_lambda_imag']:.3e}i, "
                    f"y_join={ar0['tail_polish_y_join']:.3g}, "
                    f"resid={ar0['tail_polish_fit_rel_resid']:.2e}"
                )
            else:
                subtitle = ""

            fig.suptitle(
                f"M={M:.2f}, alpha={a:.5f}, cr={float(r.get('cr', np.nan)):.6g}, "
                f"ci={float(r.get('ci', np.nan)):.6g}\n{subtitle}",
                fontsize=9,
            )
            fig.tight_layout(rect=[0, 0.02, 1, 0.91])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


spec = norm_cols(pd.read_csv(SPEC_IN))
fields = norm_cols(pd.read_csv(FIELDS_IN, low_memory=False))
fields["y"] = pd.to_numeric(fields["y"], errors="coerce")
fields = fields.dropna(subset=["Mach", "alpha", "y"]).sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)

parts = []
audit_rows = []

for (M, a), g in fields.groupby(["Mach", "alpha"], sort=True):
    pol, audit = polish_one_point(g)
    pol["tail_polish_version"] = "left_asymptotic_fit_v1"
    pol["tail_polish_left_applied"] = audit["tail_polish_left_applied"]
    pol["tail_polish_lambda_real"] = audit["tail_polish_lambda_real"]
    pol["tail_polish_lambda_imag"] = audit["tail_polish_lambda_imag"]
    pol["tail_polish_y_join"] = audit["tail_polish_y_join"]

    parts.append(pol)

    audit["Mach"] = float(M)
    audit["alpha"] = float(a)
    audit_rows.append(audit)

polished = pd.concat(parts, ignore_index=True).sort_values(["Mach", "alpha", "y"]).reset_index(drop=True)
audit = pd.DataFrame(audit_rows).sort_values(["Mach", "alpha"]).reset_index(drop=True)

spec = spec.copy()
spec["tail_polish_available"] = spec.apply(
    lambda r: bool(
        audit[
            np.isclose(audit["Mach"], float(r["Mach"]))
            & np.isclose(audit["alpha"], float(r["alpha"]))
        ]["tail_polish_left_applied"].any()
    ),
    axis=1,
)
spec["tail_polish_version"] = "left_asymptotic_fit_v1"
spec["tail_polish_note"] = (
    "Spectral values unchanged. Modal core unchanged. "
    "Weak left tail regularized by fitted complex exponential where available."
)

spec.to_csv(SPEC_OUT, index=False)
polished.to_csv(FIELDS_OUT, index=False)
audit.to_csv(AUDIT_OUT, index=False)

make_pdf(spec, fields, polished, audit)

summary = {
    "status": "tail_polished_v1_built",
    "input_dataset": str(SRC),
    "n_points": int(spec[["Mach", "alpha"]].drop_duplicates().shape[0]),
    "n_modal_rows": int(len(polished)),
    "n_tail_polished": int(audit["tail_polish_left_applied"].sum()),
    "n_tail_not_polished": int((~audit["tail_polish_left_applied"]).sum()),
    "method": {
        "side": "left",
        "join_fraction_of_peak_p": JOIN_FRAC,
        "fit_low_fraction_of_peak_p": FIT_LOW,
        "fit_high_fraction_of_peak_p": FIT_HIGH,
        "description": (
            "Fit complex exponential in weak left-tail overlap using p, then apply same lambda "
            "to p/rho/u/v for y < y_join. Core is unchanged."
        ),
    },
    "outputs": {
        "spectral": str(SPEC_OUT),
        "modal_fields": str(FIELDS_OUT),
        "audit": str(AUDIT_OUT),
        "review_pdf": str(PDF_OUT),
    },
    "important_note": (
        "This is a tail-regularized export for PINN/visual use. "
        "The raw confirmed dataset remains the primary frozen reference."
    ),
}

SUMMARY_OUT.write_text(json.dumps(summary, indent=2))

with SHA_OUT.open("w") as f:
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            f.write(f"{sha256_file(p)}  {p.name}\n")

print(json.dumps(summary, indent=2))
