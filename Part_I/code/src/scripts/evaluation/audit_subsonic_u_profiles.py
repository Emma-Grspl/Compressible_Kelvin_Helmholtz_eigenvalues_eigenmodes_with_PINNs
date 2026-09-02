#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

METHODS = {
    "hybrid_switch_a040": Path("assets/pinn_subsonic/hybrid_pressure_pq_M050_a030_a070_switch_a040"),
    "mini2d_pq_discrete": Path("assets/pinn_subsonic/mini2d_pq_firstorder_M050_a030_a070_bootp_discrete"),
    "pq_detach_old": Path("assets/pinn_subsonic/permach_pq_bootp_M050_a010070_cilearned_nodetach"),
    "pq_true_nodetach": Path("assets/pinn_subsonic/permach_pq_bootp_M050_a010070_TRUE_cilearned_nodetach"),
    "pressure_stage1q": Path("assets/pinn_subsonic/stage1quater_pressure_path1500_xlim15"),
}

ALPHAS = [0.30, 0.50, 0.55, 0.70]
OUT = Path("assets/pinn_subsonic/u_reconstruction_diagnostics_M050")
OUT.mkdir(parents=True, exist_ok=True)

def norm_name(s):
    return s.lower().replace("-", "_").replace(" ", "_")

def infer_alpha_from_path(p):
    s = str(p)
    patterns = [
        r"alpha[_=]?([0-9]+(?:\.[0-9]+)?)",
        r"_a([0-9]{3,5})(?:_|\.|$)",
        r"a([0-9]{3,5})",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if not m:
            continue
        raw = m.group(1)
        if "." in raw:
            return float(raw)
        if len(raw) == 3:
            return int(raw) / 100.0
        if len(raw) == 4:
            return int(raw) / 1000.0
        if len(raw) == 5:
            return int(raw) / 10000.0
    return None

def find_col(cols, must_include, any_include=None, real=True):
    ncols = [(c, norm_name(c)) for c in cols]
    re_keys = ["real", "_re", "re_"] if real else ["imag", "_im", "im_"]

    candidates = []
    for c, nc in ncols:
        if not all(k in nc for k in must_include):
            continue
        if any_include and not any(k in nc for k in any_include):
            continue
        if real:
            if not ("real" in nc or nc.endswith("_re") or "_re_" in nc or nc == must_include[0]):
                continue
        else:
            if not ("imag" in nc or nc.endswith("_im") or "_im_" in nc):
                continue
        candidates.append(c)

    return candidates[0] if candidates else None

def detect_complex_pair(df, var, role):
    cols = list(df.columns)

    if role == "ref":
        role_words = ["ref", "classic", "target", "true", "reference"]
    else:
        role_words = ["pred", "pinn", "model", "net"]

    # Common exact names first.
    exact_pairs = [
        (f"{var}_{role}_real", f"{var}_{role}_imag"),
        (f"{var}_{role}_re", f"{var}_{role}_im"),
        (f"{var}_real_{role}", f"{var}_imag_{role}"),
        (f"{role}_{var}_real", f"{role}_{var}_imag"),
        (f"{role}_{var}_re", f"{role}_{var}_im"),
        (f"{var}_PINN_real", f"{var}_PINN_imag") if role == "pred" else (None, None),
        (f"{var}_pinn_real", f"{var}_pinn_imag") if role == "pred" else (None, None),
        (f"{var}_ref_real", f"{var}_ref_imag") if role == "ref" else (None, None),
    ]

    lowmap = {norm_name(c): c for c in cols}
    for a, b in exact_pairs:
        if a is None:
            continue
        aa, bb = norm_name(a), norm_name(b)
        if aa in lowmap and bb in lowmap:
            return lowmap[aa], lowmap[bb]

    re_col = find_col(cols, [var], role_words, real=True)
    im_col = find_col(cols, [var], role_words, real=False)

    if re_col and im_col:
        return re_col, im_col

    return None, None

def load_profile_csv(path, alpha_target):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None

    cols = list(df.columns)
    low = {norm_name(c): c for c in cols}

    y_col = None
    for cand in ["y", "yy"]:
        if cand in low:
            y_col = low[cand]
            break
    if y_col is None:
        return None

    alpha_col = None
    for cand in ["alpha", "a"]:
        if cand in low:
            alpha_col = low[cand]
            break

    mach_col = None
    for cand in ["mach", "m"]:
        if cand in low:
            mach_col = low[cand]
            break

    d = df.copy()

    if mach_col is not None:
        try:
            d = d[np.isclose(d[mach_col].astype(float), 0.5, atol=5e-4)]
        except Exception:
            pass

    if alpha_col is not None:
        try:
            d = d[np.isclose(d[alpha_col].astype(float), alpha_target, atol=5e-4)]
        except Exception:
            pass
    else:
        a_path = infer_alpha_from_path(path)
        if a_path is None or not np.isclose(a_path, alpha_target, atol=5e-4):
            return None

    if len(d) < 10:
        return None

    ur_re, ur_im = detect_complex_pair(d, "u", "ref")
    up_re, up_im = detect_complex_pair(d, "u", "pred")

    if ur_re is None or up_re is None:
        return None

    y = d[y_col].to_numpy(float)
    order = np.argsort(y)
    y = y[order]

    if ur_im is not None:
        u_ref = d[ur_re].to_numpy(float) + 1j * d[ur_im].to_numpy(float)
    else:
        u_ref = d[ur_re].to_numpy(float).astype(complex)

    if up_im is not None:
        u_pred = d[up_re].to_numpy(float) + 1j * d[up_im].to_numpy(float)
    else:
        u_pred = d[up_re].to_numpy(float).astype(complex)

    u_ref = u_ref[order]
    u_pred = u_pred[order]

    return {
        "path": path,
        "y": y,
        "u_ref": u_ref,
        "u_pred": u_pred,
        "cols": (ur_re, ur_im, up_re, up_im),
    }

def find_best_profile(method_root, alpha):
    candidates = []
    for p in sorted(method_root.rglob("*.csv")):
        if p.name in ["diagnostics_summary.csv", "comparison_metrics.csv", "comparison_scores.csv"]:
            continue
        prof = load_profile_csv(p, alpha)
        if prof is not None:
            candidates.append(prof)

    if not candidates:
        return None

    # Prefer densest profile.
    candidates.sort(key=lambda d: len(d["y"]), reverse=True)
    return candidates[0]

def rel_l2(y, a, b, mask):
    if mask.sum() < 3:
        return np.nan
    yy = y[mask]
    num = np.trapz(np.abs(a[mask] - b[mask]) ** 2, yy)
    den = np.trapz(np.abs(b[mask]) ** 2, yy)
    return float(np.sqrt(num / max(den, 1e-300)))

def complex_align(pred, ref, mask):
    if mask.sum() < 3:
        return 1.0 + 0j
    p = pred[mask]
    r = ref[mask]
    den = np.vdot(p, p)
    if abs(den) < 1e-300:
        return 1.0 + 0j
    return np.vdot(p, r) / den

def plot_one(method, alpha, prof, pdf):
    y = prof["y"]
    u_ref = prof["u_ref"]
    u_pred_raw = prof["u_pred"]

    central = np.abs(y) <= 15
    scale = complex_align(u_pred_raw, u_ref, central)
    u_pred = scale * u_pred_raw

    masks = {
        "full": np.isfinite(y),
        "central_|y|<=15": np.abs(y) <= 15,
        "inner_|y|<=30": np.abs(y) <= 30,
        "left_y<-15": y < -15,
        "right_y>15": y > 15,
    }

    row = {
        "method": method,
        "alpha": alpha,
        "source_csv": str(prof["path"]),
        "n": len(y),
        "align_scale_real": scale.real,
        "align_scale_imag": scale.imag,
        "align_scale_abs": abs(scale),
        "max_abs_u_ref": float(np.nanmax(np.abs(u_ref))),
        "max_abs_u_pred_raw": float(np.nanmax(np.abs(u_pred_raw))),
        "max_abs_u_pred_aligned": float(np.nanmax(np.abs(u_pred))),
    }

    for name, mask in masks.items():
        row[f"u_rel_{name}"] = rel_l2(y, u_pred, u_ref, mask)

    # Plot real/imag in central window.
    m = np.abs(y) <= 30
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(y[m], u_ref.real[m], label="Re u_ref")
    ax.plot(y[m], u_pred.real[m], "--", label="Re u_PINN aligned")
    ax.plot(y[m], u_ref.imag[m], label="Im u_ref")
    ax.plot(y[m], u_pred.imag[m], "--", label="Im u_PINN aligned")
    ax.set_title(f"{method}, M=0.5, alpha={alpha:.3f}: u real/imag")
    ax.set_xlabel("y")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"u_profile_{method}_a{alpha:.3f}.png", dpi=220, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Amplitude.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(y, np.abs(u_ref) + 1e-300, label="|u_ref|")
    ax.semilogy(y, np.abs(u_pred) + 1e-300, "--", label="|u_PINN aligned|")
    ax.axvline(-15, linestyle=":", linewidth=1)
    ax.axvline(15, linestyle=":", linewidth=1)
    ax.set_title(f"{method}, M=0.5, alpha={alpha:.3f}: |u|")
    ax.set_xlabel("y")
    ax.set_ylabel("|u|")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"u_abs_{method}_a{alpha:.3f}.png", dpi=220, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Error.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(y, np.abs(u_pred - u_ref) + 1e-300, label="|u_PINN - u_ref|")
    ax.semilogy(y, np.abs(u_ref) + 1e-300, label="|u_ref|", alpha=0.7)
    ax.axvline(-15, linestyle=":", linewidth=1)
    ax.axvline(15, linestyle=":", linewidth=1)
    ax.set_title(f"{method}, M=0.5, alpha={alpha:.3f}: erreur absolue u")
    ax.set_xlabel("y")
    ax.set_ylabel("amplitude")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"u_error_{method}_a{alpha:.3f}.png", dpi=220, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    return row

def main():
    rows = []

    with PdfPages(OUT / "M050_u_reconstruction_profiles.pdf") as pdf:
        for method, root in METHODS.items():
            if not root.exists():
                print("[missing root]", method, root)
                continue

            for alpha in ALPHAS:
                prof = find_best_profile(root, alpha)
                if prof is None:
                    print("[no profile]", method, alpha)
                    continue

                print("[profile]", method, alpha, prof["path"], "n=", len(prof["y"]), "cols=", prof["cols"])
                rows.append(plot_one(method, alpha, prof, pdf))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "u_reconstruction_profile_metrics.csv", index=False)

    print("\n[OK] wrote", OUT / "M050_u_reconstruction_profiles.pdf")
    print("[OK] wrote", OUT / "u_reconstruction_profile_metrics.csv")
    if len(metrics):
        cols = [
            "method", "alpha", "n",
            "u_rel_central_|y|<=15",
            "u_rel_inner_|y|<=30",
            "u_rel_full",
            "align_scale_abs",
            "max_abs_u_ref",
            "max_abs_u_pred_raw",
            "max_abs_u_pred_aligned",
            "source_csv",
        ]
        print(metrics[[c for c in cols if c in metrics.columns]].to_string(index=False))

if __name__ == "__main__":
    main()
