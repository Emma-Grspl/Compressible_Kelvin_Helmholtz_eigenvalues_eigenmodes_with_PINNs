from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(".")
OUT = ROOT / "assets" / "supplementary_supersonic"
OUT.mkdir(parents=True, exist_ok=True)

AMB_M = 1.10
AMB_A = 0.09
RECOVERY_TOL = 1e-4


# ============================================================
# Helpers
# ============================================================

def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def bool_series(s):
    if s.dtype == bool:
        return s
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def compute_error(df):
    return np.hypot(
        df["shoot_cr"].astype(float)
        - df["cr_reference"].astype(float),
        df["shoot_ci"].astype(float)
        - df["ci_reference"].astype(float),
    )


def ambiguous_mask(df):
    mask = (
        np.isclose(
            df["Mach"].astype(float).to_numpy(),
            AMB_M,
            atol=1e-10,
        )
        &
        np.isclose(
            df["alpha"].astype(float).to_numpy(),
            AMB_A,
            atol=1e-10,
        )
    )
    return pd.Series(mask, index=df.index)


def summarize_run(name, path):
    df = pd.read_csv(require(path))

    if len(df) != 64:
        raise RuntimeError(
            f"{name}: expected 64 rows, got {len(df)}"
        )

    if df["benchmark_id"].nunique() != 64:
        raise RuntimeError(
            f"{name}: benchmark_id is not unique"
        )

    err = compute_error(df)

    completed = (
        df["shoot_status"]
        .astype(str)
        .str.upper()
        .eq("COMPLETED")
    )

    technical = (
        completed
        & bool_series(df["shoot_spectral_success"])
        & bool_series(df["shoot_mode_success"])
    )

    amb = ambiguous_mask(df)
    nonamb = ~amb

    recovered = (
        technical
        & nonamb
        & (err <= RECOVERY_TOL)
    )

    good_err = err[recovered].to_numpy()

    return {
        "name": name,
        "path": path,
        "df": df,
        "error": err.to_numpy(),
        "technical_mask": technical.to_numpy(),
        "ambiguous_mask": amb.to_numpy(),
        "nonambiguous_mask": nonamb.to_numpy(),
        "recovered_mask": recovered.to_numpy(),
        "technical": int(technical.sum()),
        "recovered": int(recovered.sum()),
        "direct_mean": float(
            df["pinn_spectral_error"].mean()
        ),
        "corrected_median": float(
            np.median(good_err)
        ),
        "corrected_p95": float(
            np.quantile(good_err, 0.95)
        ),
        "corrected_max": float(
            np.max(good_err)
        ),
    }


def save_both(fig, stem):
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )
    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    print("WROTE:", pdf)
    print("WROTE:", png)


# ============================================================
# 1. Copy already-existing Supplementary diagnostics
# ============================================================

existing = {
    "Fig_S2a_blumen_pointwise_delta_ci.pdf":
        ROOT
        / "assets/classic_supersonic/"
          "blumen_pointwise_classical_v1/"
          "blumen_pointwise_delta_ci_heatmap.pdf",

    "Fig_S2b_blumen_fixedM_delta_alpha.pdf":
        ROOT
        / "assets/classic_supersonic/"
          "blumen_true_classical_isolines_v1/"
          "blumen_true_classical_delta_alpha_heatmap.pdf",

    "Fig_S4a_spectral_integration_convergence.pdf":
        ROOT
        / "assets/classic_supersonic/"
          "dense_kappa_q_campaign_v1_CONVERGENCE_AUDIT/"
          "convergence_plots/"
          "spectral_integration_convergence.pdf",

    "Fig_S4b_spectral_box_convergence.pdf":
        ROOT
        / "assets/classic_supersonic/"
          "dense_kappa_q_campaign_v1_CONVERGENCE_AUDIT/"
          "convergence_plots/"
          "spectral_box_convergence.pdf",

    "Fig_S4c_matching_location_sensitivity.pdf":
        ROOT
        / "assets/classic_supersonic/"
          "dense_kappa_q_campaign_v1_CONVERGENCE_AUDIT/"
          "convergence_plots/"
          "matching_location_sensitivity.pdf",

    "Fig_S4d_modal_convergence.pdf":
        ROOT
        / "assets/classic_supersonic/"
          "dense_kappa_q_campaign_v1_CONVERGENCE_AUDIT/"
          "convergence_plots/"
          "modal_convergence.pdf",
}

for dst_name, src in existing.items():
    require(src)
    dst = OUT / dst_name
    shutil.copy2(src, dst)
    print("COPIED:", dst)


# ============================================================
# 2. Reviewer robustness: N60 vs N76 + anchors-only
# ============================================================

PRODUCTION_N76 = (
    ROOT
    / "assets/pinn_supersonic/"
      "atlas2d_v1_continuousM/N76/"
      "shooting_validation/"
      "shooting_validation_64.csv"
)

reviewer_paths = {
    "N60 seed 1":
        ROOT
        / "assets/pinn_supersonic/reviewer_runs/"
          "N60_fullc_seed1/shooting_V64/"
          "shooting_validation_64.csv",

    "N60 seed 2":
        ROOT
        / "assets/pinn_supersonic/reviewer_runs/"
          "N60_fullc_seed2/shooting_V64/"
          "shooting_validation_64.csv",

    "N60 seed 3":
        ROOT
        / "assets/pinn_supersonic/reviewer_runs/"
          "N60_fullc_seed3/shooting_V64/"
          "shooting_validation_64.csv",

    "N76 seed 1":
        PRODUCTION_N76,

    "N76 seed 2":
        ROOT
        / "assets/pinn_supersonic/reviewer_runs/"
          "N76_fullc_seed2/shooting_V64/"
          "shooting_validation_64.csv",

    "N76 seed 3":
        ROOT
        / "assets/pinn_supersonic/reviewer_runs/"
          "N76_fullc_seed3/shooting_V64/"
          "shooting_validation_64.csv",

    "Anchors-only":
        ROOT
        / "assets/pinn_supersonic/reviewer_runs/"
          "N76_anchors_only/shooting_V64/"
          "shooting_validation_64.csv",
}

runs = {
    name: summarize_run(name, path)
    for name, path in reviewer_paths.items()
}

summary_rows = []

for name, r in runs.items():
    summary_rows.append(
        {
            "run": name,
            "technical_success_64":
                r["technical"],
            "recovered_nonambiguous_63":
                r["recovered"],
            "direct_mean_complex_error":
                r["direct_mean"],
            "corrected_median":
                r["corrected_median"],
            "corrected_p95":
                r["corrected_p95"],
            "corrected_max":
                r["corrected_max"],
        }
    )

summary_df = pd.DataFrame(summary_rows)

summary_csv = (
    OUT / "Tab_S6_reviewer_robustness_summary.csv"
)
summary_df.to_csv(summary_csv, index=False)

print("WROTE:", summary_csv)
print()
print(summary_df.to_string(index=False))


# ------------------------------------------------------------
# Figure S6
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14.0, 4.6),
    constrained_layout=True,
)

# ------------------------------------------------------------
# S6a — recovered branch counts
# ------------------------------------------------------------

ax = axes[0]

seed_x = np.array([1, 2, 3], dtype=float)

n60 = np.array([
    runs[f"N60 seed {i}"]["recovered"]
    for i in (1, 2, 3)
])

n76 = np.array([
    runs[f"N76 seed {i}"]["recovered"]
    for i in (1, 2, 3)
])

ax.plot(
    seed_x,
    n60,
    marker="o",
    linewidth=1.8,
    label=r"$N=60$",
)

ax.plot(
    seed_x,
    n76,
    marker="s",
    linewidth=1.8,
    label=r"$N=76$",
)

ax.axhline(
    63,
    linestyle="--",
    linewidth=1.0,
)

for x, y in zip(seed_x, n60):
    ax.text(
        x,
        y - 0.12,
        f"{int(y)}/63",
        ha="center",
        va="top",
        fontsize=8,
    )

for x, y in zip(seed_x, n76):
    ax.text(
        x,
        y + 0.10,
        f"{int(y)}/63",
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax.set_xticks(seed_x)
ax.set_xlabel("Training realization")
ax.set_ylabel("Recovered target branches")
ax.set_ylim(60.8, 63.7)
ax.set_title("(a) Random-seed branch recovery")
ax.legend(frameon=False)


# ------------------------------------------------------------
# S6b — corrected error distributions
# ------------------------------------------------------------

ax = axes[1]

order = [
    "N60 seed 1",
    "N60 seed 2",
    "N60 seed 3",
    "N76 seed 1",
    "N76 seed 2",
    "N76 seed 3",
]

box_data = []

for name in order:
    r = runs[name]
    mask = r["recovered_mask"]
    box_data.append(
        r["error"][mask]
    )

bp = ax.boxplot(
    box_data,
    labels=[
        "N60\ns1",
        "N60\ns2",
        "N60\ns3",
        "N76\ns1",
        "N76\ns2",
        "N76\ns3",
    ],
    showfliers=True,
    widths=0.65,
)

ax.axhline(
    RECOVERY_TOL,
    linestyle="--",
    linewidth=1.0,
    label=r"recovery threshold $10^{-4}$",
)

ax.set_yscale("log")
ax.set_ylabel(
    r"Corrected complex error $|c-c^\star|$"
)
ax.set_title("(b) Error after successful recovery")
ax.tick_params(axis="x", labelsize=8)
ax.legend(
    frameon=False,
    fontsize=8,
    loc="upper right",
)


# ------------------------------------------------------------
# S6c — physics-informed vs anchors-only
# ------------------------------------------------------------

ax = axes[2]

physics = runs["N76 seed 1"]
anchors = runs["Anchors-only"]

phys_nonamb = physics["nonambiguous_mask"]
anch_nonamb = anchors["nonambiguous_mask"]

phys_rec = physics["recovered_mask"]
anch_rec = anchors["recovered_mask"]

compare_data = [
    physics["df"]
        .loc[
            phys_nonamb,
            "pinn_spectral_error"
        ]
        .astype(float)
        .to_numpy(),

    anchors["df"]
        .loc[
            anch_nonamb,
            "pinn_spectral_error"
        ]
        .astype(float)
        .to_numpy(),

    physics["error"][phys_rec],
    anchors["error"][anch_rec],
]

ax.boxplot(
    compare_data,
    labels=[
        "Direct\nphysics",
        "Direct\nanchors",
        "Shooting\nphysics",
        "Shooting\nanchors",
    ],
    showfliers=True,
    widths=0.65,
)

ax.set_yscale("log")
ax.set_ylabel(
    r"Complex error $|c-c^\star|$"
)
ax.set_title("(c) Matched anchors-only control")
ax.tick_params(axis="x", labelsize=8)

fig.suptitle(
    "Validation robustness of the supersonic neural atlas",
    fontsize=13,
)

save_both(
    fig,
    "Fig_S6_reviewer_robustness",
)

plt.close(fig)


# ============================================================
# 3. Inter-chart mismatch
# ============================================================

OVERLAP_ROOT = (
    ROOT / "assets/p3-supersonic-results"
)

points_path = require(
    OVERLAP_ROOT
    / "Tab_supersonic_N76_chart_overlap_points.csv"
)

pair_path = require(
    OVERLAP_ROOT
    / "Tab_supersonic_N76_chart_overlap_by_pair.csv"
)

global_path = require(
    OVERLAP_ROOT
    / "Tab_supersonic_N76_chart_overlap_global.csv"
)

points = pd.read_csv(points_path)
pairs = pd.read_csv(pair_path)
global_stats = pd.read_csv(global_path)

charts = [
    "C00", "C01", "C02",
    "C10", "C11", "C12",
    "C20", "C21", "C22",
    "C30", "C31", "C32",
]

chart_index = {
    c: i
    for i, c in enumerate(charts)
}

delta_pairs = (
    pairs[
        pairs["metric"].astype(str)
        .eq("delta_c")
    ]
    .copy()
)

if delta_pairs.empty:
    raise RuntimeError(
        "No metric='delta_c' rows found in "
        "Tab_supersonic_N76_chart_overlap_by_pair.csv"
    )

matrix = np.full(
    (len(charts), len(charts)),
    np.nan,
    dtype=float,
)

for _, row in delta_pairs.iterrows():
    c1 = str(row["chart_1"])
    c2 = str(row["chart_2"])

    i = chart_index[c1]
    j = chart_index[c2]

    value = float(row["p95"])

    matrix[i, j] = value
    matrix[j, i] = value


# Save the exact pairwise data used
delta_pairs.to_csv(
    OUT / "Tab_S7_interchart_delta_c_by_pair.csv",
    index=False,
)


# ------------------------------------------------------------
# Figure S7
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.8, 4.9),
    constrained_layout=True,
)

# ------------------------------------------------------------
# S7a — pairwise P95 matrix
# ------------------------------------------------------------

ax = axes[0]

masked = np.ma.masked_invalid(matrix)

im = ax.imshow(
    masked,
    origin="upper",
    aspect="equal",
)

ax.set_xticks(
    np.arange(len(charts)),
    charts,
    rotation=45,
    ha="right",
)

ax.set_yticks(
    np.arange(len(charts)),
    charts,
)

ax.set_title(
    r"(a) Pairwise $P_{95}(\delta_{AB})$"
)

for i in range(len(charts)):
    for j in range(len(charts)):
        if np.isfinite(matrix[i, j]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2e}",
                ha="center",
                va="center",
                fontsize=5.5,
            )

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04,
)

cbar.set_label(
    r"$P_{95}(|c_A-c_B|)$"
)


# ------------------------------------------------------------
# S7b — global ECDF
# ------------------------------------------------------------

ax = axes[1]

dc = (
    points["delta_c"]
    .astype(float)
    .to_numpy()
)

dc = dc[
    np.isfinite(dc)
    & (dc > 0)
]

x = np.sort(dc)

cdf = (
    np.arange(1, len(x) + 1)
    / len(x)
)

ax.plot(
    x,
    cdf,
    linewidth=1.8,
)

median = float(np.median(dc))
p95 = float(np.quantile(dc, 0.95))
maximum = float(np.max(dc))

ax.axvline(
    median,
    linestyle="--",
    linewidth=1.0,
    label=rf"median = {median:.2e}",
)

ax.axvline(
    p95,
    linestyle=":",
    linewidth=1.2,
    label=rf"$P_{{95}}$ = {p95:.2e}",
)

ax.set_xscale("log")
ax.set_xlabel(
    r"Inter-chart discrepancy $\delta_{AB}=|c_A-c_B|$"
)
ax.set_ylabel("Empirical cumulative fraction")
ax.set_ylim(0, 1.02)

ax.set_title(
    "(b) Distribution over overlap evaluations"
)

ax.legend(
    frameon=False,
    fontsize=8,
    loc="lower right",
)

ax.text(
    0.04,
    0.93,
    f"N = {len(dc):,}\nmax = {maximum:.2e}",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
)

fig.suptitle(
    "Consistency between independently trained atlas charts",
    fontsize=13,
)

save_both(
    fig,
    "Fig_S7_interchart_mismatch",
)

plt.close(fig)


# ============================================================
# 4. Provenance manifest
# ============================================================

manifest = OUT / "ASSET_MANIFEST.txt"

with manifest.open("w") as f:
    f.write(
        "SUPPLEMENTARY SUPERSONIC VISUAL ASSETS\n"
        "======================================\n\n"
    )

    f.write("S2 — Blumen diagnostics\n")
    f.write(
        "Fig_S2a_blumen_pointwise_delta_ci.pdf\n"
        "  source: "
        "assets/classic_supersonic/"
        "blumen_pointwise_classical_v1/"
        "blumen_pointwise_delta_ci_heatmap.pdf\n"
    )
    f.write(
        "Fig_S2b_blumen_fixedM_delta_alpha.pdf\n"
        "  source: "
        "assets/classic_supersonic/"
        "blumen_true_classical_isolines_v1/"
        "blumen_true_classical_delta_alpha_heatmap.pdf\n\n"
    )

    f.write("S4 — convergence diagnostics\n")
    for name in [
        "Fig_S4a_spectral_integration_convergence.pdf",
        "Fig_S4b_spectral_box_convergence.pdf",
        "Fig_S4c_matching_location_sensitivity.pdf",
        "Fig_S4d_modal_convergence.pdf",
    ]:
        f.write(name + "\n")

    f.write("\nS6 — reviewer robustness\n")
    f.write(
        "Fig_S6_reviewer_robustness.pdf/png\n"
    )
    f.write(
        "Tab_S6_reviewer_robustness_summary.csv\n"
    )
    f.write(
        "  recovery threshold: |c-c*| <= 1e-4\n"
        "  ambiguous point excluded from binary recovery count: "
        "(M, alpha) = (1.10, 0.09)\n\n"
    )

    f.write("S7 — inter-chart consistency\n")
    f.write(
        "Fig_S7_interchart_mismatch.pdf/png\n"
    )
    f.write(
        "Tab_S7_interchart_delta_c_by_pair.csv\n"
    )

print("WROTE:", manifest)


# ============================================================
# 5. Final verification
# ============================================================

expected = [
    "Fig_S2a_blumen_pointwise_delta_ci.pdf",
    "Fig_S2b_blumen_fixedM_delta_alpha.pdf",
    "Fig_S4a_spectral_integration_convergence.pdf",
    "Fig_S4b_spectral_box_convergence.pdf",
    "Fig_S4c_matching_location_sensitivity.pdf",
    "Fig_S4d_modal_convergence.pdf",
    "Fig_S6_reviewer_robustness.pdf",
    "Fig_S6_reviewer_robustness.png",
    "Fig_S7_interchart_mismatch.pdf",
    "Fig_S7_interchart_mismatch.png",
    "Tab_S6_reviewer_robustness_summary.csv",
    "Tab_S7_interchart_delta_c_by_pair.csv",
    "ASSET_MANIFEST.txt",
]

missing = [
    name
    for name in expected
    if not (OUT / name).is_file()
]

print()
print("=" * 68)
print("FINAL SUPPLEMENTARY ASSET CHECK")
print("=" * 68)

if missing:
    print("MISSING:")
    for name in missing:
        print(" ", name)
    raise SystemExit(1)

for name in expected:
    p = OUT / name
    print(
        f"{name:52s} "
        f"{p.stat().st_size / 1024:.1f} KiB"
    )

print()
print("ALL SUPPLEMENTARY VISUAL ASSETS READY.")
