from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PACKAGE_ROOT / "scripts/convergence"
sys.path.insert(0, str(SCRIPT_DIR))

import plots.scripts.classic_supersonic.plot_classical_convergence as plotting  # noqa: E402


FULL_ERRORS = (
    PACKAGE_ROOT
    / "reproducibility/results/classical_convergence_modalfix_v1_full/errors/convergence_errors.csv"
)


def _full_frame() -> pd.DataFrame:
    return plotting.load_plot_frame(FULL_ERRORS, figure="both")


def test_reference_runs_are_excluded_from_logarithmic_box_curves():
    fig = plotting.build_box_convergence_figure(_full_frame())
    try:
        sub_spectral_x = np.concatenate([line.get_xdata() for line in fig.axes[0].lines])
        sup_spectral_x = np.concatenate([line.get_xdata() for line in fig.axes[1].lines])
        sub_modal_x = np.concatenate([line.get_xdata() for line in fig.axes[2].lines])
        sup_modal_x = np.concatenate([line.get_xdata() for line in fig.axes[3].lines])
        assert 100.0 not in sub_spectral_x
        assert 100.0 not in sub_modal_x
        assert 2500.0 not in sup_spectral_x
        assert 2500.0 not in sup_modal_x
    finally:
        plt.close(fig)


def test_roundoff_is_censored_without_plotting_an_artificial_1e16_floor():
    fig, ax = plt.subplots()
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "error": [0.0, 1.0e-16, 1.0e-5]})
    try:
        plotting._plot_article_series(
            ax,
            frame,
            x_column="x",
            metric="error",
            label="test",
            color="black",
            zero_tolerance=1.0e-14,
        )
        plotted_y = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in ax.lines])
        assert not np.isclose(plotted_y, 1.0e-16, rtol=0.0, atol=1.0e-30).any()
        assert 1.0e-14 in plotted_y
        assert 1.0e-5 in plotted_y
    finally:
        plt.close(fig)


def test_entirely_zero_series_uses_upper_bound_markers_without_a_line():
    fig, ax = plt.subplots()
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "error": [0.0, 0.0, 0.0]})
    try:
        note = plotting._plot_article_series(
            ax,
            frame,
            x_column="x",
            metric="error",
            label="zero",
            color="black",
            zero_tolerance=1.0e-14,
        )
        assert "maximum variation < 1e-14" in note
        assert len(ax.lines) == 1
        assert ax.lines[0].get_linestyle() == "None"
        assert np.allclose(ax.lines[0].get_ydata(), 1.0e-14)
    finally:
        plt.close(fig)


def test_reference_must_share_case_id_and_sweep_type():
    frame = pd.DataFrame(
        [
            {
                "run_id": "a",
                "reference_run_id": "b",
                "case_id": "case_a",
                "sweep_type": "shooting_box",
            },
            {
                "run_id": "b",
                "reference_run_id": "b",
                "case_id": "case_b",
                "sweep_type": "shooting_box",
            },
        ]
    )
    with pytest.raises(ValueError, match="Reference mismatch"):
        plotting.validate_reference_integrity(frame)


def test_box_figure_has_at_most_four_series_and_one_core_threshold(monkeypatch):
    observed_thresholds: list[set[float]] = []
    original = plotting._plot_article_series

    def recording_plot(ax, frame, **kwargs):
        if "core_threshold" in frame and not frame.empty:
            observed_thresholds.append(set(frame["core_threshold"].astype(float)))
        return original(ax, frame, **kwargs)

    monkeypatch.setattr(plotting, "_plot_article_series", recording_plot)
    fig = plotting.build_box_convergence_figure(_full_frame(), core_threshold=1.0e-2)
    try:
        for ax in fig.axes:
            labels = [line.get_label() for line in ax.lines if not line.get_label().startswith("_")]
            assert len(labels) <= 4
        assert observed_thresholds
        assert all(values == {1.0e-2} for values in observed_thresholds)
    finally:
        plt.close(fig)


def test_pseudo_integration_sweep_with_multiple_varying_parameters_is_rejected():
    frame = pd.DataFrame(
        {
            "max_step": [2.0, 1.0, 0.5],
            "rtol": [1.0e-8, 1.0e-10, 1.0e-10],
            "atol": [1.0e-10, 1.0e-12, 1.0e-12],
        }
    )
    with pytest.raises(ValueError, match="Pseudo-sweep rejected"):
        plotting.validate_one_parameter_sweep(
            frame,
            parameter="max_step",
            fixed_columns=("rtol", "atol"),
            minimum_levels=3,
        )


def test_valid_integration_subset_changes_only_max_step():
    selected = plotting.prepare_integration_sweep(_full_frame())
    assert set(selected["case_id"]) == {"supersonic_M180", "supersonic_M190"}
    for _, group in selected.groupby("case_id"):
        assert set(group["max_step"].astype(float)) == {0.25, 0.5, 1.0, 2.0}
        assert group["rtol"].nunique() == 1
        assert group["atol"].nunique() == 1


def test_box_figure_is_generated_without_calling_integration(monkeypatch, tmp_path: Path):
    def forbidden(*args, **kwargs):
        raise AssertionError("integration builder must not be called")

    monkeypatch.setattr(plotting, "build_integration_convergence_figure", forbidden)
    result = plotting.main(
        [
            "--errors-csv",
            str(FULL_ERRORS),
            "--output-dir",
            str(tmp_path),
            "--figure",
            "box",
            "--formats",
            "png",
            "--dpi",
            "80",
        ]
    )
    assert result == 0
    assert (tmp_path / "fig_classical_box_convergence.png").exists()

