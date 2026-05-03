import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hyperparam import (
    aggregate_fold_binned_curve,
    plot_gbfs_curves,
    summarize_combo_results,
    validate_args,
)


def test_aggregate_fold_binned_curve_includes_max_feature_count_on_bin_edge():
    trajectories = [[(0, 0.4), (10, 0.3), (20, 0.2)]]

    centers, means = aggregate_fold_binned_curve(trajectories, bin_width=10)

    assert np.allclose(centers, [5.0, 15.0, 25.0])
    assert np.allclose(means, [0.4, 0.3, 0.2])


def test_aggregate_fold_binned_curve_averages_folds_before_averaging_bins():
    trajectories = [
        [(0, 1.0), (1, 1.0), (2, 1.0)],
        [(0, 0.0)],
    ]

    centers, means = aggregate_fold_binned_curve(trajectories, bin_width=10)

    assert np.allclose(centers, [5.0])
    assert np.allclose(means, [0.5])


def test_summarize_combo_results_uses_sample_standard_deviation_when_possible():
    combo_results = {
        (0.125, 3): {
            "trajectories": [[(1, 0.1)], [(2, 0.3)]],
            "final_feats": [1, 3],
        }
    }

    summary = summarize_combo_results(combo_results)[(0.125, 3)]

    assert summary["mean_err"] == pytest.approx(0.2)
    assert summary["std_err"] == pytest.approx(np.sqrt(0.02))
    assert summary["mean_feat"] == pytest.approx(2.0)
    assert summary["std_feat"] == pytest.approx(np.sqrt(2.0))


def test_plot_gbfs_curves_applies_requested_line_alpha(monkeypatch):
    combo_results = {
        (0.125, 3): {
            "trajectories": [[(0, 0.4), (10, 0.3)]],
            "final_feats": [10],
        }
    }
    monkeypatch.setattr(plt, "close", lambda fig: None)

    plot_gbfs_curves(
        combo_results,
        mus=[0.125],
        depths=[3],
        savefig=False,
        line_alpha=0.42,
    )

    lines = plt.gcf().axes[0].lines
    assert len(lines) == 1
    assert lines[0].get_alpha() == pytest.approx(0.42)
    plt.close("all")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"T": 0, "n_splits": 1, "epsilon": 0.1}, "--T"),
        ({"T": 1, "n_splits": 0, "epsilon": 0.1}, "--n-splits"),
        ({"T": 1, "n_splits": 1, "epsilon": 0.0}, "--epsilon"),
        ({"T": 1, "n_splits": 1, "epsilon": 0.1, "n_jobs": 0}, "--n-jobs"),
    ],
)
def test_validate_args_rejects_invalid_numeric_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_args(SimpleNamespace(**kwargs))
