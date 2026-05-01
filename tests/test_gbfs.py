import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gbfs import gbfs_standard, gbfs_structured


def _small_dataset():
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.2],
            [1.0, 0.8],
            [2.0, 1.0],
            [-1.5, 0.1],
            [1.5, 0.9],
        ]
    )
    y = np.array([-1, -1, 1, 1, -1, 1])
    bags = np.array([0, 1])
    return X, y, bags


def test_standard_gbfs_runs_and_tracks_features():
    X, y, bags = _small_dataset()

    trajectory, feats, bags_used = gbfs_standard(
        X, y, X, y, bags, mu=0.0, max_depth=1, epsilon=0.1, T=3
    )

    assert len(trajectory) == 3
    assert len(feats) > 0
    assert bags_used is None
    assert all(count >= 0 and 0.0 <= err <= 1.0 for count, err in trajectory)


def test_structured_gbfs_runs_and_tracks_bags():
    X, y, bags = _small_dataset()

    trajectory, feats, bags_used = gbfs_structured(
        X, y, X, y, bags, mu=0.0, max_depth=1, epsilon=0.1, T=3
    )

    assert len(trajectory) == 3
    assert len(feats) > 0
    assert len(bags_used) > 0
    assert all(count >= 0 and 0.0 <= err <= 1.0 for count, err in trajectory)


def test_standard_gbfs_breaks_zero_score_ties_as_positive_class():
    X = np.array([[-1.0], [1.0], [-2.0], [2.0]])
    y = np.array([-1, 1, -1, 1])
    bags = np.array([0])

    trajectory, _, _ = gbfs_standard(
        X, y, X, y, bags, mu=0.0, max_depth=0, epsilon=0.1, T=1
    )

    assert trajectory == [(0, 0.5)]
