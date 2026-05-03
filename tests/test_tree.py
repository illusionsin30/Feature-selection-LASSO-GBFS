import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tree import StructuredTreeLearner, TreeLearner


def test_tree_learner_predicts_after_fit():
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    grad = np.array([-1.0, -0.5, 0.5, 1.0])
    learner = TreeLearner(max_depth=1, mu=0.0)

    learner.fit(X, grad)

    pred = learner.predict(X)

    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))


def test_constant_features_stop_as_finite_leaf():
    X = np.ones((4, 2))
    grad = np.array([-1.0, 0.0, 1.0, 2.0])
    learner = StructuredTreeLearner(max_depth=3, mu=0.0, bags=np.array([0, 1]))

    root = learner.fit(X, grad)
    pred = learner.predict(X)

    assert root.value == np.mean(grad)
    assert root.left is None
    assert root.right is None
    assert np.all(np.isfinite(pred))


def test_positive_penalty_prevents_no_gain_split():
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    grad = np.ones(4)
    learner = TreeLearner(max_depth=2, mu=0.1)

    root = learner.fit(X, grad)

    assert root.value == 1.0
    assert root.left is None
    assert root.right is None
    assert learner.used_global_feats == set()


def test_tree_learner_records_global_features_when_splitting():
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    grad = np.array([-1.0, -0.5, 0.5, 1.0])
    learner = TreeLearner(max_depth=1, mu=0.0)

    learner.fit(X, grad)

    assert learner.used_global_feats == {0}


def test_split_objective_weights_child_variance_by_sample_count():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    grad = np.array([-2.0, -2.0, -2.0, -1.0, -2.0])
    learner = TreeLearner(max_depth=1, mu=0.0)

    root = learner.fit(X, grad)

    assert root.feature_idx == 0
    assert root.threshold == 2.0
