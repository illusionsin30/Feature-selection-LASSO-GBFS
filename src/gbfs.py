import time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed

from utils import *
from tree import TreeLearner, StructuredTreeLearner


def gbfs_structured(
    X_train,
    y_train,
    X_test,
    y_test,
    bags,
    mu,
    max_depth,
    epsilon,
    T
):
    n_train = X_train.shape[0]
    H = np.zeros(n_train)
    current_global_feats = set()
    current_global_bags = set()
    trees = []
    trajectory = []

    for t in range(1, T + 1):
        g = logistic_neg_gradient(y_train, H)

        learner = StructuredTreeLearner(max_depth=max_depth, mu=mu, bags=bags)
        learner.used_global_feats = set(current_global_feats)
        learner.used_global_bags = set(current_global_bags)

        learner.fit(X_train, g)
        h_pred = learner.predict(X_train)
        H += epsilon * h_pred

        current_global_feats = learner.used_global_feats
        current_global_bags = learner.used_global_bags

        trees.append(learner)

        scores_test = np.zeros(X_test.shape[0])
        for tree in trees:
            scores_test += epsilon * tree.predict(X_test)
        pred_test = np.sign(scores_test)
        test_err = np.mean(pred_test != y_test)

        trajectory.append((len(current_global_feats), test_err))

    return trajectory, current_global_feats, current_global_bags


def gbfs_standard(
    X_train,
    y_train,
    X_test,
    y_test,
    bags,
    mu,
    max_depth,
    epsilon,
    T
):
    n_train = X_train.shape[0]
    H = np.zeros(n_train)
    current_global_feats = set()
    trees = []
    trajectory = []

    for t in range(1, T + 1):
        g = logistic_neg_gradient(y_train, H)

        learner = TreeLearner(max_depth=max_depth, mu=mu)
        learner.used_global_feats = set(current_global_feats)
        learner.fit(X_train, g)
        h_pred = learner.predict(X_train)
        H += epsilon * h_pred

        current_global_feats = learner.used_global_feats
        trees.append(learner)

        scores_test = np.zeros(X_test.shape[0])
        for tree in trees:
            scores_test += epsilon * tree.predict(X_test)
        pred_test = np.sign(scores_test)
        test_err = np.mean(pred_test != y_test)

        trajectory.append((len(current_global_feats), test_err))

    return trajectory, current_global_feats, None


def run_one_fold(
    mu,
    depth,
    X,
    y,
    bags,
    train_idx,
    test_idx,
    epsilon,
    T,
    mode='structured'
):
    fns = {
        'structured': gbfs_structured,
        'standard': gbfs_standard,
    }
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    traj, feats, _ = fns[mode](
        X_train, y_train, X_test, y_test,
        bags, mu, depth, epsilon, T
    )
    return mu, depth, traj, len(feats)


def main():
    np.random.seed(42)
    X, y, bags = load_data("colon_data.npz")

    mu = 2**-3
    depth = 4
    epsilon = 0.1
    T = 250  
    # even less can be set to speed up
    # since colon dataset is to small (62 samples)
    # larger T will overfit
    n_splits = 10
    random_state = 42

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2,
                                 random_state=random_state)
    split_indices = list(sss.split(X, y))

    # ---------- Standard GBFS ----------
    tasks_std = [(mu, depth, fold_id, train_idx, test_idx)
                 for fold_id, (train_idx, test_idx) in enumerate(split_indices)]
    print("Evaluating Standard GBFS …")
    start = time.perf_counter()
    #! TODO: Change the setting according to your device
    # 10 tasks so n_jobs is set to 10
    # make sure available cpu cores for other tasks
    results_std = Parallel(n_jobs=10, verbose=10)(
        delayed(run_one_fold)(mu, depth, X, y, bags, train_idx, test_idx, epsilon, T, mode="standard")
        for _, _, fold_id, train_idx, test_idx in tasks_std
    )
    std_errors = [t[-1][1] for _, _, t, _ in results_std]
    std_feats = [n for _, _, _, n in results_std]
    print(f"Standard GBFS: error={np.mean(std_errors):.4f}±{np.std(std_errors):.4f}, "
          f"features={np.mean(std_feats):.1f}±{np.std(std_feats):.1f}")

    # ---------- Structured GBFS ----------
    tasks_struct = [(mu, depth, fold_id, train_idx, test_idx)
                    for fold_id, (train_idx, test_idx) in enumerate(split_indices)]
    print("Evaluating Structured GBFS …")
    start = time.perf_counter()
    results_struct = Parallel(n_jobs=10, verbose=10)(
        delayed(run_one_fold)(mu, depth, X, y, bags, train_idx, test_idx, epsilon, T, mode="structured")
        for _, _, fold_id, train_idx, test_idx in tasks_struct
    )
    struct_errors = [t[-1][1] for _, _, t, _ in results_struct]
    struct_feats = [n for _, _, _, n in results_struct]
    print(f"Structured GBFS: error={np.mean(struct_errors):.4f}±{np.std(struct_errors):.4f}, "
          f"features={np.mean(struct_feats):.1f}±{np.std(struct_feats):.1f}")

    elapsed = time.perf_counter() - start
    print(f"Cross‑validation finished in {elapsed:.2f} s")

    # ---------- Feature bag plot for Structured GBFS ----------
    print("Plotting feature selection bag visualization …")
    start = time.perf_counter()
    _, Omega = gbfs_structured(X, y, X, y, bags,
                               mu=0.001, max_depth=depth,
                               epsilon=epsilon, T=T)[:2]
    elapsed = time.perf_counter() - start
    print(f"Single‑fit for bag plot: {elapsed:.2f} s")

    mask = np.zeros(X.shape[1], dtype=bool)
    for f in Omega:
        mask[f] = True
    plot_feature_selection_bag(
        mask, bags,
        title="Feature selection on structured feature data (Structured GBFS)",
        filename="feature_selection_by_bag_gbfs.svg"
    )


if __name__ == "__main__":
    main()