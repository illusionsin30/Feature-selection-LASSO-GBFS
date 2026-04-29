import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from itertools import product

from tree import *

def logistic_neg_gradient(y, H):
    return y / (1.0 + np.exp(y * H))

def gbfs_structured(X_train, y_train, X_test, y_test, bags,
                    mu, max_depth, epsilon, T):
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

        n_feat = len(current_global_feats)
        trajectory.append((n_feat, test_err))

    return trajectory, current_global_feats, current_global_bags

def run_one_fold(mu, depth, X, y, bags, train_idx, test_idx, epsilon, T):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    traj, feats, _ = gbfs_structured(
        X_train, y_train, X_test, y_test,
        bags, mu, depth, epsilon, T
    )

    return mu, depth, traj, len(feats)

def main():
    data = np.load('../colon_data.npz')
    X = data['X']
    y = data['y']
    bags = data['bag_id']

    mus = [2**-3, 2**-1, 2**1, 2**3, 2**5]
    # mu_values = [mu / 10 for mu in mu_values]
    depths = [3, 4, 5]
    epsilon = 0.1
    T = 2000

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    split_indices = list(sss.split(X, y))

    tasks = []
    for mu, depth in product(mus, depths):
        for fold_id, (train_idx, test_idx) in enumerate(split_indices):
            tasks.append((mu, depth, fold_id, train_idx, test_idx))

    print(f"num of parallel tasks: {len(tasks)}.")
    #! TODO: adapt n_jobs according to the cpu cores
    # since cpu of my server has totally 110+ or 280+ cores,
    # I set 80 here 
    results = Parallel(n_jobs=80, verbose=10)(
        delayed(run_one_fold)(mu, depth, X, y, bags, train_idx, test_idx, epsilon, T)
        for mu, depth, fold_id, train_idx, test_idx in tasks
    )

    combo_results = {}
    for (mu, depth, traj, feats) in results:
        key = (mu, depth)
        if key not in combo_results:
            combo_results[key] = {'trajectories': [], 'final_feats': []}
        combo_results[key]['trajectories'].append(traj)
        combo_results[key]['final_feats'].append(feats)

    for key, val in combo_results.items():
        final_errors = [t[-1][1] for t in val['trajectories']]
        final_feats = val['final_feats']
        print(f"mu={key[0]:.3f}, depth={key[1]}: "
              f"error={np.mean(final_errors):.4f}±{np.std(final_errors):.4f}, "
              f"features={np.mean(final_feats):.1f}±{np.std(final_feats):.1f}")

    fig, axes = plt.subplots(1, len(depths), figsize=(18, 5))
    for ax, depth in zip(axes, depths):
        for mu in mus:
            key = (mu, depth)
            traj_list = combo_results[key]['trajectories']

            all_feats = []
            all_errors = []
            for traj in traj_list:
                f_arr = [p[0] for p in traj]
                e_arr = [p[1] for p in traj]
                all_feats.extend(f_arr)
                all_errors.extend(e_arr)

            if len(all_feats) == 0:
                continue
            sort_idx = np.argsort(all_feats)
            all_feats = np.array(all_feats)[sort_idx]
            all_errors = np.array(all_errors)[sort_idx]

            max_feat = max(all_feats)
            bins = np.arange(0, max_feat + 10, 10)
            bin_centers = []
            bin_means = []
            for i in range(len(bins) - 1):
                mask = (all_feats >= bins[i]) & (all_feats < bins[i+1])
                if np.any(mask):
                    bin_centers.append((bins[i] + bins[i+1]) / 2)
                    bin_means.append(np.mean(all_errors[mask]))

            ax.plot(bin_centers, bin_means, marker='o', markersize=3, label=f'μ={mu:.3f}')

        ax.set_title(f'Tree depth = {depth}')
        ax.set_xlabel('Number of selected features')
        ax.set_ylabel('Test error rate')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('task3_results.png', dpi=150)
    plt.show()

    print("\n===== Final Results =====")
    print(f"{'mu':<8} {'depth':<6} {'mean_err':<10} {'std_err':<10} {'mean_feat':<10} {'std_feat':<10}")
    for mu in mus:
        for depth in depths:
            key = (mu, depth)
            errs = [t[-1][1] for t in combo_results[key]['trajectories']]
            feats = combo_results[key]['final_feats']
            print(f"{mu:<8.3f} {depth:<6} {np.mean(errs):<10.4f} {np.std(errs):<10.4f} {np.mean(feats):<10.1f} {np.std(feats):<10.1f}")

if __name__ == "__main__":
    main()