import argparse
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from itertools import product

from utils import load_data
from gbfs import default_n_jobs, run_one_fold


def validate_args(args):
    if args.n_splits <= 0:
        raise ValueError("--n-splits must be greater than 0")
    if args.T <= 0:
        raise ValueError("--T must be greater than 0")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be greater than 0")
    if getattr(args, "n_jobs", None) == 0:
        raise ValueError("--n-jobs must not be 0")


def sample_std(values):
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def summarize_combo_results(combo_results):
    summaries = {}
    for key, val in combo_results.items():
        trajectories = val["trajectories"]
        if any(len(t) == 0 for t in trajectories):
            raise ValueError("Each trajectory must contain at least one iteration")

        final_errors = np.array([t[-1][1] for t in trajectories], dtype=float)
        final_feats = np.array(val["final_feats"], dtype=float)
        summaries[key] = {
            "mean_err": float(np.mean(final_errors)),
            "std_err": sample_std(final_errors),
            "mean_feat": float(np.mean(final_feats)),
            "std_feat": sample_std(final_feats),
        }
    return summaries


def aggregate_fold_binned_curve(trajectories, bin_width=10):
    if bin_width <= 0:
        raise ValueError("bin_width must be greater than 0")

    trajectories = [traj for traj in trajectories if len(traj) > 0]
    if len(trajectories) == 0:
        return np.array([]), np.array([])

    max_feat = max(max(p[0] for p in traj) for traj in trajectories)
    upper = (max_feat // bin_width + 2) * bin_width
    bins = np.arange(0, upper + bin_width, bin_width)
    fold_means_by_bin = [[] for _ in range(len(bins) - 1)]

    for traj in trajectories:
        feats = np.array([p[0] for p in traj])
        errors = np.array([p[1] for p in traj], dtype=float)
        for i in range(len(bins) - 1):
            mask = (feats >= bins[i]) & (feats < bins[i + 1])
            if np.any(mask):
                fold_means_by_bin[i].append(float(np.mean(errors[mask])))

    bin_centers = []
    bin_means = []
    for i, fold_means in enumerate(fold_means_by_bin):
        if fold_means:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_means.append(float(np.mean(fold_means)))

    return np.array(bin_centers), np.array(bin_means)


def plot_gbfs_curves(
    combo_results,
    mus,
    depths,
    savefig=True,
    filename="task3_results.png",
    show=False,
    line_alpha=0.65,
):
    fig, axes = plt.subplots(1, len(depths), figsize=(18, 5))
    if len(depths) == 1:
        axes = [axes]

    for ax, depth in zip(axes, depths):
        for mu in mus:
            key = (mu, depth)
            if key not in combo_results:
                continue
            traj_list = combo_results[key]["trajectories"]
            bin_centers, bin_means = aggregate_fold_binned_curve(traj_list)
            if len(bin_centers) == 0:
                continue

            ax.plot(
                bin_centers,
                bin_means,
                marker="o",
                markersize=3,
                alpha=line_alpha,
                label=f"μ={mu:.3f}",
            )

        ax.set_title(f"Tree depth = {depth}")
        ax.set_xlabel("Number of selected features")
        ax.set_ylabel("Test error rate")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if savefig:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Figure saved as {filename}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run structured GBFS hyperparameter search."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel workers. Defaults to min(cpu_count, number of tasks).",
    )
    parser.add_argument(
        "--n-splits", type=int, default=10, help="Number of stratified shuffle splits."
    )
    parser.add_argument(
        "--T", type=int, default=2000, help="Number of boosting iterations per run."
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Boosting step size."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    np.random.seed(42)
    X, y, bags = load_data("colon_data.npz")

    mus = [2**-3, 2**-1, 2**1, 2**3, 2**5]
    depths = [3, 4, 5]
    epsilon = args.epsilon
    T = args.T

    sss = StratifiedShuffleSplit(n_splits=args.n_splits, test_size=0.2, random_state=42)
    split_indices = list(sss.split(X, y))

    tasks = []
    for mu, depth in product(mus, depths):
        for fold_id, (train_idx, test_idx) in enumerate(split_indices):
            tasks.append((mu, depth, fold_id, train_idx, test_idx))

    print(f"Total parallel tasks: {len(tasks)}")
    n_jobs = args.n_jobs if args.n_jobs is not None else default_n_jobs(len(tasks))
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one_fold)(
            mu, depth, X, y, bags, train_idx, test_idx, epsilon, T, mode="structured"
        )
        for mu, depth, fold_id, train_idx, test_idx in tasks
    )

    combo_results = {}
    for mu, depth, traj, feats in results:
        key = (mu, depth)
        if key not in combo_results:
            combo_results[key] = {"trajectories": [], "final_feats": []}
        combo_results[key]["trajectories"].append(traj)
        combo_results[key]["final_feats"].append(feats)

    summaries = summarize_combo_results(combo_results)
    for key, summary in summaries.items():
        print(
            f"mu={key[0]:.3f}, depth={key[1]}: "
            f"error={summary['mean_err']:.4f}±{summary['std_err']:.4f}, "
            f"features={summary['mean_feat']:.1f}±{summary['std_feat']:.1f}"
        )

    plot_gbfs_curves(
        combo_results, mus, depths, savefig=True, filename="task3_results.svg"
    )

    print("\n===== Final Results =====")
    print(
        f"{'mu':<8} {'depth':<6} {'mean_err':<10} {'std_err':<10} "
        f"{'mean_feat':<10} {'std_feat':<10}"
    )
    for mu in mus:
        for depth in depths:
            key = (mu, depth)
            if key not in summaries:
                continue
            summary = summaries[key]
            print(
                f"{mu:<8.3f} {depth:<6} {summary['mean_err']:<10.4f} "
                f"{summary['std_err']:<10.4f} {summary['mean_feat']:<10.1f} "
                f"{summary['std_feat']:<10.1f}"
            )

    if summaries:
        best_key = min(
            summaries,
            key=lambda k: (summaries[k]["mean_err"], summaries[k]["mean_feat"]),
        )
        best = summaries[best_key]
        print(
            "\nBest CV configuration for comparison only "
            "(use nested CV or a held-out test set for an unbiased final estimate):"
        )
        print(
            f"mu={best_key[0]:.3f}, depth={best_key[1]}, "
            f"mean_err={best['mean_err']:.4f}, mean_feat={best['mean_feat']:.1f}"
        )


if __name__ == "__main__":
    main()
