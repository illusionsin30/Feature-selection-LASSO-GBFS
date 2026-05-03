import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import load_data, plot_error_vs, plot_feature_selection_bag


def evaluate_lasso_for_C(X, y, C, n_splits=10, random_state=42):
    rng = np.random.RandomState(random_state)
    split_seeds = rng.randint(0, 1_000_000, size=n_splits)
    errors, n_features_list = [], []
    for seed in split_seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=C, max_iter=10000, random_state=seed
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        err = 1.0 - accuracy_score(y_test, y_pred)
        n_feat = np.sum(np.abs(clf.coef_) > 1e-6)
        errors.append(err)
        n_features_list.append(n_feat)
    
    err_mean = np.mean(errors)
    err_std = np.std(errors)
    feats_mean = np.mean(n_features_list)
    feats_std = np.std(n_features_list)

    return err_mean, err_std, feats_mean, feats_std


def plot_figure_error_features(X, y, C_values, n_splits=10, random_state=42, savefig=True):
    test_errors, avg_features = [], []
    err_stds, feats_stds = [], []
    results = []

    for C in C_values:
        err_mean, err_std, feats_mean, feats_std = evaluate_lasso_for_C(X, y, C, n_splits, random_state)
        test_errors.append(err_mean)
        avg_features.append(feats_mean)
        err_stds.append(err_std)
        feats_stds.append(feats_std)
        results.append({
            "C": C,
            "features_mean": feats_mean,
            "features_std": feats_std,
            "test_error_mean": err_mean,
            "test_error_std": err_std,
        })
    
    avg_features = np.array(avg_features)
    test_errors = np.array(test_errors)
    order = np.argsort(avg_features)
    log_C = np.log10(C_values)

    plot_error_vs(
        x=avg_features[order],
        y=test_errors[order],
        x_label="Number of selected features",
        title="LASSO: Test error vs. Number of selected features",
        scatter=True,
        c_values=log_C[order],
        colorbar_label="log C",
        line=True,
        savefig=savefig,
        filename="test_error_vs_selected_features.svg"
    )

    sort_idx = np.argsort(log_C)
    log_C_sorted = log_C[sort_idx]
    test_errors_sorted = test_errors[sort_idx]
    err_stds_sorted = np.array(err_stds)[sort_idx]
    features_sorted = avg_features[sort_idx]
    feats_stds_sorted = np.array(feats_stds)[sort_idx]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.errorbar(log_C_sorted, test_errors_sorted, yerr=err_stds_sorted,
                 fmt='o', color="#DD7D34", ecolor="#F8BDA6", capsize=1, markersize=1,
                 alpha=0.8, label='Test error')
    ax1.plot(log_C_sorted, test_errors_sorted, color="#DD7D34")
    ax1.set_xlabel(r'$\log_{10}(C)$')
    ax1.set_ylabel('Test error (0-1 loss)', color="#DD7D34")
    ax1.tick_params(axis='y', labelcolor="#DD7D34")
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    ax2 = ax1.twinx()
    ax2.errorbar(log_C_sorted, features_sorted, yerr=feats_stds_sorted,
                 fmt='s', color="#3FB3E5", ecolor="#99D6F0", capsize=1, markersize=1,
                 alpha=0.8, label='Number of selected features')
    ax2.plot(log_C_sorted, features_sorted, color="#3FB3E5")
    ax2.set_ylabel('Number of selected features', color="#3FB3E5")
    ax2.tick_params(axis='y', labelcolor="#3FB3E5")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title('LASSO: Test error and feature count vs. regularization strength C')
    
    if savefig:
        plt.savefig("lasso_test_error_and_features_vs_C.svg", format='svg', bbox_inches='tight')
    plt.show()

    # pd.DataFrame(results).to_csv("result.csv", index=False)
    

def main():
    np.random.seed(42)
    X, y, bag_id = load_data("colon_data.npz")

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf = LogisticRegression(
        penalty="l1", solver="liblinear", C=1.0, max_iter=10000, random_state=42
    )
    clf.fit(X_train, y_train)
    mask = np.ravel(clf.coef_ != 0).astype(bool)
    plot_feature_selection_bag(
        mask, bag_id,
        title="Feature selection on structured feature data (LASSO)",
        filename="feature_selection_by_bag_lasso.svg"
    )

    C_values = np.logspace(-4, 4, base=10, num=300)
    plot_figure_error_features(X, y, C_values, n_splits=10, random_state=42, savefig=True)


if __name__ == "__main__":
    main()
