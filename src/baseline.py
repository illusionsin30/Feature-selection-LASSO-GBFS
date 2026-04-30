import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import *


def evaluate_lasso_for_C(X, y, C, n_splits=10, random_state=42):
    rng = np.random.RandomState(random_state)
    split_seeds = rng.randint(0, 1_000_000, size=n_splits)
    errors, n_features_list = [], []
    for seed in split_seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=C, max_iter=1000, random_state=seed
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        err = 1.0 - accuracy_score(y_test, y_pred)
        n_feat = np.sum(np.abs(clf.coef_) > 1e-6)
        errors.append(err)
        n_features_list.append(n_feat)
    return np.mean(errors), np.mean(n_features_list)


def plot_figure_error_features(X, y, C_values, n_splits=10, random_state=42, savefig=True):
    test_errors, avg_features = [], []
    for C in C_values:
        err, feat = evaluate_lasso_for_C(X, y, C, n_splits, random_state)
        test_errors.append(err)
        avg_features.append(feat)
    
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


def main():
    np.random.seed(42)
    X, y, bag_id = load_data("colon_data.npz")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = LogisticRegression(
        penalty="l1", solver="liblinear", C=1.0, max_iter=1000, random_state=42
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