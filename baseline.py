import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    return LogisticRegression, accuracy_score, np, plt, train_test_split


@app.cell
def _(np):
    data = np.load('colon_data.npz')
    return (data,)


@app.cell
def _(data, train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y'],
        test_size=0.2,
        random_state=42
    )
    return X_train, y_train


@app.cell
def _(LogisticRegression, X_train, y_train):
    clf = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return (clf,)


@app.cell
def _(clf, data, np, plt):
    _coef_mask = np.ravel(clf.coef_ != 0).astype(bool)
    _bag_ids = np.ravel(data["bag_id"])

    if _bag_ids.shape[0] != _coef_mask.shape[0]:
        raise ValueError("data['bag_id'] 的长度需要与 clf.coef_ 对应的特征数一致。")

    _unique_bags = np.unique(_bag_ids)
    _plot_bags = _unique_bags

    _fig_width = max(10, 0.8 * len(_plot_bags) + 2)
    _fig, _ax = plt.subplots(figsize=(_fig_width, 8))

    _bar_width = 0.72
    _display_height = 1.0

    for _x, _bag in enumerate(_plot_bags):
        _bag_mask = _bag_ids == _bag
        _bag_selected = _coef_mask[_bag_mask]
        _n_features_in_bag = _bag_selected.shape[0]

        if _n_features_in_bag == 0:
            continue

        _segment_height = _display_height / _n_features_in_bag
        _bottom = np.arange(_n_features_in_bag) * _segment_height
        _colors = np.where(_bag_selected, "#2ca02c", "#1f77b4")

        _ax.bar(
            np.full(_n_features_in_bag, _x),
            np.full(_n_features_in_bag, _segment_height),
            bottom=_bottom,
            width=_bar_width,
            color=_colors,
            edgecolor="none",
            align="center",
        )

        _border_color = "red" if _bag_selected.any() else "gray"
        _ax.add_patch(
            plt.Rectangle(
                (_x - _bar_width / 2, 0),
                _bar_width,
                _display_height,
                fill=False,
                edgecolor=_border_color,
                linewidth=3,
            )
        )

    _ax.set_xlim(-0.7, len(_plot_bags) - 0.3)
    _ax.set_ylim(0, _display_height)
    _ax.set_xticks(np.arange(len(_plot_bags)))
    _ax.set_xticklabels([f"Bag {_bag}" for _bag in _plot_bags], rotation=45, ha="right")
    _ax.set_xlabel("Bag")
    _ax.set_title("Feature selection on structured feature data", pad=12)

    _ax.yaxis.set_visible(False)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.spines["left"].set_visible(False)
    _ax.spines["bottom"].set_visible(False)

    _legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c"),
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="red", linewidth=3),
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="gray", linewidth=3),
    ]
    _legend_labels = [
        "Selected feature",
        "Unselected feature",
        "Bag with selected feature",
        "Bag without selected feature",
    ]
    _ax.legend(
        _legend_handles,
        _legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    _fig.tight_layout()
    plt.savefig("images/feature_selection_by_bag.png", dpi=300, bbox_inches="tight")
    plt.gca()
    return


@app.cell
def _(np):
    regularization_strengths = np.logspace(-4, 4, base=10, num=300)
    return (regularization_strengths,)


@app.cell
def _(LogisticRegression, accuracy_score, data, np, train_test_split):
    def evaluate_model(C):
        np.random.seed(42)
        split_seeds = np.random.randint(0, 1000000, size=10)
        accuracies = []
        selected_features_nums = []
        for seed in split_seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                data['X'], data['y'],
                test_size=0.2,
                random_state=seed
            )
            clf = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=C,
                max_iter=1000,
                random_state=seed
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
            selected_features_nums.append(np.sum(clf.coef_ != 0))
        return 1 - np.mean(accuracies), np.mean(selected_features_nums)

    return (evaluate_model,)


@app.cell
def _(evaluate_model, regularization_strengths):
    test_errors = []
    avg_selected_features = []
    for C in regularization_strengths:
        _test_error, _avg_selected_features = evaluate_model(C)
        test_errors.append(_test_error)
        avg_selected_features.append(_avg_selected_features)
    return avg_selected_features, test_errors


@app.cell
def _(avg_selected_features, np, plt, regularization_strengths, test_errors):
    _fig2, _ax2 = plt.subplots(figsize=(10, 6))
    _log_C = np.log10(regularization_strengths)
    _sc = _ax2.scatter(
        avg_selected_features, test_errors,
        c=_log_C, cmap='viridis', s=24, edgecolors='none', zorder=3
    )
    _cb = _fig2.colorbar(_sc, ax=_ax2)
    _cb.set_label("log₁₀(C)", fontsize=12)
    _ax2.plot(avg_selected_features, test_errors, color='#d62728', linewidth=1, alpha=0.4, zorder=2)
    _ax2.set_xlabel("Number of selected features", fontsize=13)
    _ax2.set_ylabel("Test error", fontsize=13)
    _ax2.set_title("Test error vs. Number of selected features", fontsize=14, pad=12)
    _ax2.grid(True, linestyle='--', alpha=0.5)
    _fig2.tight_layout()
    plt.savefig("images/test_error_vs_selected_features.png", dpi=300)
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
