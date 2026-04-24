import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from tree import TreeLearner, StructuredTreeLearner
    from sklearn.model_selection import train_test_split
    from matplotlib import pyplot as plt

    return StructuredTreeLearner, TreeLearner, np, plt, train_test_split


@app.cell
def _(np):
    data = np.load("colon_data.npz")
    return (data,)


@app.cell
def _(data, train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(
        data["X"], data["y"], test_size=0.2, random_state=43
    )
    return X_train, y_train


@app.cell
def _(StructuredTreeLearner, TreeLearner, data, np, train_test_split):
    def gbfs(X, y, T=2000, epsilon=0.1, mu=0.1):
        H = np.zeros_like(y, dtype=np.float64)
        trees = []
        Omega = set()
        for t in range(T):
            g = y * np.exp(-y * H) / (1 + np.exp(-y * H))
            learner_tree = TreeLearner(max_depth=4, mu=mu)
            learner_tree.used_global_feats = Omega
            learner_tree.fit(X, g)
            Omega.update(learner_tree.used_global_feats)
            H += epsilon * learner_tree.predict(X)
            trees.append(learner_tree)
        return trees

    def structured_gbfs(X, y, T=2000, epsilon=0.1, mu=0.1):
        H = np.zeros_like(y, dtype=np.float64)
        trees = []
        Omega = set()
        opened_bags = set()
        for t in range(T):
            g = y * np.exp(-y * H) / (1 + np.exp(-y * H))
            learner_tree = StructuredTreeLearner(
                max_depth=4, mu=mu, bags=data["bag_id"]
            )
            learner_tree.used_global_feats = Omega
            learner_tree.used_global_bags = opened_bags
            learner_tree.fit(X, g)
            Omega.update(learner_tree.used_global_feats)
            H += epsilon * learner_tree.predict(X)
            trees.append(learner_tree)
        return trees

    def evaluate_model(T=2000, epsilon=0.1, mu=0.1):
        np.random.seed(42)
        split_seeds = np.random.randint(0, 1000000, size=10)
        accuracies = []
        selected_features_nums = []
        for seed in split_seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                data["X"], data["y"], test_size=0.2, random_state=seed
            )
            struc_trees = structured_gbfs(X_train, y_train, T=T, epsilon=epsilon, mu=mu)
            y_pred_test = np.zeros_like(y_test, dtype=np.float64)
            accuracy = []
            for tree in struc_trees:
                y_pred_test += 0.1 * tree.predict(X_test)
                y_pred_test_labels = np.where(y_pred_test > 0, 1, -1)
                accu = np.mean(y_pred_test_labels == y_test)
                accuracy.append(accu)
            accuracies.append(accuracy)
        return np.mean(accuracies, axis=0)

    return evaluate_model, structured_gbfs


@app.cell
def _(evaluate_model):
    accuracys = evaluate_model(T=2000, epsilon=0.1, mu=1)
    return (accuracys,)


@app.cell
def _(accuracys, plt):
    plt.figure(figsize=(10, 5))
    plt.plot(accuracys, label="Structured Test Error")
    plt.xscale("log")
    plt.xlabel("Number of Trees")
    plt.ylabel("Error Rate")
    plt.title("Error Rate vs Number of Trees")
    plt.legend()
    plt.grid()
    plt.savefig("images/error-vs-trees.png")
    plt.show()
    return


@app.cell
def _(X_train, data, np, plt, structured_gbfs, y_train):
    trees = structured_gbfs(X_train, y_train, mu=0.001)
    _used_feats = trees[-1].used_global_feats
    _n_features = X_train.shape[1]
    _mask = np.zeros(_n_features, dtype=bool)
    for _f in _used_feats:
        if 0 <= _f < _n_features:
            _mask[_f] = True

    _bag_ids = np.ravel(data["bag_id"])

    if _bag_ids.shape[0] != _mask.shape[0]:
        raise ValueError("data['bag_id'] 的长度需要与特征数一致。")

    _unique_bags = np.unique(_bag_ids)
    _plot_bags = _unique_bags

    _fig_width = max(10, 0.8 * len(_plot_bags) + 2)
    _fig, _ax = plt.subplots(figsize=(_fig_width, 8))

    _bar_width = 0.72
    _display_height = 1.0

    for _x, _bag in enumerate(_plot_bags):
        _bag_mask = _bag_ids == _bag
        _bag_selected = _mask[_bag_mask]
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
    _ax.set_title(
        "Feature selection on structured feature data (Tree-based GBFS)", pad=12
    )

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
    plt.savefig("images/feature_selection_by_bag_gbfs.png", dpi=300, bbox_inches="tight")
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
