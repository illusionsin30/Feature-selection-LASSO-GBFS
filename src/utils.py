import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath="colon_data.npz"):
    data = np.load(filepath)
    X = data["X"]
    y = data["y"]
    bag_id = data["bag_id"]
    return X, y, bag_id


def logistic_neg_gradient(y, H):
    return y / (1.0 + np.exp(y * H))


def plot_error_vs(
    x,
    y,
    x_label="Complexity",
    y_label="Test Error",
    title="Error vs. Complexity",
    log_x=False,
    scatter=False,
    c_values=None,
    cmap="viridis",
    colorbar_label=None,
    line=True,
    line_color="#d62728",
    label=None,
    grid=True,
    savefig=True,
    filename="plot.png",
):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if scatter and c_values is not None:
        sc = ax.scatter(x, y, c=c_values, cmap=cmap, s=24, edgecolors="none", zorder=3)
        if colorbar_label is not None:
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label(colorbar_label, fontsize=12)
    
    if line:
        ax.plot(x, y, color=line_color, linewidth=1, zorder=2, label=label)
    
    if log_x:
        ax.set_xscale("log")
    
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=14, pad=12)
    
    if grid:
        ax.grid(True, linestyle="--", alpha=0.5)
    
    if label is not None:
        ax.legend()
    
    fig.tight_layout()
    if savefig:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {filename}")
    else:
        plt.show()
    plt.close(fig)


def plot_feature_selection_bag(feature_mask, bag_ids, title="Feature selection on structured feature data",
                               savefig=True, filename="feature_selection_by_bag.png"):
    feature_mask = np.asarray(feature_mask, dtype=bool)
    bag_ids = np.asarray(bag_ids, dtype=int)
    unique_bags = np.unique(bag_ids)

    fig_width = max(10, 0.8 * len(unique_bags) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    bar_width = 0.72
    display_height = 1.0

    for x, bag in enumerate(unique_bags):
        bag_mask = bag_ids == bag
        bag_selected = feature_mask[bag_mask]
        n = bag_selected.size
        if n == 0:
            continue

        h = display_height / n
        bottom = np.arange(n) * h
        colors = np.where(bag_selected, "#0dab0d", "#0b65a6")

        if bag_selected.any():
            margin_x, margin_y = 0.08, 0.02
            ax.add_patch(plt.Rectangle(
                (x - bar_width/2 - margin_x, -margin_y),
                bar_width + 2*margin_x, display_height + 2*margin_y,
                facecolor="white", edgecolor="red", linewidth=2.5, zorder=1))
        else:
            ax.add_patch(plt.Rectangle(
                (x - bar_width/2, 0), bar_width, display_height,
                fill=False, edgecolor="gray", linewidth=2))
        
        ax.bar(np.full(n, x), np.full(n, h), bottom=bottom, width=bar_width,
               color=colors, edgecolor="none", align="center")


    ax.set_xlim(-0.9, len(unique_bags) - 0.1)
    ax.set_ylim(-0.05, display_height + 0.05)
    ax.set_xticks(np.arange(len(unique_bags)))
    ax.set_xticklabels([f"Bag {b}" for b in unique_bags], rotation=45, ha="right")
    ax.set_xlabel("Bag")
    ax.set_title(title, pad=12)
    ax.yaxis.set_visible(False)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    legend_handles = [
        plt.Rectangle((0,0),1,1,color="#0dab0d"),
        plt.Rectangle((0,0),1,1,color="#0b65a6"),
        plt.Rectangle((0,0),1,1,facecolor="white",edgecolor="red",linewidth=2.5),
        plt.Rectangle((0,0),1,1,fill=False,edgecolor="gray",linewidth=2)]
    legend_labels = ["Selected feature", "Unselected feature",
                     "Selected bag (highlighted)", "Unselected bag"]
    ax.legend(legend_handles, legend_labels, loc="upper left",
              bbox_to_anchor=(1.02,1), borderaxespad=0)
    fig.tight_layout()
    if savefig:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Figure saved as {filename}")
    else:
        plt.show()
    plt.close(fig)