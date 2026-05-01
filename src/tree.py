import numpy as np


class Node:
    def __init__(
        self, depth, left=None, right=None, feature_idx=None, value=None, threshold=None
    ):
        self.depth = depth
        self.left = left
        self.right = right
        self.feature_idx = feature_idx
        self.value = value
        self.threshold = threshold


class TreeLearner:
    def __init__(self, max_depth, mu):
        self.depth = max_depth
        self.mu = mu
        self.used_tree_feats = set()
        self.used_global_feats = set()
        self.root = None

    def fit(self, X, grad):
        self.used_tree_feats = set()
        n_samples, self._n_feats = X.shape

        self._sorted_idx = np.argsort(X, axis=0)  # (n, p)
        self._X_sorted = np.sort(X, axis=0)  # (n, p)
        grad_tiled = np.tile(grad[:, None], (1, self._n_feats))
        self._grad_sorted = np.take_along_axis(grad_tiled, self._sorted_idx, axis=0)

        self._X = X
        self._grad = grad

        self.root = self._find_split(np.ones(n_samples, dtype=bool), depth=0)
        return self.root

    def _find_split(self, active_mask, depth):
        active_idx = np.where(active_mask)[0]
        n = len(active_idx)
        p = self._n_feats

        if n == 0:
            return Node(depth, value=0.0)
        if depth >= self.depth or n <= 1:
            return Node(depth, value=np.mean(self._grad[active_idx]))

        grad_active = self._grad[active_mask]
        total_sum = np.sum(grad_active)
        total_sq = np.sum(grad_active**2)

        sorted_active = active_mask[self._sorted_idx]  # (N, p)
        order = np.argsort(sorted_active, axis=0, kind="stable")  # (N, p)
        active_rows = order[-n:, :]  # (n, p)

        sx = np.take_along_axis(self._X_sorted, active_rows, axis=0)  # (n, p)
        sg = np.take_along_axis(self._grad_sorted, active_rows, axis=0)  # (n, p)
        cum_sum = np.cumsum(sg, axis=0)  # (n, p)
        cum_sq = np.cumsum(sg**2, axis=0)  # (n, p)
        cum_n = np.arange(1, n + 1, dtype=np.float64)[:, None] * np.ones(p)  # (n, p)

        valid = np.zeros((n, p), dtype=bool)
        valid[:-1, :] = sx[:-1, :] != sx[1:, :]

        n_L = cum_n
        n_R = n - cum_n
        sum_L = cum_sum
        sum_R = total_sum - cum_sum
        sq_L = cum_sq
        sq_R = total_sq - cum_sq

        losses = np.full((n, p), np.inf)
        v = valid
        mse_L = sq_L[v] / n_L[v] - (sum_L[v] / n_L[v]) ** 2
        mse_R = sq_R[v] / n_R[v] - (sum_R[v] / n_R[v]) ** 2
        # Fix: use sse instead of mse here.
        losses[v] = (n_L[v] * mse_L + n_R[v] * mse_R) # / n
        losses *= 0.5

        leaf_loss = 0.5 * (total_sq - total_sum**2 / n)

        phi = np.ones(p)
        if self.used_global_feats:
            phi[list(self.used_global_feats)] = 0
        F = np.ones(p)
        if self.used_tree_feats:
            F[list(self.used_tree_feats)] = 0
        losses += (self.mu * phi * F)[None, :]  # (n, p)

        flat = np.argmin(losses)
        best_row, best_feat = divmod(flat, p)
        best_loss = losses[best_row, best_feat]

        if np.isinf(best_loss) or best_loss >= leaf_loss:
            return Node(depth, value=np.mean(grad_active))

        best_threshold = sx[best_row, best_feat]
        self.used_tree_feats.add(best_feat)
        self.used_global_feats.add(best_feat)

        left_mask = active_mask & (self._X[:, best_feat] <= best_threshold)
        right_mask = active_mask & (self._X[:, best_feat] > best_threshold)

        left = self._find_split(left_mask, depth + 1)
        right = self._find_split(right_mask, depth + 1)

        return Node(depth, left, right, best_feat, None, best_threshold)

    def predict(self, X):
        def _predict_single(x, node):
            if node.value is not None:
                return node.value
            if x[node.feature_idx] <= node.threshold:
                return _predict_single(x, node.left)
            return _predict_single(x, node.right)

        return np.array([_predict_single(x, self.root) for x in X])


class StructuredTreeLearner(TreeLearner):
    def __init__(self, max_depth, mu, bags):
        super().__init__(max_depth, mu)
        self.bags = np.asarray(bags)
        self.used_global_bags = set()

    def _find_split(self, active_mask, depth):
        active_idx = np.where(active_mask)[0]
        n = len(active_idx)
        p = self._n_feats

        if n == 0:
            return Node(depth, value=0.0)
        if depth >= self.depth or n <= 1:
            return Node(depth, value=np.mean(self._grad[active_idx]))

        grad_active = self._grad[active_mask]
        total_sum = np.sum(grad_active)
        total_sq = np.sum(grad_active**2)

        sorted_active = active_mask[self._sorted_idx]
        order = np.argsort(sorted_active, axis=0, kind="stable")
        active_rows = order[-n:, :]

        sx = np.take_along_axis(self._X_sorted, active_rows, axis=0)
        sg = np.take_along_axis(self._grad_sorted, active_rows, axis=0)

        cum_sum = np.cumsum(sg, axis=0)
        cum_sq = np.cumsum(sg**2, axis=0)
        cum_n = np.arange(1, n + 1, dtype=np.float64)[:, None] * np.ones(p)

        valid = np.zeros((n, p), dtype=bool)
        valid[:-1, :] = sx[:-1, :] != sx[1:, :]

        n_L = cum_n
        n_R = n - cum_n
        sum_L = cum_sum
        sum_R = total_sum - cum_sum
        sq_L = cum_sq
        sq_R = total_sq - cum_sq

        losses = np.full((n, p), np.inf)
        v = valid
        mse_L = sq_L[v] / n_L[v] - (sum_L[v] / n_L[v]) ** 2
        mse_R = sq_R[v] / n_R[v] - (sum_R[v] / n_R[v]) ** 2
        losses[v] = (n_L[v] * mse_L + n_R[v] * mse_R) # / n
        losses *= 0.5

        leaf_loss = 0.5 * (total_sq - total_sum**2 / n)

        phi = np.ones(p)
        if self.used_global_bags:
            phi[np.isin(self.bags, list(self.used_global_bags))] = 0
        F = np.ones(p)
        if self.used_tree_feats:
            F[list(self.used_tree_feats)] = 0
        losses += (self.mu * phi * F)[None, :]

        flat = np.argmin(losses)
        best_row, best_feat = divmod(flat, p)
        best_loss = losses[best_row, best_feat]

        if np.isinf(best_loss) or best_loss >= leaf_loss:
            return Node(depth, value=np.mean(grad_active))

        best_threshold = sx[best_row, best_feat]
        self.used_tree_feats.add(best_feat)
        self.used_global_feats.add(best_feat)
        self.used_global_bags.add(int(self.bags[best_feat]))

        left_mask = active_mask & (self._X[:, best_feat] <= best_threshold)
        right_mask = active_mask & (self._X[:, best_feat] > best_threshold)

        left = self._find_split(left_mask, depth + 1)
        right = self._find_split(right_mask, depth + 1)

        return Node(depth, left, right, best_feat, None, best_threshold)
