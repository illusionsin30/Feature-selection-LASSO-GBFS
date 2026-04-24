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
        """预排序 + 向量化建树。"""
        self.used_tree_feats = set()
        n_samples, self._n_feats = X.shape

        # ---- 预排序：每个特征只排一次 ----
        self._sorted_idx = np.argsort(X, axis=0)  # (n, p)
        self._X_sorted = np.sort(X, axis=0)  # (n, p)
        grad_tiled = np.tile(grad[:, None], (1, self._n_feats))
        self._grad_sorted = np.take_along_axis(grad_tiled, self._sorted_idx, axis=0)

        # 保留原始数据，用于 mask 分裂
        self._X = X
        self._grad = grad

        # 用 boolean mask 标记活跃样本，递归建树
        self.root = self._find_split(np.ones(n_samples, dtype=bool), depth=0)
        return self.root

    def _find_split(self, active_mask, depth):
        """递归寻找最优分裂点。

        全向量化：用 (n, p) 矩阵一次计算所有特征×阈值组合的损失，
        消除特征维度上的 Python 循环。
        """
        active_idx = np.where(active_mask)[0]
        n = len(active_idx)
        p = self._n_feats

        # ---- 停止条件 ----
        if depth >= self.depth or n <= 1:
            return Node(depth, value=np.mean(self._grad[active_idx]))

        # 当前节点所有活跃样本的梯度统计
        grad_active = self._grad[active_mask]
        total_sum = np.sum(grad_active)
        total_sq = np.sum(grad_active**2)

        # ---- Step 1: 用 argsort 技巧构造 (n, p) 稠密矩阵 ----
        # sorted_active[r, f] = active_mask[sorted_idx[r, f]]
        # True 表示该位置属于活跃样本
        sorted_active = active_mask[self._sorted_idx]  # (N, p)

        # 稳定排序：False(0) 在前，True(1) 在后，True 间保持原序
        order = np.argsort(sorted_active, axis=0, kind="stable")  # (N, p)

        # 取最后 n 行 = 每列 True 值在 sorted_* 中的行号（按特征值升序）
        active_rows = order[-n:, :]  # (n, p)

        # 批量提取：每列是该特征下排好序的活跃样本值
        sx = np.take_along_axis(self._X_sorted, active_rows, axis=0)  # (n, p)
        sg = np.take_along_axis(self._grad_sorted, active_rows, axis=0)  # (n, p)

        # ---- Step 2: 2D cumsum 计算所有候选阈值 ----
        cum_sum = np.cumsum(sg, axis=0)  # (n, p)
        cum_sq = np.cumsum(sg**2, axis=0)  # (n, p)
        cum_n = np.arange(1, n + 1, dtype=np.float64)[:, None] * np.ones(p)  # (n, p)

        # ---- Step 3: valid mask（值变化处，排除最后一行） ----
        valid = np.ones((n, p), dtype=bool)
        valid[1:, :] = sx[1:, :] != sx[:-1, :]
        valid[-1, :] = False

        # ---- Step 4: 批量计算 MSE（先屏蔽无效位避免除零） ----
        n_L = cum_n
        n_R = n - cum_n
        sum_L = cum_sum
        sum_R = total_sum - cum_sum
        sq_L = cum_sq
        sq_R = total_sq - cum_sq

        # 预填 inf，只在 valid 位计算
        losses = np.full((n, p), np.inf)
        v = valid  # 别名
        mse_L = sq_L[v] / n_L[v] - (sum_L[v] / n_L[v]) ** 2
        mse_R = sq_R[v] / n_R[v] - (sum_R[v] / n_R[v]) ** 2
        losses[v] = (n_L[v] * mse_L + n_R[v] * mse_R) / n

        # ---- Step 5: 惩罚项 ----
        phi = np.ones(p)
        if self.used_global_feats:
            phi[list(self.used_global_feats)] = 0
        F = np.ones(p)
        if self.used_tree_feats:
            F[list(self.used_tree_feats)] = 0
        losses += (self.mu * phi * F)[None, :]  # (n, p)

        # ---- Step 6: 全局最优 ----
        flat = np.argmin(losses)
        best_row, best_feat = divmod(flat, p)
        best_loss = losses[best_row, best_feat]

        if np.isinf(best_loss):
            return Node(depth, value=np.mean(grad_active))

        best_threshold = sx[best_row, best_feat]
        self.used_tree_feats.add(best_feat)

        # ---- 生成左右子集 mask（不拷贝数据） ----
        left_mask = active_mask & (self._X[:, best_feat] <= best_threshold)
        right_mask = active_mask & (self._X[:, best_feat] > best_threshold)

        left = self._find_split(left_mask, depth + 1)
        right = self._find_split(right_mask, depth + 1)

        return Node(depth, left, right, best_feat, None, best_threshold)


class StructuredTreeLearner(TreeLearner):
    """Structured GBFS 树：惩罚按 bag 粒度，而非单个特征。

    ϕ_f = 1  当且仅当特征 f 所属的 bag 中没有任何特征被全局使用过（新 bag）。
    ϕ_f = 0  当 f 所属的 bag 已被打开过。
    """

    def __init__(self, max_depth, mu, bags):
        """
        Args:
            bags: 长度为 p 的数组，bags[f] = bag_id，表示特征 f 属于哪个 bag。
        """
        super().__init__(max_depth, mu)
        self.bags = np.asarray(bags)
        self.used_global_bags = set()

    def _find_split(self, active_mask, depth):
        active_idx = np.where(active_mask)[0]
        n = len(active_idx)
        p = self._n_feats

        if depth >= self.depth or n <= 1:
            return Node(depth, value=np.mean(self._grad[active_idx]))

        grad_active = self._grad[active_mask]
        total_sum = np.sum(grad_active)
        total_sq = np.sum(grad_active**2)

        # ---- Step 1: argsort 技巧构造 (n, p) 稠密矩阵 ----
        sorted_active = active_mask[self._sorted_idx]
        order = np.argsort(sorted_active, axis=0, kind="stable")
        active_rows = order[-n:, :]

        sx = np.take_along_axis(self._X_sorted, active_rows, axis=0)
        sg = np.take_along_axis(self._grad_sorted, active_rows, axis=0)

        # ---- Step 2: 2D cumsum ----
        cum_sum = np.cumsum(sg, axis=0)
        cum_sq = np.cumsum(sg**2, axis=0)
        cum_n = np.arange(1, n + 1, dtype=np.float64)[:, None] * np.ones(p)

        # ---- Step 3: valid mask ----
        valid = np.ones((n, p), dtype=bool)
        valid[1:, :] = sx[1:, :] != sx[:-1, :]
        valid[-1, :] = False

        # ---- Step 4: 批量 MSE ----
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
        losses[v] = (n_L[v] * mse_L + n_R[v] * mse_R) / n

        # ---- Step 5: bag 级惩罚项 ----
        phi = np.ones(p)
        if self.used_global_bags:
            phi[np.isin(self.bags, list(self.used_global_bags))] = 0
        F = np.ones(p)
        if self.used_tree_feats:
            F[list(self.used_tree_feats)] = 0
        losses += (self.mu * phi * F)[None, :]

        # ---- Step 6: 全局最优 ----
        flat = np.argmin(losses)
        best_row, best_feat = divmod(flat, p)
        best_loss = losses[best_row, best_feat]

        if np.isinf(best_loss):
            return Node(depth, value=np.mean(grad_active))

        best_threshold = sx[best_row, best_feat]
        self.used_tree_feats.add(best_feat)
        self.used_global_feats.add(best_feat)
        self.used_global_bags.add(int(self.bags[best_feat]))

        # ---- 生成左右子集 mask ----
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
            else:
                return _predict_single(x, node.right)

        return np.array([_predict_single(x, self.root) for x in X])
