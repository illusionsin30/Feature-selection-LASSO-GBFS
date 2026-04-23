import numpy as np

class Node:
    def __init__(
        self,
        depth,
        left=None,
        right=None,
        feature_idx=None,
        value=None,
        threshold=None
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

    def _cal_mse(self, grad):
        if len(grad) == 0:
            return 0
        return np.mean((grad - np.mean(grad)) ** 2)

    def _find_split(self, X, grad, depth):
        num_samples, num_feats = X.shape
        if depth >= self.depth or num_samples <= 1:
            node = Node(depth, value=-np.mean(grad))
            return node
        
        best_loss = float('inf')
        best_feat = None
        best_split = None
        best_threshold = None

        for f in range(num_feats):
            phi = 1 if f not in self.used_global_feats else 0
            F = 1 if f not in self.used_tree_feats else 0
            penalty = self.mu * phi * F
            thresholds = np.unique(X[:, f])
            for t in thresholds:
                left = X[:, f] <= t
                right = ~left
                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue

                mse = (np.sum(left) / num_samples) * self._cal_mse(grad[left]) + (np.sum(right) / num_samples) * self._cal_mse(grad[right])
                loss = mse + penalty
                if loss < best_loss:
                    best_loss = loss
                    best_feat = f
                    best_threshold = t
                    best_split = (left, right)
            
        if best_split is None:
            return Node(depth, value=-np.mean(grad))
        
        self.used_tree_feats.add(best_feat)
        left = self._find_split(X[best_split[0]], grad[best_split[0]], depth=depth+1)
        right = self._find_split(X[best_split[1]], grad[best_split[1]], depth=depth+1)

        return Node(depth, left, right, best_feat, None, best_threshold)
    
    def fit(self, X, grad):
        self.used_tree_feats = set()
        self.root = self._find_split(X, grad, depth=0)

        return self.root
