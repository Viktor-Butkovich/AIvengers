import numpy as np
import pandas as pd
from multiprocessing import Pool
import itertools

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        if isinstance(y, pd.Series):
            y = y.values
        self.tree = self._build_tree(X, y)

    def _build_tree(self, data, labels, depth=0):
        n_samples, n_features = data.shape
        if n_samples < self.min_samples_split or depth >= self.max_depth:
            return np.bincount(labels).argmax()

        best_feature, best_threshold = self._find_best_split(data, labels)
        if best_feature is None:
            return np.bincount(labels).argmax()

        left_mask = data[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_child = self._build_tree(data[left_mask], labels[left_mask], depth + 1)
        right_child = self._build_tree(data[right_mask], labels[right_mask], depth + 1)
        return (best_feature, best_threshold, left_child, right_child)

    def _find_best_split(self, data, labels):
        best_gini = np.inf
        best_feature, best_threshold = None, None
        for feature in range(data.shape[1]):
            thresholds, classes = zip(*sorted(zip(data[:, feature], labels)))
            left_counts = np.zeros(np.max(labels)+1)
            right_counts = np.bincount(labels)
            n_left = 0
            n_right = len(labels)
            
            for i in range(1, len(labels)):
                c = classes[i-1]
                left_counts[c] += 1
                right_counts[c] -= 1
                n_left += 1
                n_right -= 1

                gini_left = 1.0 - np.sum((left_counts / n_left) ** 2)
                gini_right = 1.0 - np.sum((right_counts / n_right) ** 2)
                gini = (n_left / len(labels)) * gini_left + (n_right / len(labels)) * gini_right

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
        return best_feature, best_threshold

    def predict(self, sample):
        node = self.tree
        while isinstance(node, tuple):
            feature, threshold, left, right = node
            node = left if sample[feature] <= threshold else right
        return node

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, sample_size_ratio=0.8, n_jobs=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sample_size_ratio = sample_size_ratio
        self.n_jobs = n_jobs
        self.trees = []

    def fit(self, X, y):
        n_samples = int(self.sample_size_ratio * len(X))
        pool = Pool(self.n_jobs)
        results = pool.starmap(self._fit_tree, [(X, y, n_samples) for _ in range(self.n_estimators)])
        self.trees = results

    def _fit_tree(self, X, y, n_samples):
        indices = np.random.choice(len(X), n_samples, replace=True)
        sample_X, sample_y = X[indices], y.iloc[indices]
        tree = DecisionTree(self.max_depth)
        tree.fit(sample_X, sample_y)
        return tree

    def predict(self, X):
        pool = Pool(self.n_jobs)
        predictions = np.array(pool.map(self._predict_tree, [(tree, X) for tree in self.trees]))
        predictions = predictions.T
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=2).argmax(), axis=1, arr=predictions)
        return majority_votes

    def _predict_tree(self, args):
        tree, X = args
        return [tree.predict(x) for x in X]
