import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        n_samples = X.shape[0]

        # Initialize f_m(x) for each class to 0
        f_m = np.zeros((n_samples, self.n_classes))
        self.trees = {cls: [] for cls in self.classes}

        for cls in self.classes:
            # Create initial uniform probabilities and pseudo-residuals
            y_cls = np.where(y == cls, 1, 0)
            p = np.full(n_samples, np.mean(y_cls))
            for _ in range(self.n_estimators):
                # Compute residuals as gradient of the loss (Cross-Entropy Loss Gradient)
                residuals = y_cls - p

                # Fit a tree to the residuals
                tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                tree.fit(X, residuals)
                predictions = tree.predict(X)

                # Update f_m(x) and probabilities
                f_m[:, self.classes == cls] += self.learning_rate * predictions.reshape(-1, 1)
                p = self.softmax(f_m)[:, self.classes == cls].ravel()

                # Store the tree under the current class
                self.trees[cls].append(tree)

    def softmax(self, z):
        # Compute softmax probabilities
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        # Compute f_m(x) for each class
        f_m = np.zeros((X.shape[0], self.n_classes))
        for cls in self.classes:
            for tree in self.trees[cls]:
                f_m[:, self.classes == cls] += self.learning_rate * tree.predict(X).reshape(-1, 1)

        # Compute probabilities using softmax
        probabilities = self.softmax(f_m)
        return probabilities

    def predict(self, X):
        # Get the class with the highest probability
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]
