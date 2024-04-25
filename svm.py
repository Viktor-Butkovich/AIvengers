import numpy as np

class MultiClassSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.classifiers = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            # Create a binary target variable for each class
            y_binary = np.where(y == c, 1, -1)
            svm = SVM(self.lr, self.lambda_param, self.n_iters)
            svm.fit(X, y_binary)
            self.classifiers.append(svm)

    def predict(self, X):
        # Predict across all classifiers
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        # Use argmax to select the class with the highest confidence
        return self.classes[np.argmax(predictions, axis=0)]

class SVM:
    def __init__(self, learning_rate, lambda_param, n_iters):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        return np.dot(X, self.w) - self.b
