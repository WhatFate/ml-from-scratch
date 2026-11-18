import numpy as np


class MeanBaseline:
    """Regression baseline: always predicts the mean of y_train."""

    def __init__(self):
        self.mean = None

    def fit(self, X, y):
        """Store the mean of y. X: (m, n) is unused. y: (m,). Returns self."""
        self.mean = np.mean(y)

        return self

    def predict(self, X):
        """Predict the stored mean. X: (m, n). Returns array of shape (m,)."""
        if self.mean == None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return np.full(X.shape[0], self.mean)


class MajorityBaseline:
    """Classification baseline: always predicts the most frequent class of y_train."""

    def __init__(self):
        self.majority = None

    def fit(self, X, y):
        """Store the most frequent class. X: (m, n) is unused. y: (m,). Returns self."""
        values, counts = np.unique(y, return_counts=True)
        self.majority = values[np.argmax(counts)]

        return self

    def predict(self, X):
        """Predict the stored class. X: (m, n). Returns array of shape (m,)."""
        if self.majority == None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return np.full(X.shape[0], self.majority)