import numpy as np

class MyLinearRegression:

    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.losses = []

        for _ in range(self.n_iters):
            y_pred = X @ self.w + self.b
            error = y_pred - y
            dw = X.T @ error / X.shape[0]
            db = np.mean(error)

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
            self.losses.append(np.mean(error ** 2))

        return self

    def predict(self, X):
        return X @ self.w + self.b 

