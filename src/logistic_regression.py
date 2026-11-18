import numpy as np
from src.metrics import log_loss

def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    result = np.zeros_like(z)
    mask = z >= 0
    result[mask] = 1 / (1 + np.exp(-z[mask]))
    result[~mask] = np.exp(z[~mask]) / (1 + np.exp(z[~mask]))
    
    return result

class MyLogisticRegression:
    """Binary logistic regression trained with gradient descent (log loss).

    Parameters:
        lr: learning rate.
        n_iters: number of gradient descent steps.

    Attributes (after fit):
        w: weights, shape (n,).
        b: bias, a single number.
        losses: log loss on each iteration, list of length n_iters.
    """
    
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.losses = []
        
    def predict_proba(self, X):
        """Probability of class 1. X: (m, n). Returns array of shape (m,)."""
        
        if self.w is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        z = X @ self.w + self.b
        return _sigmoid(z)
    
    def _compute_gradients(self, X, y):
        p = self.predict_proba(X)
        error = p - y
        
        dw = (X.T @ error) / len(y)
        db = np.mean(error)
        
        return dw, db
    
    def fit(self, X, y):
        """Train with gradient descent. X: (m, n), y: (m,) of 0/1. Returns self."""
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.losses = []
        
        for _ in range(self.n_iters):
            dw, db = self._compute_gradients(X, y)
            
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

            self.losses.append(log_loss(y, self.predict_proba(X)))
            
        return self
    
    def predict(self, X, threshold=0.5):
        """Predict classes 0/1 by threshold. X: (m, n). Returns int array of shape (m,)."""
        p = self.predict_proba(X)
        mask = p >= threshold
        
        return mask.astype(int)