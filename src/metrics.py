import numpy as np

def mean_absolute_error(y_valid, pred):
    return np.mean(np.abs(y_valid - pred))

def mean_squared_error(y_valid, pred):
    return np.mean((y_valid - pred) ** 2)