import numpy as np

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def log_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    
    if TP == 0:
        return 0.0
    
    FP = np.sum((y_true == 0) & (y_pred == 1))
    return TP / (TP + FP)

def recall_score(y_true, y_pred):    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    
    if TP == 0:
        return 0.0
    
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP / (TP + FN)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if (precision == 0) & (recall == 0):
        return 0.0
    
    return 2 * precision * recall / (precision + recall)