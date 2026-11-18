# ml-from-scratch

Machine learning algorithms implemented from scratch with NumPy and compared with scikit-learn.

## What's inside

- **Linear regression**: gradient descent with MSE loss.
- **Logistic regression**: numerically stable sigmoid, log loss, and gradient descent
- **Metrics and baselines**: MSE, MAE, accuracy, precision, recall, F1, and simple baseline models.

## Results

On a synthetic dataset (1000 samples, 4 features, 30% positive class), logistic regression from scratch:

- reaches 85.5% accuracy, 76.1% precision, and 66.0% recall on the test set;
- makes the same predictions as scikit-learn on 100% of test samples;
- beats the majority-class baseline, which never finds a positive sample (recall 0%).

See the notebooks for details and plots.

## Project structure

```
ml-from-scrtch/
│
├── notebooks/                         # Comparison of my models and scikit-learn
│   ├── 01_linear_regression.ipynb
│   └── 02_logistic_regression.ipynb
│
├── src/                               # My models and metrics
│   ├── __init__.py
│   ├── baseline.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   └── metrics.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/WhatFate/ml-from-scratch
cd ml-from-scratch
pip install -r requirements.txt
```

## Usage

Train a model and make predictions:

```python
import numpy as np
from src.logistic_regression import MyLogisticRegression
from src.metrics import accuracy_score

X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = MyLogisticRegression(lr=0.1, n_iters=1000).fit(X, y)
predictions = model.predict(X)

print(accuracy_score(y, predictions))
```

Run it from the root of the repository.

## Roadmap

- Gradient checking tests
- Feature scaling
- L2 regularization
- Experiments with the learning rate