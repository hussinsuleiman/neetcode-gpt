import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        pass

        n = len(y_true)
        tot = 0 
        EPS = 10**(-7)

        for i in range(n):
            tot += y_true[i] * math.log(y_pred[i] + EPS) + (1-y_true[i]) * math.log(1-y_pred[i]-EPS)
        
        return round(-tot/n, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        pass

        n, C = len(y_true), len(y_true[0])
        tot = 0
        EPS = 10**(-7)

        for i in range(n):
            for j in range(C):
                tot += y_true[i][j] * math.log(y_pred[i][j] + EPS)
        
        return round(-tot/n, 4)