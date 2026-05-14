import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x = np.array(x)
        gamma = np.array(gamma)

        n = len(x)
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        rms_x = np.sqrt(np.sum(x**2) / n + eps)
        x_hat = x / rms_x
        y = gamma*x_hat
        return np.round(y.tolist(), 4)