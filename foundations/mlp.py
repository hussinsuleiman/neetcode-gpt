import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        pass

        x = np.array(x)
        n = len(weights)

        for i in range(n-1):
            x = x @ weights[i] + biases[i]
            x = np.maximum(x, 0.0)
        
        x = x @ weights[n-1] + biases[n-1]
        return np.round(x, 5).tolist()