import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        pass

        e = [math.exp(elt-max(z)) for elt in z]
        s = sum(e)
        ans = [round(elt/s,4) for elt in e]
        return ans
