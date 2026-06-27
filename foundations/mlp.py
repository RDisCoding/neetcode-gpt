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
        print(x.shape)
        print(weights)
        print(biases)
        print()
        layers = len(weights)
        print(layers)
        curr = x
        for l in range(layers):
            print(curr.shape)
            print(weights[l].shape)
            z = np.dot(curr, weights[l]) + biases[l]
            print(z)
            print(z.shape)
            if l+1 != layers:
                curr = np.maximum(0,z)
            else:
                curr = z
        return np.round(curr, 5)