import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        x = np.array(x, dtype=np.float64)
        W1 = np.array(W1, dtype=np.float64)
        b1 = np.array(b1, dtype=np.float64)
        W2 = np.array(W2, dtype=np.float64)
        b2 = np.array(b2, dtype=np.float64)

        y_true = np.array(y_true, dtype=np.float64)

        def relu(x): return np.maximum(0,x)

        # forward pass
        print(W1.shape)
        print(x.shape)
        print(b1.shape)
        z1 = W1@x + b1
        print(z1.shape)

        h = relu(z1)
        print(h.shape)

        print(W2.shape)
        print(b2.shape)
        z2 = W2@h + b2
        print(z2.shape)

        y_pred = z2

        print(y_pred)

        # loss
        loss = np.sum((y_pred - y_true)**2)
        print(loss)

        # backward pass
        delta2 = 2*(y_pred - y_true)
        print(delta2.shape)
        print(h.shape)
        
        dW2 = np.outer(delta2, h)
        print(dW2.shape)
        print(W2.shape)

        db2 = delta2

        delta1 = (W2.T@delta2)*(z1>0)
        print(delta1.shape)

        dW1 = np.outer(delta1, x)
        db1 = delta1
        print(dW1.shape)
        print(W1.shape)

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1,4).tolist(),
            "db1": np.round(db1,4).tolist(),
            "dW2": np.round(dW2,4).tolist(),
            "db2": np.round(db2,4).tolist()
        }

