import numpy as np


# step layers
class ReLu:
    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1 / (1 + np.exp(- x))
        self.out = out

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout * (1.0 - self.out) * self.out

        return dx

