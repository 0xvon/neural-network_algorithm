import numpy as np
from functions import *


# step layers
class ReLu:
    def __init__(self):
        self.mask: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x <= 0)
        out: np.ndarray = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout[self.mask] = 0
        dx: np.ndarray = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out: np.ndarray = 1 / (1 + np.exp(- x))
        self.out = out

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx: np.ndarray = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, w: np.ndarray, b: np.ndarray):
        self.w: np.ndarray = w
        self.b: np.ndarray = b
        self.x: np.ndarray = None
        self.dW: np.ndarray = None
        self.db: np.ndarray = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        out: np.ndarray = np.dot(x, self.w) + self.b

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx: np.ndarray = np.dot(dout, self.w.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss: np.ndarray = None
        self.y: np.ndarray = None
        self.t: np.ndarray = None

    def forward(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        self.y = y
        self.t = t
        self.loss = cross_entropy(y, t)

        return self.loss

    def backward(self, dout: int = 1) -> np.ndarray:
        batch_size: int = self.t.shape[0]
        dx: np.ndarray = (self.y - self.t) / batch_size

        return dx
