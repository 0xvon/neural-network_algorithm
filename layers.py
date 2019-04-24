import numpy as np
from common.functions import *


# to instant types "function" and  "NoneType"
def a():
    print()


NoneType = type(None)
function = type(a)


class MultiLayer:
    def __init__(self):
        self.x: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        out: np.ndarray = x * y

        return out

    def backward(self, dout: np.ndarray):
        dx: np.ndarray = dout * self.y
        dy: np.ndarray = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out: np.ndarray = x + y

        return out

    def backward(self, dout: np.ndarray):
        dx: np.ndarray = dout * 1
        dy: np.ndarray = dout * 1

        return dx, dy


# step layers
class ReLu:
    def __init__(self):
        self.mask: np.ndarray = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x <= 0)
        out: np.ndarray = x
        out[self.mask] = 0

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout[self.mask] = 0
        dx: np.ndarray = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out: np.ndarray = np.array([])

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
        self.x: np.ndarray = np.array([])
        self.dW: np.ndarray = np.array([])
        self.db: np.ndarray = np.array([])

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
        self.loss: float = 1
        self.y: np.ndarray = np.array([])
        self.t: np.ndarray = np.array([])

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        self.y = y
        self.t = t
        self.loss = cross_entropy_error(y, t)

        return self.loss

    def backward(self, dout: int = 1) -> np.ndarray:
        batch_size: int = self.t.shape[0]
        dx: np.ndarray = (self.y - self.t) / batch_size

        return dx


class Convolution:
    def __init__(self, W: np.ndarra, b: np.ndarra, stride: int = 1, pad: int = 0):
        self.W: np.ndarray = W
        self.b: np.ndarray = b
        self.stride: int = stride
        self.pad: int = pad

    def forwared(self, x: np.ndarray) -> np.ndarray:
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        out_h: int = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w: int = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col: np.ndarray = im2col(x, FH, FW, self.stride, self.pad)
        col_W: np.ndarray = self.W.reshape(FN, -1).T
        out: np.ndarray = np.dot(col, col_W) + self.b

        out: np.ndarray = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out


class Pooling:

    def __init__(self, pool_h: int, pool_w: int, stride: int = 1, pad: int = 0):
        self.pool_h: int = pool_h
        self.pool_w: int = pool_w
        self.stride: int = stride
        self.pad: int = pad

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        out_h: int = int(1 + (H - self.pool_h) / self.stride)
        out_w: int = int(1 + (W - self.pool_w) / self.stride)

        col: np.ndrray = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        out: np.ndarray = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out
