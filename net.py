import numpy as np
from functions import *
from data import *


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict()
        y = softmax(z)
        loss = cross_entropy(y, t)

        return loss


class MultiLayer:
    def __init__(self):
        self.x = None
        self.y = None

    @staticmethod
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    @staticmethod
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    @staticmethod
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x + y

        return out

    @staticmethod
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
