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

