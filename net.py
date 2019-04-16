from functions import *
from layers import *
from collections import OrderedDict


# from data import *


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


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std):
        self.params: dict = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers: dict = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer: type(SoftmaxWithLoss()) = SoftmaxWithLoss()

    def predict(self, x: np.ndarray):
        return

    def loss(self, x: np.ndarray, t: np.ndarray) -> np.float:
        return


def accracy(self, x: np.ndarray) -> np.float:
    return


def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> np.float:
    return


def gradient(self, x: np.ndarray, t: np.ndarray) -> np.float:
    return
