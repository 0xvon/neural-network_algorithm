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
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> np.float:
        y: np.ndarray = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> np.float:
        y: np.ndarray = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acr: np.float = np.sum(y == t) / float(x.shape[0])

        return acr

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> np.float:
        loss_w: function = lambda w: self.loss(x, t)

        grads: dict = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])

        return grads

    def gradient(self, x: np.ndarray, t: np.ndarray) -> np.float:
        self.loss(x, t)

        dout: int = 1
        dout = self.lastLayer.backward(dout)

        layers: list = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
