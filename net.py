from functions import *
from layers import *


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
        self.params: np.ndarray = np.array([])
        self.layers: dict = {}
        self.lastLayer = None


    def predict(self, x: np.ndarray):
        return


    def loss(self, x: np.ndarray, t: np.ndarray) -> np.float:
        return


   def accracy(self, x: np.ndarray) -> np.float:
       return