import numpy as np

NoneType = type(None)

class Dropout:
    def __init__(self, dropout_raito: float = 0.5):
        self.dropout_ratio: float = dropout_raito
        self.mask: np.ndarray or NoneType = None

    def forward(self, x: np.ndarray, train_flg: bool = True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask

    def backward(self, dout: float):
        return dout * self.mask
