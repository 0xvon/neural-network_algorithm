import numpy as np

NoneType = type(None)

class SGD:
    def __init__(self, lr: float = 0.01):
        self.lr: float = lr

    def update(self, params: dict, grads: np.ndarray):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr: float = lr
        self.momentum: float = momentum
        self.v: dict or NoneType = None

    def update(self, params: dict, grads: np.ndarray):
        if self.v is None:
            self.v = {}
            for key, value in params.items():
                self.v[key] = np.zeros_like(value)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr: float = 0.01):
        self.lr: float = lr
        self.h: dict or NoneType = None

    def update(self, params: dict, grads: np.ndarray):
        if self.h is None:
            self.h = {}
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + 1e-7)


class RMSprop:
    def __init__(self, lr: float = 0.01, decay_rate: float = 0.99):
        self.lr: float = lr
        self.decay_rate: float = decay_rate
        self.h: dict or None = None

    def update(self, params: dict, grads: np.ndarray):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
        self.lr: float = lr
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.iter: int = 0
        self.m: dict or NoneType = None
        self.v: dict or NoneType = None

    def update(self, params: dict, grads: np.ndarray):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t: float = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
