import numpy as np


# step
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    c: np.int = np.max(x)
    return np.exp(x - c) / np.sum(np.exp(x - c))


def step(x: np.ndarray) -> np.ndarray:
    return np.array(x > 0, dtype=np.int)


def identity(x: np.ndarray) -> np.ndarray:
    return x


# loss
def mean_squad_error(y: np.ndarray, t: np.ndarray) -> np.int:
    return 0.5 * np.sum((y - t) ** 2)


# in the case of one-hot expression
def cross_entropy(y: np.ndarray, t: np.ndarray) -> np.int:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return - np.sum(t * np.log(y + delta)) / batch_size


# slope
def numerical_gradient(f, x: np.ndarray) -> np.ndarray:
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def gradient_descent(f, init_x: np.ndarray, lr=0.01, step_num=100) -> np.ndarray:
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
