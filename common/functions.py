import numpy as np


def identity_function(x: np.ndarray) -> np.ndarray:
    return x


def step_function(x: np.ndarray) -> np.ndarray:
    return np.array(x > 0, dtype=np.int)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    grad: np.ndarray = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y: np.ndarray = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y: np.ndarray, t: np.ndarray) -> float or np.float:
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y: np.ndarray, t: np.ndarray or int) -> float or np.float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size: int = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-1)) / batch_size


def softmax_loss(X: np.ndarray, t: np.ndarray or int) -> float or np.float:
    y: np.ndarray = softmax(X)
    return cross_entropy_error(y, t)


def im2col(input_data: tuple, filter_h: int, filter_w: int, stride: int=1, pad: int=0):
    N, C, H, W = input_data.shape
    out_h: int = (H + 2 * pad - filter_h) // stride + 1
    out_w: int = (W + 2 * pad - filter_w) // stride + 1

    img: np.ndarray = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col: np.ndarray = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max: int = y + stride * out_h
        for x in range(filter_w):
            x_max: int = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col: np.ndarray, input_shape: tuple, filter_h: int, filter_w: int, stride: int=1, pad: int=0):
    N, C, H, W = input_shape
    out_h: int = (H + 2 * pad - filter_h) // stride + 1
    out_w: int = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img: np.ndarray = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max: int = y + stride * out_h
        for x in range(filter_w):
            x_max: int = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
