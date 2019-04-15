from functions import *


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


class MultiLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy


apple = 100
apple_num = 2
tax = 1.1
layer = MultiLayer()

apple_price = layer.forward(apple, apple_num)
price = layer.forward(apple_price, tax)
print(price)

dprice = 1
dapple_price, dtax = layer.backward(dprice)
dapple, dapple_num = layer.backward(dapple_price)
print(dapple, dapple_num, tax)
