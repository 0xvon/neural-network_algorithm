from net import TwoLayerNet
from data import *
from functions import *

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
y_batch = y_train[:3]


grad_numerical = network.numerical_gradient(x_batch, y_batch)
grad_backdrop = network.gradient(x_batch, y_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backdrop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))

