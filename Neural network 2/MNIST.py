from keras.datasets import mnist
from keras.utils import np_utils
from Net import Network, ActivationLayer, FCLayer
from ActivationsLosses import Activations, Loss
import numpy as np

#fetch data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
X_train = X_train.reshape(X_train.shape[0], 1, 28*28)
X_train = X_train.astype('float32')
X_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
X_test = X_test.reshape(X_test.shape[0], 1, 28*28)
X_test = X_test.astype('float32')
X_test /= 255
y_test = np_utils.to_categorical(y_test)

net = Network(Loss.MSE, Loss.MSE_der)
net.add(FCLayer(28*28, 15))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))
net.add(FCLayer(15, 15))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))
net.add(FCLayer(15, 10))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))

net.backpropogate(X_train, y_train, 0.1, 30)

print("Predicted:")
rand = np.random.randint(0, X_test.shape[0] - 4)
print(net.forward(X_test[rand:rand+3]))
print("True:")
print(y_test[rand:rand+3])