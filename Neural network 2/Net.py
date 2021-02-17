import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from os import error
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils

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

'''x, y_labels = fetch_openml('mnist_784', version=1, return_X_y=True)

x = (x/255).astype('float32')
y = np.zeros((y_labels.shape[0], 10))

for i in range(y_labels.shape[0]):
    y[i][int(y_labels[i])] = 1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)'''


'''np.random.seed(42)

cat_images = np.random.randn(700, 2) + np.array([0, -3])
mouse_images = np.random.randn(700, 2) + np.array([3, 3])
dog_images = np.random.randn(700, 2) + np.array([-3, 3])

feature_set = np.vstack([cat_images, mouse_images, dog_images])

labels = np.array([0]*700 + [1]*700 + [2]*700)

one_hot_labels = np.zeros((2100, 3))
for i in range(2100):
    one_hot_labels[i][labels[i]] = 1'''

class Network(object):
    def __init__(self, Loss, Loss_der):
        self.Loss = Loss
        self.Loss_der = Loss_der
        # initialize layers
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs):
        # sample dimension first
        samples = len(inputs)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = inputs[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def backpropogate(self, inputs, targets, learning_rate, epochs):
        # sample dimension first
        samples = len(inputs)
        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = inputs[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # compute loss (for display purpose only)
                err += self.Loss(targets[j], output)

                # backward propagation
                error = self.Loss_der(targets[j], output)
                for layer in reversed(self.layers):
                    error = layer.backpropogate(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
                
class FCLayer(object):
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.randn(1, n_neurons)

    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
    
    def backpropogate(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * output_error
        return input_error
    
class ActivationLayer(object):
    def __init__(self, activation, activation_der):
        self.activation = activation
        self.activation_der = activation_der

    def forward(self, inputs):
        self.input = inputs
        self.output = self.activation(self.input)
        return self.output
    
    def backpropogate(self, output_error, learning_rate):
        return self.activation_der(self.input) * output_error


# activations 
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))
def Sigmoid_der(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True) 
def Softmax_der(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))

def tanh(x):
    return np.tanh(x)
def tanh_prime(x):
    return 1-np.tanh(x)**2

def CrossEntropy(y, yhat):
    L_sum = np.sum(np.multiply(y, np.log(yhat)))
    m = y.shape[1]
    L = -(1/m) * L_sum

    return L

def MSE(y, yhat):
    return np.mean(np.abs(y - yhat))

def MSE_der(y, yhat):
    return 2*(yhat-y)/y.size

net = Network(MSE, MSE_der)
net.add(FCLayer(28*28, 15))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(15, 15))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(15, 10))
net.add(ActivationLayer(tanh, tanh_prime))

net.backpropogate(X_train, y_train, 0.1, 50)

print("Predicted:")
rand = np.random.randint(0, X_test.shape[0] - 4)
print(net.forward(X_test[rand:rand+3]))
print("True:")
print(y_test[rand:rand+3])
