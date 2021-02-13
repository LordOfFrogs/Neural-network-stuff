from keras.utils.np_utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from os import error
import numpy as np
import matplotlib.pyplot as plt

#fetch data
mnist = fetch_openml('mnist_784', version=1)
x, y = np.array(mnist["data"]), np.array(mnist["target"])

x = (x/255).astype('float32')
y = to_categorical(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

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
    def __init__(self, n_inputs, n_hiddens, n_hiddenLayers, n_outputs, activation, activation_der, output_activation, outactivation_der, Loss):
        self.Loss = Loss
        self.activation = activation
        self.output_activation = output_activation
        self.activation_der = activation_der
        self.output_activation_der = outactivation_der
        
        # initialize layers
        self.hiddenLayers = []
        for i in range(n_hiddenLayers - 1):
            self.hiddenLayers.append(Layer(n_hiddens, n_hiddens, activation))
        self.hiddenLayers.insert(0, Layer(n_inputs, n_hiddens, activation))
        self.outputs = Layer(n_hiddens, n_outputs, output_activation)

    def forward(self, inputs):
        # forward feed each layer
        self.hiddenLayers[0].forward(inputs)
        for i in range(1, len(self.hiddenLayers)):
            self.hiddenLayers[i].forward(self.hiddenLayers[i-1].activatedOutput)
        self.outputs.forward(self.hiddenLayers[-1].activatedOutput)
        self.output = self.outputs.activatedOutput

    def backpropogate(self, inputs, targets, learningRate, epochs):
        self.forward(inputs)
        for epoch in range(epochs):
            #output
            error = 2 * (self.output - y_train) / self.output.shape[0] * self.output_activation_der(self.outputs.output)
            print(error.shape)
            print(self.hiddenLayers[-1].activatedOutput.shape)
            dW = np.outer(error, self.hiddenLayers[-1].activatedOutput)
            weights = self.outputs.weights
            self.outputs.weights -= dW * learningRate
            
            #hidden layers
            for i in range(len(self.hiddenLayers), 1):
                error = np.dot(weights.T, error) * self.activation_der(self.hiddenLayers[i].output)
                dW = np.outer(error, self.hiddenLayers[i - 1].activatedOutput)
                weights = self.hiddenLayers[i].weights
                self.hiddenLayers[i].weights -= dW * learningRate
            
            #first hidden layer
            error = np.dot(weights.T, error) * self.activation_der(self.hiddenLayers[i].output)
            dW = np.outer(error, inputs)
            weights = self.hiddenLayers[0].weights
            self.hiddenLayers[0].weights -= dW * learningRate
            
            self.forward(inputs)
            print("Epoch: " + str(epoch) + " Cost: " + str(self.Loss(targets, self.output)))
                


class Layer(object):
    def __init__(self, n_inputs, n_neurons, activation):
        # init weights as random and biases as 0s
        self.weights = 0.1*np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def forward(self, inputs):
        # run layer
        self.output = np.dot(self.weights, inputs) + self.biases
        self.activatedOutput = self.activation(self.output)


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

def CrossEntropy(y, yhat):
    L_sum = np.sum(np.multiply(y, np.log(yhat)))
    m = y.shape[1]
    L = -(1/m) * L_sum

    return L

def MAE(y, yhat):
    return np.mean(np.abs(y - yhat))

net = Network(X_train.shape[1], 10, 2, 10, Sigmoid, Sigmoid_der, Softmax, Softmax_der, MAE)

net.backpropogate(X_train, y_train, 0.1, 1000)

losses = []
net.forward(X_test)
print(np.mean(net.Loss(y_test, net.output)))

image = X_test[np.random.randint(0, m_test)]
pixels = np.reshape(image, (28, 28))
plt.imshow(pixels, cmap='gray')
net.forward(np.array([image]))
print(net.output)
plt.show()