from sklearn.datasets import fetch_openml
from os import error
import numpy as np
import matplotlib.pyplot as plt

#fetch data
mnist = fetch_openml('mnist_784', version=1)
X, y = np.array(mnist["data"]), np.array(mnist["target"])

X /= 255 #normalize
digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)

Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)

m = 60000 #how many are training vs test
m_test = X.shape[0] - m

X_train, X_test = X[:m].T, X[m:]
y_train, y_test = Y_new[:,:m], Y_new[:,m:].T

shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:, shuffle_index].T, y_train[:, shuffle_index].T

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
    def __init__(self, n_inputs, n_hiddens, n_outputs, activation, activation_der, output_activation, outactivation_der, Loss):
        self.Loss = Loss
        self.activation = activation
        self.output_activation = output_activation
        self.activation_der = activation_der
        self.output_activation_der = outactivation_der
        
        # initialize layers
        self.hiddenLayer = Layer(n_inputs, n_hiddens, activation)
        self.outputs = Layer(n_hiddens, n_outputs, output_activation)

    def forward(self, inputs):
        # forward feed each layer
        self.hiddenLayer.forward(inputs)
        self.outputs.forward(self.hiddenLayer.activatedOutput)
        self.output = self.outputs.activatedOutput

    def backpropogate(self, inputs, targets, learningRate, epochs):
        self.forward(inputs)
        for epoch in range(epochs):
            #outputs
            y_error = targets - self.output
            dy = y_error * self.output_activation_der(self.output)
            
            z_error = np.dot(dy, self.outputs.weights.T)
            dz = z_error * self.activation_der(self.hiddenLayer.activatedOutput)
            self.hiddenLayer.weights += np.dot(inputs.T, dz) * learningRate
            self.outputs.weights += np.dot(self.hiddenLayer.activatedOutput.T, dy) * learningRate
            
            self.forward(inputs)
            print("Epoch: " + str(epoch) + " Cost: " + str(self.Loss(targets, self.output)))
                


class Layer(object):
    def __init__(self, n_inputs, n_neurons, activation):
        # init weights as random and biases as 0s
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def forward(self, inputs):
        # run layer
        self.output = np.dot(inputs, self.weights) + self.biases
        self.activatedOutput = self.activation(self.output)


# activations 
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))
def Sigmoid_der(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True) 
def Softmax_der(x):
    return (np.exp(x)*np.sum(np.exp(x), axis=1, keepdims=True) - np.exp(2*x))/(np.sum(np.exp(x))**2)


def CrossEntropy(y, yhat):
    L_sum = np.sum(np.multiply(y, np.log(yhat)))
    m = y.shape[1]
    L = -(1/m) * L_sum

    return L

def MAE(y, yhat):
    return np.mean(np.abs(y - yhat))


net = Network(X_train.shape[1], 10, 10, Sigmoid, Sigmoid_der, Softmax, Softmax_der, MAE)

net.backpropogate(X_train, y_train, 1000000, 1000)

losses = []
net.forward(X_test)
print(np.mean(net.Loss(y_test, net.output)))

image = X_test[np.random.randint(0, m_test)]
pixels = np.reshape(image, (28, 28))
plt.imshow(pixels, cmap='gray')
net.forward(np.array([image]))
print(net.output)
plt.show()