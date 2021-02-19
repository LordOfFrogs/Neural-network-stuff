import numpy as np

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

    def fit(self, inputs, targets, learning_rate, epochs):
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