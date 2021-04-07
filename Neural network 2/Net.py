import numpy as np
from ProgressBar import ProgressBar


class CNN(object):
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
            indicies = np.arange(inputs.shape[0])
            np.random.shuffle(indicies)
            inputs = inputs[indicies]
            targets = targets[indicies]
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
                ProgressBar.printProgressBar(
                    i*samples+j+1, epochs*samples, 'Training progress', f'Error: {(err/j if j != 0 else 0):.5f}', length=50)

            # calculate average error on all samples
            err /= samples


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


class RNN(object):
    # This is code I took from https://datascience-enthusiast.com/DL/Building_a_Recurrent_Neural_Network-Step_by_Step_v1.html
    # also there is no backpropogation
    def __init__(self, n_inputs, n_hidden, n_output, Activation, Actiavation_output):
        self.params = {}
        self.params["Wf"] = np.random.randn(n_hidden, n_hidden + n_inputs)
        self.params["bf"] = np.random.randn(n_hidden, 1)
        self.params["Wi"] = np.random.randn(n_hidden, n_hidden + n_inputs)
        self.params["bi"] = np.random.randn(n_hidden, 1)
        self.params["Wc"] = np.random.randn(n_hidden, n_hidden + n_inputs)
        self.params["bc"] = np.random.randn(n_hidden, 1)
        self.params["Wo"] = np.random.randn(n_hidden, n_hidden + n_inputs)
        self.params["bo"] = np.random.randn(n_hidden, 1)
        self.params["Wy"] = np.random.randn(n_output, n_hidden)
        self.params["by"] = np.random.randn(n_output, 1)
        self.act = Activation
        self.act_out = Actiavation_output
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

    def cell_forward(self, xt, a_prev, c_prev):
        Wf = self.params["Wf"]
        bf = self.params["bf"]
        Wi = self.params["Wi"]
        bi = self.params["bi"]
        Wc = self.params["Wc"]
        bc = self.params["bc"]
        Wo = self.params["Wo"]
        bo = self.params["bo"]
        Wy = self.params["Wy"]
        by = self.params["by"]
        Activation = self.act
        Activation_output = self.act_out

        # Retrieve dimensions from shapes of xt and Wy
        n_x, m = xt.shape
        n_y, n_a = Wy.shape

        # Concatenate a_prev and xt (≈3 lines)
        concat = np.zeros((n_a + n_x, m))
        concat[: n_a, :] = a_prev
        concat[n_a:, :] = xt

        # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
        ft = Activation(np.dot(Wf, concat) + bf)
        it = Activation(np.dot(Wi, concat) + bi)
        cct = Activation(np.dot(Wc, concat) + bc)
        c_next = ft * c_prev + it * cct
        ot = Activation(np.dot(Wo, concat) + bo)
        a_next = ot * Activation(c_next)

        # Compute prediction of the LSTM cell (≈1 line)
        yt_pred = Activation_output(np.dot(Wy, a_next) + by)

        return a_next, c_next, yt_pred

    def forward(self, x, a0):
        # Retrieve dimensions from shapes of x and Wy (≈2 lines)
        n_x, m, T_x = x.shape
        n_y, n_a = self.params["Wy"].shape

        # initialize "a", "c" and "y" with zeros (≈3 lines)
        a = np.zeros((n_a, m, T_x))
        c = a
        y = np.zeros((n_y, m, T_x))

        # Initialize a_next and c_next (≈2 lines)
        a_next = a0
        c_next = np.zeros(a_next.shape)

        # loop over all time-steps
        for t in range(T_x):
            # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
            a_next, c_next, yt = self.cell_forward(x[:, :, t], a_next, c_next)
            # Save the value of the new "next" hidden state in a (≈1 line)
            a[:, :, t] = a_next
            # Save the value of the prediction in y (≈1 line)
            y[:, :, t] = yt
            # Save the value of the next cell state (≈1 line)
            c[:, :, t] = c_next

        return y
