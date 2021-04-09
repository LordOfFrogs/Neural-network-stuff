import numpy as np


class Activations:
    def Sigmoid(x):
        z = 1.
        try:
            z =  np.where(x >= 0,
                            1 / (1 + np.exp(-x)),
                            np.exp(x) / (1 + np.exp(x)))
        except:
            print(x)
        return z

    def Sigmoid_der(x):
        return Activations.Sigmoid(x)*(1-Activations.Sigmoid(x))

    def Softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    def tanh(x):
        return np.tanh(x)

    def tanh_der(x):
        return 1-np.tanh(x)**2


class Loss:
    def CrossEntropy(y, yhat):
        L_sum = np.sum(np.multiply(y, np.log(yhat)))
        m = y.shape[1]
        L = -(1/m) * L_sum

        return L

    def MSE(y, yhat):
        return np.mean(np.abs(y - yhat))

    def MSE_der(y, yhat):
        return 2*(yhat-y)/y.size
