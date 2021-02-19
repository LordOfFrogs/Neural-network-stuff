import numpy as np

class Activations:
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