from time import time
from Net import Network, ActivationLayer, FCLayer
from ActivationsLosses import Activations, Loss
import numpy as np
import yfinance as yf
from datetime import datetime as dt, timedelta
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from ProgressBar import ProgressBar

money = 500.00

API_KEY = 'NONCIQQR4FHVRL2B'

#get data
historical_data = []
mins = []
maxes = []
for i in range(100):
    historical_data.append(yf.download("MSFT", end=dt.today() - timedelta(days=i), interval='1m', period='1d', progress=False).Close.to_numpy())
    ProgressBar.printProgressBar(i, 100, 'Downloading Stock Data: ', length=50)
print('\nDownload Complete')

#prepare data
for i in range(len(historical_data)):
    mins.append(np.min(historical_data[i]))
    historical_data[i] -= mins[i]
    maxes.append(np.max(historical_data[i]))
    historical_data[i] /= maxes[i]

X = []
y = []
for i in historical_data:
    X.append([i[1:]])
    y.append([i[0]])
X = np.array(X)
y = np.array(y)
m = 100
X_train = X[:m]
X_test = X[m:]
y_train = y[:m]
y_test = y[m:]

#create network
net = Network(Loss.MSE, Loss.MSE_der)
net.add(FCLayer(388, 15))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))
net.add(FCLayer(15, 15))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))
net.add(FCLayer(15, 1))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))

net.fit(X_train, y_train, 0.1, 30)

#test
