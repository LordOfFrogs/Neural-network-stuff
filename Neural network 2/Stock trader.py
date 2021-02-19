from pandas._libs.tslibs import Period
from pandas_datareader.quandl import QuandlReader
from Net import Network, ActivationLayer, FCLayer
from ActivationsLosses import Activations, Loss
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt 
import datetime
import pandas as pd

money = 500.00

API_KEY = 'NONCIQQR4FHVRL2B'

# Get the data for Coca-cola 
ts = TimeSeries(key=API_KEY, output_format='pandas')
data, metadata = ts.get_intraday(symbol='MSFT', interval='1min', outputsize='full')

dayData = data['4. close']['2021-02-18']
min = np.min(np.array(dayData))
dayData-=min
max = np.max(np.array(dayData))
dayData /= max
#plt.title('Intraday Times Series for the MSFT stock (1 min)')
#plt.show()

net = Network(Loss.MSE, Loss.MSE_der)
net.add(FCLayer(548, 15))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))
net.add(FCLayer(15, 15))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))
net.add(FCLayer(15, 1))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))

net.fit(np.array([[dayData[1:]]]), np.array([dayData[0]]), 0.1, 30)
print("True: " + str((dayData[0] + min) *max))
print("preidicted: " + str((net.forward(np.array([[dayData[1:]]]))[0][0][0] + min) * max))
