from time import time

from bitfinex.bitfinex_v2 import api_v2
from numpy.core.fromnumeric import shape
from Net import Network, ActivationLayer, FCLayer
from ActivationsLosses import Activations, Loss
import numpy as np
import bitfinex
from datetime import datetime as dt, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from ProgressBar import ProgressBar
from pytz import timezone
import time


def unix_time_millis(dt_obj):
    return dt_obj.timestamp() * 1000
def flatten_list_2d(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        for item in element:
            flat_list.append(item)
    return flat_list

def fetch_data(start, stop, symbol, interval='1m', tick_limit=1000, step=60000000):
    api = bitfinex.bitfinex_v2.api_v2()

    data = []
    start = start - step
    while start < stop:
        start += step
        end = start + step
        res = api.candles(symbol=symbol, interval=interval,
                          limit=tick_limit, start=start, end=end)
        data.extend(res)
        print('Retrieving data from {} to {} for {}'.format(pd.to_datetime(
            start, unit='ms'), pd.to_datetime(end, unit='ms'), symbol))
        time.sleep(1)
    # Remove error messages
    # print(data)
    ind = [np.ndim(x) != 0 for x in data]
    data = [i for (i, v) in zip(data, ind) if v]

    # Create pandas data frame and clean data
    names = ['time', 'open', 'close', 'high', 'low', 'volume']
    df = pd.DataFrame(data, columns=names)
    df.drop_duplicates(inplace=True)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    data = list(df['close'])
    return data


startingMoney = 5000.00
money = startingMoney
shares = 0
ticker = 'BTCUSD'
risk = 5
n_inputs = 5000
training_len = 3

raw_data = fetch_data(start=unix_time_millis(dt.utcnow() - timedelta(days=20)),
                      stop=unix_time_millis(dt.utcnow()), symbol='btcusd')
raw_data = raw_data[:training_len*n_inputs]
print('Download Complete')

X_train = []
y_train = []

for i in range(n_inputs-1, len(raw_data)-1):
    X_train.append(raw_data[i-n_inputs-1:i-1])
    y_train.append(raw_data[i])
    ProgressBar.printProgressBar(i-n_inputs+1, len(raw_data)-n_inputs,
                                 prefix='Preparing Data: {}/{}'.format(i-n_inputs+1, len(raw_data)-n_inputs), length=25)
print()
for i in range(len(X_train)):
    if len(X_train[i]) == 0:
        print("Error at: {}".format(i))
all = 0
length = 0
for i in X_train:
    length += 1
    for j in i:
        all+=1
print(all)
print(length)
inputs = 0
print(inputs)
X_train = np.array(flatten_list_2d(X_train)).reshape(training_len*n_inputs-n_inputs, n_inputs)
y_train = np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
y_train = np.reshape(y_train, (y_train.shape[0], 1))


# create network
net = Network(Loss.MSE, Loss.MSE_der)
net.add(FCLayer(X_train.shape[0], 15))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))
net.add(FCLayer(15, 15))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))
net.add(FCLayer(15, 1))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))
net.fit(X_train, y_train, 0.1, 30)

# run
prevData = np.zeros((n_inputs-1))
while True:
    data = fetch_data(unix_time_millis(dt.utcnow()-timedelta(days=1) +
                                       timedelta(minutes=1)), unix_time_millis(dt.now()), ticker)
    if(prevData[0] != data[0]):
        min = np.min(data)
        X = data - min
        max = np.max(X)
        X /= max
        predicted_norm = net.forward(np.array([[X]]))[0][0][0]
        print(predicted_norm)
        prediction = (predicted_norm * max) + min
        if(prediction - data[0] > 0.0 and money > data[0]):
            money -= data[0]*risk
            stocks += risk
        elif(prediction - data[0] < 0.0 and stocks > 0):
            money += data[0]*risk
            stocks -= risk
        prevData = np.copy(data)
        print(money - startingMoney + stocks*data[0])
