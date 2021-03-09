from time import time

from bitfinex.bitfinex_v2 import api_v2
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

tz = timezone('EST')
epoch = dt.fromtimestamp(0, tz=tz)


def unix_time_millis(dt):
    return time.mktime(dt.timetuple()) * 1000


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
        time.sleep(0)
        
    # Remove error messages
    ind = [np.ndim(x) != 0 for x in data]
    data = [i for (i, v) in zip(data, ind) if v]

    # Create pandas data frame and clean data
    names = ['time', 'open', 'close', 'high', 'low', 'volume']
    df = pd.DataFrame(data, columns=names)
    df.drop_duplicates(inplace=True)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    data = np.array(df['close'])
    return data


startingMoney = 5000.00
money = startingMoney
shares = 0
ticker = 'BTCUSD'
risk = 5
n_inputs = 5000

raw_data = fetch_data(start=unix_time_millis(dt.now(tz=tz) - timedelta(days=10)),
                       stop=unix_time_millis(dt.now(tz=tz)), symbol='btcusd')
print('Download Complete')

X_train = []
y_train = []
for i in range(n_inputs - 1, raw_data.shape[0]):
    X_train.append(raw_data[i-n_inputs+1:i-1])
    y_train.append(raw_data[i])
X_train = np.array(X_train)
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
    data = fetch_data(unix_time_millis(dt.now(tz=tz)-timedelta(days=1) +
                                       timedelta(minutes=1)), unix_time_millis(dt.now(tz=tz)), ticker)
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
