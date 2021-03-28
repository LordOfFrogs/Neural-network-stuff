from time import time

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
    return time.mktime(dt_obj.timetuple()) * 1000


def flatten_list_2d(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        for item in element:
            flat_list.append(item)
    return flat_list


def fetch_data(start, stop, symbol='btcusd', interval='1m', tick_limit=1000, step=60000000):
    # Create api instance
    api_v2 = bitfinex.bitfinex_v2.api_v2()

    data = []
    starting = start - step
    while starting < stop:

        starting = starting + step
        end = starting + step
        res = api_v2.candles(symbol=symbol, interval=interval,
                             limit=tick_limit, start=starting, end=end)
        data.extend(res)
        ProgressBar.printProgressBar(
            starting-start, stop-start, prefix='Downloading data', length=50)
        time.sleep(1)
    print()
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


# Collect the data
startingMoney = 5000.00
money = startingMoney
amount = 0.00092
ticker = 'btcusd'
risk = 5.00
n_inputs = 500
training_len = 20
raw_data = fetch_data(start=unix_time_millis(dt.utcnow() - timedelta(days=20)),
                      stop=unix_time_millis(dt.utcnow()), symbol=ticker)
print(len(raw_data))
raw_data = raw_data[-training_len*n_inputs:]
starting_total = startingMoney + amount*raw_data[-1]
print('Download Complete')

X_train = []
y_train = []

for i in range(n_inputs, len(raw_data)):
    X_train.append(raw_data[i-n_inputs:i])
    y_train.append(-1.0 if raw_data[i] < raw_data[i-1] else 1.0)
    ProgressBar.printProgressBar(i-n_inputs+1, len(raw_data)-n_inputs,
                                 prefix='Preparing Data: {}/{}'.format(i-n_inputs+1, len(raw_data)-n_inputs), length=25)

X_train = np.array(X_train)
X_train = X_train.reshape(training_len*n_inputs-n_inputs, n_inputs)
y_train = np.array(y_train)

for i in range(X_train.shape[0]):
    min = np.min(X_train[i])
    X_train[i] -= min
    max = np.max(X_train[i])
    X_train[i] /= max
    ProgressBar.printProgressBar(
        i+1, X_train.shape[0], prefix='Normalizing Data', length=50)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
y_train = np.reshape(y_train, (y_train.shape[0], 1))


# create network
net = Network(Loss.MSE, Loss.MSE_der)
net.add(FCLayer(n_inputs, 15))
net.add(ActivationLayer(Activations.tanh, Activations.tanh_der))
net.add(FCLayer(15, 15))
net.add(ActivationLayer(Activations.tanh, Activations.tanh_der))
net.add(FCLayer(15, 1))
net.add(ActivationLayer(Activations.tanh, Activations.tanh_der))
net.fit(X_train, y_train, 0.1, 50)

print(np.reshape(net.forward(X_train[-5:]), (5)))
print(y_train[-5:])

# run
prevData = np.zeros((n_inputs))
while True:
    data = np.array(fetch_data(unix_time_millis(dt.utcnow()-timedelta(weeks=1)),
                               unix_time_millis(dt.utcnow()), ticker)[-n_inputs:])
    if(prevData[0] != data[0]):
        min = np.min(data)
        X = data - min
        max = np.max(X)
        X /= max
        prediction = net.forward(np.array([[X]]))[0][0][0]
        print(f"Prediction: {prediction}")
        if(prediction >= 0.0):
            print('Trying to buy')
            if(money > risk):
                print('Bought')
                money -= risk
                amount += risk/data[-1]
            else:
                print('Not enough funds')
        else:
            print('Trying to sell')
            if(amount > risk/data[-1]):
                print('Sold')
                money += risk
                amount -= risk/data[-1]
            else:
                print('Not enough shares')
        prevData = np.copy(data)
        gain = money - starting_total + amount*data[-1]
        in_shares = amount*data[-1]
        print()
        print(f'Gain: {gain:.2f}')
        print(f'$: {money:.2f}')
        print(f'$ in shares: {in_shares:.9f}')
    else:
        print('No differences found')
