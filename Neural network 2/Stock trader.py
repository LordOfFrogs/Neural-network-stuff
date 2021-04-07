##### OLD CODE #####

from Net import CNN, ActivationLayer, FCLayer
from ActivationsLosses import Activations, Loss
import numpy as np
import bitfinex
from datetime import datetime as dt, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from ProgressBar import ProgressBar
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers


def unix_time_millis(dt_obj):
    return time.mktime(dt_obj.timetuple()) * 1000

def flatten_list_2d(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        for item in element:
            flat_list.append(item)
    return flat_list


def Classify_gain(current, past):
    if current > past:
        return 1.0
    else:
        return 0.0

def Normalize(x):
    min = np.min(x)
    max = np.max(x)
    return (x-min)/(max-min), max, min

def Normalize_pre(x, max, min):
    return (x-min)/(max-min)

def Denormalize(x, max, min):
    return x*(max - min)+min

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
            starting-start+step, stop-start, prefix='Downloading data', length=50)
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

    return df


startingMoney = 5000.00
money = startingMoney
starting_amount = 0.0
amount = starting_amount
ticker = 'btcusd'
risk = 5.00
n_inputs = 60
raw_data = fetch_data(start=unix_time_millis(dt.utcnow() - timedelta(days=70)),
                      stop=unix_time_millis(dt.utcnow()), symbol=ticker)
raw_data = raw_data.reset_index(drop=True)['close']
print(raw_data)
starting_total = startingMoney + amount*raw_data.to_list()[-1]
raw_data = np.array(raw_data)
print('Download Complete')

X_unscaled = []
y_unscaled = []

for i in range(n_inputs, len(raw_data)):
    X_unscaled.append(raw_data[i-n_inputs:i])
    y_unscaled.append(raw_data[i])
    ProgressBar.printProgressBar(i-n_inputs+1, len(raw_data)-n_inputs,
                                 prefix='Preparing Data: {}/{}'.format(i-n_inputs+1, len(raw_data)-n_inputs), length=25)

X_unscaled = np.array(X_unscaled)
y_unscaled = np.array(y_unscaled)
y_unscaled = y_unscaled.reshape(-1, 1)
print(X_unscaled.shape)
print(y_unscaled.shape)

print('Normalizing Data...')
X = X_unscaled.copy()
y = y_unscaled.copy()
for i in range(X_unscaled.shape[0]):
    X[i], max, min = Normalize(X_unscaled[i])
    y[i] = Normalize_pre(y_unscaled[i], max, min)
    ProgressBar.printProgressBar(i+1, X_unscaled.shape[0], prefix='Normalizing', length=50)

X = X.reshape((len(X), n_inputs, 1))

X_train = X[:90000, :, :]
X_test = X[90000:, :, :]
print(X_test.shape)
y_train = y[:90000, :]
y_test = y[90000:, :]

model = tf.keras.Sequential()
model.add(layers.LSTM(units=32, return_sequences=True,
                      input_shape=(n_inputs, 1), dropout=0.2))
model.add(layers.LSTM(units=32, return_sequences=True, dropout=0.2))
model.add(layers.LSTM(units=32, dropout=0.2))
model.add(layers.Dense(units=1))
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=5, batch_size=32)

loss = history.history['loss']
epoch_count = range(1, len(loss)+1)
plt.figure(figsize=(12, 8))
plt.plot(epoch_count, loss, 'r--')
plt.legend(['Training loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

pred = model.predict(X_test)

plt.figure(figsize=(12, 8))
plt.plot(y_test, color='blue', label='Real')
plt.plot(pred, color='red', label='Prediction')
plt.title('Bitcoin prediction')
plt.legend()
plt.show()

# run
prevData = np.zeros((n_inputs))
while True:
    data = fetch_data(unix_time_millis(dt.utcnow()-timedelta(days=2)),
                      unix_time_millis(dt.utcnow()), ticker)
    data = data.reset_index(drop=True)['close']
    data = data.tail(n_inputs)
    data = data.to_numpy()
    if(prevData[0] != data[0]):
        X, min, max = Normalize(data)
        X = X.reshape((1, X.size, 1))
        prediction = model.predict(X)[0][0]
        print(f"Prediction: {'increase' if prediction >= X[0][-1] else 'decrease'}")
        if(prediction >= X[0][-1]):
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
        print(f'Gain: {gain:.2f}')
        print(f'$: {money:.2f}')
        print(f'$ in shares: {in_shares:.9f}')
    else:
        print('No differences found')
