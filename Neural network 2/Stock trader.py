from time import time
from Net import Network, ActivationLayer, FCLayer
from ActivationsLosses import Activations, Loss
import numpy as np

from datetime import datetime as dt, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from ProgressBar import ProgressBar
from pytz import timezone

startingMoney = 5000.00
money = startingMoney
stocks = 0
ticker = 'BTC-USD'
risk = 5
stocks_time = []

tz = timezone('EST')
MINUTES_IN_DAY = 1415
# get data
historical_data = []
mins = []
maxes = []
days = 29

data = pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv")
data.head()


for i in range(days):
    data = yf.download(ticker, start=dt.now(tz=tz) - timedelta(days=i+1), end=dt.now(tz=tz) - timedelta(days=i),
                       interval='1m', progress=False).Close.to_numpy()
    historical_data.append(data)
    ProgressBar.printProgressBar(
        i+1, days, 'Downloading Stock Data: ', length=50)
print('Download Complete')
# prepare data
for i in range(len(historical_data)):
    mins.append(np.min(historical_data[i]))
    historical_data[i] -= mins[i]
    maxes.append(np.max(historical_data[i]))
    historical_data[i] /= maxes[i]

X_train = np.ndarray((len(historical_data), MINUTES_IN_DAY-1))
y_train = np.ndarray((len(historical_data)))
x = 0
for i in historical_data:
    X_train[x] = np.array(i[1:])
    y_train[x] = np.array(i[0])
    x += 1
X_train = X_train.reshape((len(historical_data), 1, MINUTES_IN_DAY-1))
y_train = y_train.reshape((len(historical_data), 1))


# create network
net = Network(Loss.MSE, Loss.MSE_der)
net.add(FCLayer(MINUTES_IN_DAY-1, 15))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))
net.add(FCLayer(15, 15))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))
net.add(FCLayer(15, 1))
net.add(ActivationLayer(Activations.Sigmoid, Activations.Sigmoid_der))
net.fit(X_train, y_train, 0.1, 30)

# region test
'''# test
test_data = yf.download(ticker, end=dt.today(), period='1wk',
                        interval='1m', progress=False).Close.to_numpy()
test_data = np.flip(test_data)
min = np.min(test_data)
test_data_norm = test_data - min
max = np.max(test_data_norm)
test_data_norm /= max
for i in range(test_data_norm.shape[0] - 1 - MINUTES_IN_DAY):
    X = test_data_norm[i:i+MINUTES_IN_DAY]
    prediction = net.forward([[X]])
    prediction = (prediction[0][0][0] * max) + min
    if(prediction - test_data[i] > 0.0 and money > test_data[i]):
        money -= test_data[i]*risk
        stocks += risk
    elif(prediction - test_data[i] < 0.0 and stocks > 0):
        money += test_data[i]*risk
        stocks -= risk
    stocks_time.append(stocks)

gain = money - startingMoney + stocks*test_data[-1]
raw_gain = money - startingMoney
print('Final Stock Price: %f' % (test_data[-1]))
print(f'Money after week: {money}')
print(f'Money gained after week: {gain}')
print(f'Raw Gain: {raw_gain}')
print(f'Stonks after week: {stocks}')
plt.plot(stocks_time)
plt.show()'''
# endregion

# run
prevData = np.zeros((MINUTES_IN_DAY-1))
while True:
    data = yf.download(ticker, period='1d',
                       end=dt.now(tz=tz), interval='1m', progress=False).Close
    data = data.to_numpy()[:-1]
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
