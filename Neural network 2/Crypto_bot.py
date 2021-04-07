import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import bitfinex
from Net import RNN
from ActivationsLosses import Activations
from ProgressBar import ProgressBar


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


class Agent_controller:
    def __init__(self, base_inputs, base_hidden, base_risk, mutation_amount):
        self.agents = []
        self.base_inputs = base_inputs
        self.base_hidden = base_hidden
        self.base_model = RNN(base_inputs, base_hidden, 3,
                              Activations.Sigmoid, Activations.Softmax)
        self.mutation_amount = mutation_amount

    def spawn(self, amount):
        for i in range(amount):
            self.agents.append(Agent(self.base_model))
        for agent in self.agents:
            agent.Mutate(self.mutation_amount)

    def step(self, data):
        for agent in self.agents:
            agent.step(data)

    def Generation(self, data):
        max_inputs = 0
        for agent in self.agents:
            if agent.n_inputs > max_inputs:
                max_inputs = agent.n_inputs

        for i in range(max_inputs, data.shape[0]-1):
            self.step(data[i-max_inputs:i])


class Agent:
    def __init__(self, model, funds, risk):
        self.model = model
        self.n_inputs = model.n_inputs
        self.funds = funds
        self.crypto = 0.0
        self.risk = risk

    def Mutate(self, amount):
        for param in self.model.params.values():
            param += np.random.random(param.shape) * amount

    def step(self, data):
        data = data[-self.n_inputs:]
        X, _, _ = Normalize(data)
        X = X.reshape((self.n_inputs, 1, 1))
        prediction = self.model.forward(
            X, np.random.randn(self.model.n_hidden, 1)).reshape((3,))
        if np.argmax(prediction) is 0:
            self.funds -= self.risk
            self.crypto += data[-1]
        return prediction


controller = Agent_controller(60, 100, 5, 0.1)
controller.spawn(50)
controller.Generation(np.random.randn(1000))
