from datetime import datetime as dt, timedelta
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import bitfinex
import pickle
from Net import RNN
from ActivationsLosses import Activations
from ProgressBar import ProgressBar

n_RNN_PARAMS = 13


def Classify_gain(current, past):
    if current > past:
        return 1.0
    else:
        return 0.0


def unix_time_millis(dt_obj):
    # convert Datetime to millis since epoch
    return time.mktime(dt_obj.timetuple()) * 1000


def Normalize(x):
    min = np.min(x)
    max = np.max(x)
    return (x-min)/(max-min), max, min


def Normalize_pre(x, max, min):  # Normalize with already set min and max
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
            starting-start, stop-start, prefix='Downloading data', length=50)
        time.sleep(1)
    print('\n')
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
    df = df.reset_index(drop=True)['close']
    data = df.to_numpy()

    return data


class Agent_controller:
    def __init__(self, base_inputs, base_hidden, base_risk, mutation_amount, funds):
        self.agents = []
        self.base_inputs = base_inputs  # number of inputs at start
        self.base_hidden = base_hidden  # starting hidden state
        self.base_model = RNN(base_inputs, base_hidden, 2,
                              Activations.Sigmoid, Activations.Softmax)
        self.mutation_amount = mutation_amount
        self.base_risk = base_risk
        self.funds = funds

    def spawn(self, amount):
        for i in range(amount):
            self.agents.append(
                Agent(self.base_model, self.funds, self.base_risk, self.base_hidden))  # create new model
        for agent in self.agents:
            agent.Mutate(self.mutation_amount)  # mutate

    def step(self, data):
        for agent in self.agents:
            agent.step(data)

    def Generation(self, data):
        # find agent with most inputs
        max_inputs = 0
        for agent in self.agents:
            if agent.model.n_inputs > max_inputs:
                max_inputs = agent.model.n_inputs
        # test
        for i in range(max_inputs, data.shape[0]-1):
            self.step(data[i-max_inputs:i])

        # find best agents
        best_agents = self.get_best_agents(int(len(self.agents)/2))
        np.random.shuffle(best_agents)
        pairs = []
        for i in range(int(len(best_agents)/2)):
            pairs.append((best_agents[i], best_agents[i+1]))

        self.agents.clear()
        
        # create new generation
        for pair in pairs:
            for _ in range(2):  # to have same # of agents
                # to choose between agents
                choices = np.random.choice(1, n_RNN_PARAMS)
                self.base_risk = pair[choices[11]].risk
                self.a0 = pair[choices[12]].a0
                self.base_model = RNN(self.base_inputs, self.base_hidden, 2,
                                      Activations.Sigmoid, Activations.Softmax)
                params = pair[0].model.params
                for i in range(len(params.values())):
                    if choices[i]:
                        params[i] = pair[1].model.params.values()
                self.base_model.params = params
                self.spawn(1)

    def Run(self, generations):
        if generations != 0:
            for _ in range(generations):
                data = fetch_data(unix_time_millis(
                    dt.utcnow()-timedelta(weeks=1)), unix_time_millis(dt.utcnow()), interval='1m')
                self.Generation(data)
            return self.get_best_agents(1)
        else:
            try:
                while True:
                    data = fetch_data(unix_time_millis(
                        dt.utcnow()-timedelta(weeks=1)), unix_time_millis(dt.utcnow()), interval='1m')
                    self.Generation(data)
            except KeyboardInterrupt:
                pickle.dump(self.get_best_agents(
                    1), open('Saved_model.pickle', 'wb'))
                raise

    def get_best_agents(self, num):
        exchange = fetch_data(unix_time_millis(dt.utcnow()-timedelta(days=1)),
                              unix_time_millis(dt.utcnow()), interval='1m')[-1]  # get current price
        sorted_agents = sorted(
            self.agents, key=lambda agent: self.get_score(agent, exchange))
        return sorted_agents[:num]

    def get_score(_, agent, exchange):
        return agent.funds + agent.crypto*exchange


class Agent:
    def __init__(self, model, funds, risk, a0):
        self.model = model
        self.funds = funds
        self.crypto = 0.0
        self.risk = risk
        self.a0 = a0

    def Mutate(self, amount):
        for param in self.model.params.values():
            param += np.random.random(param.shape) * amount
        self.risk += np.random.randn(1)[0]*amount
        self.a0 += np.random.randn(self.model.n_hidden, 1)

    def step(self, data):
        data = data[-self.model.n_inputs:]
        X, _, _ = Normalize(data)
        X = X.reshape((self.model.n_inputs, 1, 1))
        prediction = self.model.forward(
            X, self.a0).reshape((2,))  # predict
        if prediction[0] > prediction[1]:  # buy
            if self.funds > self.risk:
                self.funds -= self.risk
                self.crypto += self.risk/data[-1]
        elif self.crypto > self.risk/data[-1]:  # sell
            self.funds += self.risk
            self.crypto -= self.risk/data[-1]


controller = Agent_controller(60, 100, 5, 0.1, 1000)
controller.spawn(50)
controller.Run(0)
