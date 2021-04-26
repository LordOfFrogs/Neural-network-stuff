from datetime import datetime as dt, timedelta, time
import copy
import os
import sys

import matplotlib.pyplot as plt
from numpy.random import f
import pandas as pd
import numpy as np
from Historic_Crypto import HistoricalData
from Historic_Crypto import LiveCryptoData
import pickle
from Net import RNN
from ActivationsLosses import Activations
from ProgressBar import ProgressBar
import multiprocess as multiprocessing

n_RNN_PARAMS = 13


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def dt_to_string(dt):
    return dt.strftime('%Y-%m-%d-%H-%M')


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


def fetch_data(period, stop, symbol='LTC-USD', interval=60):
    with HiddenPrints():
        data = HistoricalData(symbol, 60, dt_to_string(
            stop-period), dt_to_string(stop), False).retrieve_data()
    data = data['close'].to_numpy()

    return data


def get_current(symbol='LTC-USD'):
    with HiddenPrints():
        data = HistoricalData(symbol, 60, dt_to_string(
            dt.utcnow()-timedelta(minutes=1)), verbose=False).retrieve_data()['close'][0]
    return data


class Agent_controller:
    def __init__(self, base_inputs, base_hidden, base_risk, mutation_amount, funds, training_len):
        self.agents = []
        self.base_inputs = base_inputs  # number of inputs at start
        self.base_hidden = base_hidden  # starting hidden state
        self.base_model = RNN(base_inputs, base_hidden, 2,
                              Activations.Sigmoid, Activations.Softmax)
        self.mutation_amount = mutation_amount
        self.base_risk = base_risk
        self.funds = funds
        self.base_a0 = np.random.randn(base_hidden, 1)
        self.training_len = training_len

    def load(self, agent):
        self.base_model = agent.model
        self.base_hidden = agent.model.n_hidden
        self.base_inputs = agent.model.n_inputs
        self.base_risk = agent.risk
        self.base_a0 = agent.a0

    def save(self, length):
        best_agent = self.Test(length)
        pickle.dump(best_agent, open('Saved_model.pickle', 'wb'))

    def spawn(self, amount, from_base=True):
        try:
            amount = int(amount)
        except ValueError:
            print('Not int. Try again.')
            return

        for i in range(amount):
            if from_base:
                self.agents.append(
                    Agent(self.base_model, self.funds, self.base_risk, self.base_a0))  # create new model
            else:
                self.agents.append(
                    Agent(RNN(self.base_inputs, self.base_hidden, 2,
                              Activations.Sigmoid, Activations.Softmax), self.funds, self.base_risk, np.random.randn(self.base_hidden, 1)))  # create new model
        for agent in range(1, len(self.agents)):
            self.agents[agent].Mutate(self.mutation_amount)  # mutate

    def step(self, data):
        for agent in self.agents:
            agent.step(data)

    def Generation(self, data):
        exchange = get_current()

        start = np.random.randint(0, data.shape[0]-self.training_len)
        for i in range(start, start+self.training_len):
            self.step(copy.copy(data[i:i+self.base_inputs]))
            ProgressBar.printProgressBar(
                i-start, self.training_len, prefix='Running generation', length=50)

        # find best agents
        best_agents = self.get_best_agents(len(self.agents))
        np.random.shuffle(best_agents)
        pairs = []
        for i in range(int(len(best_agents)/2)):
            pairs.append((best_agents[i], best_agents[i+1]))
        best_score = self.get_best_agents(1)[0].get_score(exchange)
        print(
            f'\nBest Score: {best_score}')
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
                params = pair[0].model.params.copy()
                for i in range(len(params.values())):
                    if choices[i]:
                        params[i] = pair[1].model.params.values()
                self.base_model.params = params
                self.spawn(1)
        return best_score

    def Train(self, generations):
        training_data = pd.read_csv('ltcusd.csv')['close'].to_numpy()

        if generations != 0:
            scores = []
            for generation in range(generations):
                scores.append(self.Generation(training_data))
                print(f'Generation: {generation+1}')
            plt.plot(scores)
            plt.show()

        else:
            generation = 0
            while True:
                try:
                    self.Generation(training_data)
                    generation += 1
                    print(f'Generation: {generation}')
                except Exception as error:
                    print(str(error))
                    pass

    def get_best_agents(self, num):
        exchange = get_current()  # get current price
        sorted_agents = sorted(
            self.agents, key=lambda agent: agent.get_score(exchange), reverse=True)
        return sorted_agents[:num]

    def Test(self, length, max_days_ago=600, num_agents=1):
        scores = []
        current = get_current()
        for _ in range(len(self.agents)):
            scores.append(0)
        for i in range(length):
            daysAgo = np.random.randint(0, max_days_ago)
            for agent in range(len(self.agents)):
                with HiddenPrints():
                    self.agents[agent].Test(
                        1, dt.utcnow()-timedelta(days=daysAgo), False)

                scores[agent] += self.agents[agent].get_score(current)
                ProgressBar.printProgressBar(
                    i*len(self.agents)+agent+1, length*len(self.agents), prefix='Testing', length=50)
        if num_agents == 1:
            best_agent = self.agents[scores.index(max(scores))]
            return best_agent
        else:
            best_agents = sorted(
                self.agents, key=lambda agent: scores[self.agents.index(agent)], reverse=True)
            return best_agents[:num_agents]


class Agent:
    def __init__(self, model, funds, risk, a0):
        self.model = copy.deepcopy(model)
        self.funds = funds
        self.crypto = 0.
        self.risk = risk
        self.a0 = copy.copy(a0)
        self.buys = 0
        self.sells = 0
        self.buy_sell_ratio = 1.

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
            self.buys += 1
            if self.funds > self.risk:
                self.funds -= self.risk
                self.crypto += self.risk/data[-1]
        else:  # sell
            self.sells += 1
            if self.crypto > self.risk/data[-1]:
                self.funds += self.risk
                self.crypto -= self.risk/data[-1]
        if self.sells != 0:
            self.buy_sell_ratio = self.buys/self.sells
        else:
            self.buy_sell_ratio = 0.

    def get_score(self, exchange):
        return self.funds + self.crypto*exchange

    def Test(self, days, stop, progress=True):
        data = fetch_data(timedelta(days=days), stop)
        current = get_current()
        total_current = []
        total_past = []
        for i in range(len(data)-self.model.n_inputs):
            self.step(data[i:i+self.model.n_inputs])
            # total_current.append(self.get_score(current))
            # total_past.append(self.get_score(data[i+self.model.n_inputs]))
            if progress:
                ProgressBar.printProgressBar(
                    i+1, len(data)-self.model.n_inputs, prefix=f'Testing: {i+1}/{len(data)-self.model.n_inputs}', length=50)
        print(f'Final score: {self.get_score(current)}')
        # plt.plot(total_current)
        # plt.plot(total_past)
        # plt.show()

    def Run(self, timesteps):
        prev = 0.0
        if timesteps == 0:
            timestep = 0
            while True:
                price = get_current()
                if price != prev:
                    timestep += 1
                    data = fetch_data(timedelta(hours=1),
                                      dt.utcnow())[-self.model.n_inputs:]
                    self.step(data)
                    prev = price
                    print(f'Total: ${self.get_score(price)}')
                    print(f'Funds: ${self.funds}')
                    print(f'Crypto: {self.crypto}')
                    print(f'Timesteps: {timestep}')
                # else:
                #    print(f'No new data: {price} {prev}')
        else:
            timestep = 0
            prev = 0.0
            while timestep <= timesteps:
                price = get_current()
                if price != prev:
                    data = fetch_data(timedelta(hours=1),
                                      dt.utcnow())[-self.model.n_inputs:]
                    self.step(data)
                    prev = price
                    timestep += 1
                    print(f'Total: ${self.get_score(price)}')
                    print(f'Funds: ${self.funds}')
                    print(f'Crypto: {self.crypto}')
                    print(f'Timestep: {timestep}/{timesteps}')
                # else:
                #    print(f'No new data: {price} {prev}')


if __name__ == '__main__':
    controller = Agent_controller(60, 100, 5.0, 0.1, 1000.0, 1440)
    agent = pickle.load(open('Saved_model.pickle', 'rb'))

    train_th = multiprocessing.Process(target=controller.Train)
    run_th = multiprocessing.Process(target=agent.Run)

    def Get_input():
        command = input('Enter Command: ')
        command = command.split()
        global controller
        global agent
        global train_th
        global run_th

        if command[0] == 'load':
            if command[1] == 'controller':
                controller.load(pickle.load(
                    open('Saved_model.pickle', 'rb')))
            elif command[1] == 'agent':
                agent = pickle.load(open('Saved_model.pickle', 'rb'))
            return
        if command[0] == 'save':
            try:
                len = int(command[1])
                controller.save(len)
                return
            except:
                pass
        if command[0] == 'spawn':
            try:
                num = int(command[1])
                from_base = True
                if command[2] == 't':
                    from_base = True
                elif command[2] == 'f':
                    from_base = False
                else:
                    raise Exception('Boolean value was not entered correctly')
                controller.spawn(num, from_base)
                return
            except:
                pass
        if command[0] == 'train':
            try:
                generations = int(command[1])
                train_th = multiprocessing.Process(
                    target=controller.Train, args=(generations,))
                train_th.start()
                return
            except:
                pass
        if command[0] == 'run':
            try:
                timesteps = int(command[1])
                run_th = multiprocessing.Process(
                    target=agent.Run, args=(timesteps,))
                run_th.start()
                return
            except:
                pass

        if command[0] == 'test':
            try:
                days = int(command[1])
                agent.Test(days, dt.utcnow())
                return
            except:
                pass

        if command[0] == 'reset':
            if command[1] == 'agent':
                agent.funds = controller.funds
                agent.crypto = 0
                agent.buys = 0
                agent.sells = 0
                agent.buy_sell_ratio = 1.
                return
            elif command[1] == 'controller':
                controller.agents.clear()
                return

        if command[0] == 'stop':
            if command[1] == 'training':
                train_th.terminate()
                return
            elif command[1] == 'running':
                run_th.terminate()
                return

        if command[0] == 'exit':
            exit()

        print(f'Error when processing input: {command}')

    while True:
        Get_input()
