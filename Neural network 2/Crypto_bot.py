from datetime import datetime as dt, timedelta
import time
import copy
from tkinter.constants import BOTTOM, LEFT, RIGHT, TOP

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tkinter as tk
import yfinance as yf
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


def fetch_data(days, stop, symbol='btc-usd', interval='1m'):
    df = yf.download(tickers=symbol, interval=interval, stop=stop,
                     start=stop-timedelta(days=days), progress=True)
    df = df.reset_index(drop=True)['Close'].to_numpy()
    df = df[::-1]

    return df


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

    def save(self):
        pickle.dump(self.get_best_agents(1), open('Saved_model.pickle', 'wb'))
    
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
        exchange = fetch_data(2,
                              dt.now(), interval='1m')[-1]

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

        print(
            f'Best Score: {self.get_best_agents(1)[0].get_score(exchange)}')
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
        return self.get_best_agents(1)[0].get_score(exchange)

    def Run(self, generations):
        try:
            generations = int(generations)
        except ValueError:
            print('Not int. Try again.')
            return

        if generations != 0:
            scores = []
            for generation in range(generations):
                data = fetch_data(
                    7, dt.now(), interval='1m')
                scores.append(self.Generation(data))
                print(f'Generation: {generation}')
            plt.plot(scores)
            plt.show()
            
        else:
            generation = 0
            while True:
                try:
                    data = fetch_data(
                        7, dt.now(), interval='1m')
                    self.Generation(data)
                    generation += 1
                    print(f'Generation: {generation}')
                except Exception as error:
                    print(str(error))
                    pass

    def get_best_agents(self, num):
        exchange = fetch_data(1,
                              dt.now(), interval='1m')[-1]  # get current price
        sorted_agents = sorted(
            self.agents, key=lambda agent: agent.get_score(exchange), reverse=True)
        return sorted_agents[:num]


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

    def Test(self):
        data = fetch_data(7, dt.now())
        total = []
        for i in range(len(data)-self.model.n_inputs):
            self.step(data[i:i+self.model.n_inputs])
            total.append(self.get_score(data[-1]))
        plt.plot(total)
        plt.show()


def Load_a():
    global agent
    agent = pickle.load(open('Saved_model.pickle', 'rb'))[0]


controller = Agent_controller(60, 100, 5.0, 0.1, 1000.0, 3000)
#controller.load(pickle.load(open('Saved_model.pickle', 'rb'))[0])
# controller.spawn(50)
# controller.Run(20)
agent = pickle.load(open('Saved_model.pickle', 'rb'))[0]
window = tk.Tk()
window.geometry('300x200')
frame = tk.Frame()
frame.pack()
load_c = tk.Button(master=frame, text='Load to controller', command=lambda: controller.load(
    pickle.load(open('Saved_model.pickle', 'rb'))[0]))
load_c.pack()

load_a = tk.Button(master=frame, text='Load to agent', command=Load_a)
load_a.pack()

num_f = tk.Frame(frame)
num_f.pack()

spawn_l = tk.Label(master=num_f, text='Number of agents')
spawn_l.pack( side = LEFT)

n_agents = tk.Entry(master=num_f)
n_agents.pack( side = LEFT)

spawn_f = tk.Frame(frame)
spawn_f.pack()

from_base = tk.BooleanVar()
from_base_check = tk.Checkbutton(master=spawn_f, variable=from_base, onvalue=True, offvalue=False, text='From base')
from_base_check.pack( side = LEFT)

spawn = tk.Button(master=spawn_f, text='Spawn',
                command = lambda: controller.spawn(n_agents.get(), from_base.get()))
spawn.pack( side = TOP)


run_f = tk.Frame(frame)
run_f.pack()

generations_l = tk.Label(master=run_f, text='Generations')
generations_l.pack()

generations = tk.Entry(master=run_f)
generations.pack()

run = tk.Button(master=run_f, text='Run',
                command = lambda: controller.Run(generations.get()))
run.pack()

save = tk.Button(master=frame, text='Save',
                command = controller.save)
save.pack()

test = tk.Button(master=frame, text='Test', command = lambda: controller.get_best_agents(1)[0].Test())
test.pack()

window.mainloop()
