# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:52:26 2021

@author: Subham
"""

# Ref: https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

import gym
from gym import spaces

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DDPG, A2C

import torch as th

import random
import datetime as dt
import pandas as pd
import numpy as np


MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 1000000

balance = INITIAL_ACCOUNT_BALANCE
net_worth = INITIAL_ACCOUNT_BALANCE
max_net_worth = INITIAL_ACCOUNT_BALANCE

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    
    metadata = {'render.modes': ['human']}

    
    
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        5, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            balance / MAX_ACCOUNT_BALANCE,
            max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs

    def _take_action(self, action):
        global balance, net_worth, max_net_worth
        
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        net_worth = balance + self.shares_held * current_price

        if net_worth > max_net_worth:
            max_net_worth = net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        global balance, net_worth, max_net_worth
        print(balance)
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = balance * delay_modifier
        done = net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        global balance, net_worth, max_net_worth
        
        # Reset the state of the environment to an initial state
        balance = INITIAL_ACCOUNT_BALANCE
        net_worth = INITIAL_ACCOUNT_BALANCE
        max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)

        return self._next_observation()

    def render(self, mode='human', close=False):
        global balance, net_worth, max_net_worth
        
        # Render the environment to the screen
        profit = net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {net_worth} (Max net worth: {max_net_worth})')
        print(f'Profit: {profit}')
        

ticker_names = df['tic'].unique()

# Custom actor (pi) and value function (vf) networks
# of two layers of size 32 each with Relu activation function
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[32, 32], vf=[32, 32])])

env = []
obs = []
model = []

# df - contains all ticker informations (OHLCV)

for ticker in ticker_names:
  df_i = df[(df['tic'] == ticker)]
  df_i = df_i.reset_index()
  df_i = df_i.drop(['tic', 'day', 'index'], axis=1)
  print(df_i.shape)
  env_i = DummyVecEnv([lambda: StockTradingEnv(df_i)])
  env.append(env_i)
  obs_i = env_i.reset()
  obs.append(obs_i)
  model_i = A2C("MlpPolicy", env_i, policy_kwargs=policy_kwargs, verbose=1)
  model.append(model_i)

# improvement - parallel programming
for counter in range(20000):
  print("\nLoop-{}-".format(counter))
  for i in range(len(ticker_names)):
    model[i].learn(total_timesteps=1)

# trading
for i in range(len(ticker_names)):
  obs[i] = env[i].reset()

balance = INITIAL_ACCOUNT_BALANCE
net_worth = INITIAL_ACCOUNT_BALANCE
max_net_worth = INITIAL_ACCOUNT_BALANCE
for _ in range(1000):
  for i in range(len(ticker_names)):
    action, _states = model[i].predict(obs[i])
    obs_i, rewards_i, done_i, info_i = env[i].step(action)
    env[i].render()
