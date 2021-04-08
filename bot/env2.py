from collections import deque
from utils import TradingGraph, Write_to_file

import random
import numpy as np 

class CustomEnv:
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, Render_range=100, Show_reward=False, Show_indicators=False, normalize_value=40000):
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df)-1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.Render_range = Render_range
        self.Show_reward = Show_reward
        self.Show_indicators = Show_indicators

        self.orders_history = deque(maxlen=self.lookback_window_size)
        
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.indicators_history = deque(maxlen=self.lookback_window_size)

        self.normalize_value = normalize_value
    def reset(self, env_steps_size = 0):
        self.visualization = TradingGraph(Render_range=self.Render_range, Show_reward=self.Show_reward, Show_indicators=self.Show_indicators)
        self.trades = deque(maxlen=self.Render_range)
        self.open_order = False 
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0
        self.prev_episode_orders = 0 
        self.rewards = deque(maxlen=self.Render_range)
        self.env_steps_size = env_steps_size
        self.punish_value = 0
        self.stop = None
        self.target = None
        self.trades.append({'Date' : 0, 'High' : 0, 'Low' : 0, 'total': 0, 'type': "hold", 'current_price': 0, 'Stop' : 0, 'Target' : 0})
        if env_steps_size > 0: 
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
            
        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

            self.market_history.append([
                                        self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'Volume'],
                                        ])

            self.indicators_history.append([self.df.loc[current_step, 'Obv'] / self.normalize_value,
                                        self.df.loc[current_step, 'Rsi'] / self.normalize_value
                                        ])
            

        state = np.concatenate((self.market_history, self.orders_history), axis=1) / self.normalize_value
        state = np.concatenate((state, self.indicators_history), axis=1)

        return state

    def _next_observation(self):
        self.market_history.append([
                                    self.df.loc[self.current_step, 'Close'],
                                    self.df.loc[self.current_step, 'Volume'],
                                    ])

        self.indicators_history.append([self.df.loc[self.current_step, 'Obv'] / self.normalize_value,
                                    self.df.loc[self.current_step, 'Rsi'] / self.normalize_value
                                    ])
        
        obs = np.concatenate((self.market_history, self.orders_history), axis=1) / self.normalize_value
        obs = np.concatenate((obs, self.indicators_history), axis=1)
        
        return obs

    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1
        self.leverage = 30
        current_price = random.uniform(
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'Close'])
        Date = self.df.loc[self.current_step, 'Date']
        High = self.df.loc[self.current_step, 'High']
        Low = self.df.loc[self.current_step, 'Low']
        
        try:
            Low_next = self.df.loc[self.current_step+1, 'Low']
            High_next = self.df.loc[self.current_step+1, 'High']
        except:
            Low_next = self.df.loc[self.current_step, 'Low']
            High_next = self.df.loc[self.current_step, 'High']
        if self.crypto_held == 0 or self.crypto_held == None:
            self.open_order = False
            self.crypto_held = 0
        
        if self.stop == None or self.target == None:
            self.stop = 0
            self.target = 0

        if action == 0:
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_held, 'type': "hold", 'current_price': current_price, 'Stop' : self.stop, 'Target' : self.target})
            pass

        elif action == 1 and self.open_order == False:
            self.crypto_bought = (self.balance * self.leverage) * 0.8 / current_price
            self.balance -= (self.crypto_bought * current_price) 
            self.crypto_held += self.crypto_bought
            self.bought_price = current_price
            self.stop = current_price * 0.98
            self.target = current_price * 1.04
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_bought, 'type': "buy", 'current_price': current_price, 'Stop' : self.stop, 'Target' : self.target, 'Bought_price' : self.bought_price})
            self.episode_orders += 1
            self.open_order = True

        elif action == 2 and self.open_order == False:
            self.crypto_sold = (self.balance * self.leverage) * 0.8 / current_price
            self.balance -= (self.crypto_sold * current_price) 
            self.crypto_held += self.crypto_sold
            self.sold_price = current_price
            self.stop = current_price * 1.02
            self.target = current_price * 0.96
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_sold, 'type': "sell", 'current_price': current_price, 'Stop' : self.stop, 'Target' : self.target, 'Sold_price' : self.sold_price})
            self.episode_orders += 1
            self.open_order = True

        if self.episode_orders >= 2:
            if self.open_order == True:
                if self.trades.tail('type') == "buy":
                    self.net_worth = self.balance + self.crypto_held * (self.trades.tail('Bought_price') - current_price)
                elif self.trades.tail('type') == "sell":
                    self.net_worth = self.balance + self.crypto_held * (current_price - self.trades.tail('Sold_price'))
        else:
            self.net_worth = self.balance + self.crypto_held * current_price       
        
        self.prev_net_worth = self.net_worth
        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
        reward = self.get_reward()

        if self.net_worth <= -self.initial_balance:
            done = True
            reward = reward * 2
        else:
            done = False

        obs = self._next_observation()
        
        return obs, reward, done

    def get_reward(self):
        self.punish_value += self.net_worth * 0.001
        if self.episode_orders >= 2:
            self.prev_episode_orders = self.episode_orders
            if self.trades.tail('type') == "buy":
                reward += 0.01
                if Low_next <= self.trades.tail('Stop') or High_next >= self.trades.tail('Target'): 
                    self.balance += self.crypto_held * (self.trades.tail('Bought_price') - current_price)
                    self.net_worth = self.balance
                    reward = self.balance - self.initial_balance
                    self.open_order = False
                    self.crypto_held = 0
                    self.punish_value = 0
                    return reward
                else:
                    return reward

            elif self.trades.tail('type') == "sell":
                reward += 0.01 
                if Low_next >= self.trades.tail('Stop') or High_next <= self.trades.tail('Target'): 
                    self.balance += self.crypto_held * (current_price - self.trades.tail('Sold_price'))
                    self.net_worth = self.balance
                    reward = self.balance - self.initial_balance
                    self.open_order = False
                    self.crypto_held = 0
                    self.punish_value = 0
                    return reward
                else:
                    return reward
            else:
                return 0 - self.punish_value
        else:
            return 0 - self.punish_value

    def render(self, visualize = False):
        if visualize:
            img = self.visualization.render(self.df.loc[self.current_step], self.net_worth, self.trades)
            return img


