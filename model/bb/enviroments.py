#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:07:52 2017

@author: reg

These define the enviroments used to train the reinforcement learning algorithms.
The base class is defined here https://github.com/openai/gym/blob/master/gym/core.py
"""

import numpy as np
from bb.features import rsi, cci_short, stoc_osc
from gym import Env
from gym import spaces
from gym.utils import seeding


class BuySellPerfect(Env):

    """
    Simple enviroment that allows a buy order, sell order, or do nothing.
    Order quantity is fixed at 1.
    Sell price and buy price are determined by Ask/Bid

    This enviroment has 'perfect' foresight. It is used for testing purposes.
    """
        
    def __init__(self, BitIntObj):
        #
        self.BitInt = BitIntObj
        # history = self.BitInt.history
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5,))
        
        self.max_buy = 1
        self.max_sell = self.max_buy
        
        self.curr_val = (0, 0)
        self.new_val  = (0, 0)
        
        self.inital_object = self
        
        self.buy_count = 0
        self.sell_count = 0
        
        self.value_history = []
        self.value_before = 0
        self.past_reward = [0]
        
    def _step(self, a):        

        done = bool(False)

        value_before = self.BitInt.get_BTC() + self.BitInt.get_coin() * self.BitInt.get_last()

        self.do_action(a)

        # Go to next timestep, calculate reward        
        self.BitInt.inc()
        Btc_after = self.BitInt.get_BTC()
        coin_after = self.BitInt.get_coin() 
        value_after = Btc_after + coin_after*self.BitInt.get_last()        
        value_gained = value_after - value_before      
        reward = value_gained*10000
        
        observation = self.get_observation()
            
        return observation, reward, done, {}

    def _reset(self):
        
        Btc_before = self.BitInt.get_BTC()
        coin_before = self.BitInt.get_coin() 
        value_before = Btc_before + coin_before*self.BitInt.get_last()
        print "BTC: ", value_before
        
        print "Buy Orders: ", self.buy_count
        print "Sell Orders: ", self.sell_count
        
        
        self.value_history.append(value_before)
        
        self.BitInt.reset()
        self.new_val  = (0, 0)
        self.buy_count = 0
        self.sell_count = 0
        self.curr_val = (np.mean(self.BitInt.get_history()), 0)
        
        # Build observation
        
        observation = self.get_observation()

        return observation

    def _render(self, mode='human', close=False):
        return
    
    def _seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_value(self):
        return self.BitInt.get_BTC() + self.BitInt.get_coin()*self.BitInt.get_last()
    
    def get_observation(self):
        
        price_list = self.BitInt.get_history()
        
        mean = np.mean(price_list)
        std = np.std(price_list)
        
        tmp_index = self.BitInt.index
        future_mean = (np.mean(self.BitInt.price_list_total[tmp_index:tmp_index+1000])-mean)/std
        current_price = (np.mean(self.BitInt.price_list_total[tmp_index-100:tmp_index])-mean)/std
        current_value = ((self.curr_val[0]-mean)/std)
        coin_quant_feat = np.log(self.BitInt.get_coin()+1)  
        price_diff = future_mean-current_price
        
        observation = np.array([future_mean, current_price, current_value, coin_quant_feat, price_diff])
        
        return observation

    def do_action(self, a):

        if self.BitInt.get_index() == self.BitInt.get_max_index() - 1:
            done = bool(True)

        # a == 0 is buy
        # a == 1 is sell
        # a == 2 do nothing

        # Execute actions ####################################################
        self.BitInt.clear_orders()

        # Retrive data for current, and next timestamp
        Btc_before = self.BitInt.get_BTC()
        coin_before = self.BitInt.get_coin()

        max_buy = self.BitInt.get_BTC() / (self.BitInt.get_Ask() * 1.05)

        quant_buy = max_buy
        quant_sell = coin_before

        if max_buy < quant_buy:
            quant_buy = max_buy

        # buy order
        if a == 0:
            self.buy_count += 1
            price = self.BitInt.get_Ask() * 1.05
            self.BitInt.buy('', quant_buy, price)
            Btc_after = self.BitInt.get_BTC()
            coin_after = self.BitInt.get_coin()
            if coin_after > coin_before:
                coin_bought = coin_after - coin_before
                BTC_spent = -(Btc_after - Btc_before)
                price = BTC_spent / coin_bought  # Effective price of coin purchased
                self.curr_val = (self.curr_val[0] * self.curr_val[1] + price * coin_bought) / coin_after, coin_after

        # sell order
        if (a == 1) & (coin_before >= quant_sell):
            self.sell_count += 1
            price = self.BitInt.get_Bid()
            self.BitInt.sell_by_quant(quant_sell)
            self.curr_val = self.curr_val[0], self.BitInt.get_coin()


class SimpleFeature(BuySellPerfect):
    """
    This class replaced the 'perfect' feature in BuySellPerfect with an engineered features

    NEEDS UPDATING!
    """

    def __init__(self, BitIntObj):

        super(SimpleFeature, self).__init__(BitIntObj)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(14,))

        self.run_history = []
        self.run_current = []

    def get_observation(self):
        """
        build observation from enviromental data
        """

        price_list = self.BitInt.get_history()

        current_price = np.mean(price_list[-10:])

        # Build observation
        observation = []

        # Price features ###########

        # medium time scale features

        # Coin quant
        observation.append(2 * (int(self.BitInt.get_coin() > 0) - 0.5))

        # Diff features
        price_list_norm = np.array(price_list) / current_price
        mean_diff1 = np.mean(np.diff(price_list_norm))
        mean_diff2 = np.mean(np.diff(np.diff(price_list_norm)))

        observation.append(np.tanh(mean_diff1 * 58233.31761 * 2))
        observation.append(np.tanh(mean_diff2 * 409026.7290 * 2))

        observation.append(np.tanh((rsi(price_list) - 0.5) * 77.8216073))
        observation.append(np.tanh((cci_short(price_list) + 0.6667) * 237 / 0.791584))
        observation.append(np.tanh((stoc_osc(price_list) - 0.5) * 3.2 / 0.944009 * 2))  # 7

        # short time scale features
        mean_diff1_short = np.mean(np.diff(price_list_norm[-200:]))
        mean_diff2_short = np.mean(np.diff(np.diff(price_list_norm[-200:])))
        observation.append(np.tanh(mean_diff1_short * 10000 * 5))
        observation.append(np.tanh(mean_diff2_short * 100000 * 5 / 6.086887))
        observation.append(np.tanh((rsi(price_list[-200:]) - 0.5) * 61))
        observation.append(np.tanh((cci_short(price_list[-200:]) + 0.6667) * 237 / 0.394910))
        observation.append(np.tanh((stoc_osc(price_list[-200:]) - 0.5) * 3.2 / 2.219762))

        # quant features ###########

        quant_list = self.BitInt.get_quant_hist()
        current_quant = np.mean(quant_list[-10:])
        quant_list_norm = quant_list / current_quant

        # medium time scale
        mean_diff1 = np.mean(np.diff(quant_list_norm))
        observation.append(np.tanh(mean_diff1 / 0.000109))

        # short time scale
        mean_diff1_short = np.mean(np.diff(quant_list_norm[-100:]))
        observation.append(np.tanh(mean_diff1_short / 0.000139))

        return np.array(observation)

    def _step(self, a):

        done = bool(False)

        value_before = self.BitInt.get_value()

        self.do_action(a)

        # Go to next timestep, calculate reward
        self.BitInt.inc()

        value_after = self.BitInt.get_value()
        value_gained = value_after / value_before - 1

        if value_gained > 0:
            value_gained = value_gained / 1.0156381008851967  # Rescale for class imbalance

        reward = value_gained * 1000
        self.past_reward.append(reward)
        observation = np.append(self.get_observation(), np.mean(self.past_reward))

        if a != 2:
            reward -= 0.01

        # Record values for troubleshooting
        run_data = {}
        run_data['index'] = self.BitInt.index - 1
        run_data['observation'] = observation
        run_data['BTC'] = self.BitInt.get_BTC()
        run_data['coin'] = self.BitInt.get_coin()
        run_data['value_before'] = self.value_before
        run_data['value_after'] = value_after
        run_data['value_gained'] = value_gained
        run_data['action'] = a

        self.run_current.append(run_data)

        return observation, reward, done, {}

    def _reset(self):

        Btc_after = self.BitInt.get_BTC()
        coin_after = self.BitInt.get_coin()
        value = Btc_after + coin_after * self.BitInt.get_last()
        if value < 0.98:
            self.run_history.append(self.run_current)

        observation = super(SimpleFeature, self)._reset()
        self.past_reward = [0]
        self.run_current = []

        return np.append(observation, np.mean(self.past_reward))