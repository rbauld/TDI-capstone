#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:18:34 2017

@author: reg

This module defines objects used to interface with the bittrex servers.

The goal of these objects is to act as a abstraction layer between my software and the bittrex servers.

_train objects are supposed to emulate interaction with the bittrex server. They are also used for model training.
"""


import numpy as np
from pymongo import cursor


class BitInterface_train(object):

    # A slightly faster version of BitInterface_train

    def __init__(self, mongo_in, init_BTC, index_start=0, index_end=40000, step=100, history=100, fee=0.0025,
                 book_length=1200, end_buffer=1000):

        # Internal index. Used for syncing up with MongoDB

        self.history = history
        self.index_start = index_start
        self.index_end = index_end
        self.index_reset = index_start + history
        self.init_BTC = init_BTC
        self.index = history  # this index considers index_start = 0
        self.step = step

        # Mongo collection that has gathered data
        self.mongoCol = mongo_in
        self.BTC = init_BTC
        self.coin_quant = 0  # Current ammount of alt-coin purchased
        self.fee = fee
        self.book_length = book_length
        self.end_buffer = end_buffer

        self.market = self.mongoCol.find_one({"index": 0})['marketsum']['result'][0]['MarketName']

        # Build orderbook sell/buy
        self.price_list_total = []
        self.quant_list_total = []
        self.order_buy = np.zeros((self.book_length, 2, index_end - index_start))
        self.order_sell = np.zeros((self.book_length, 2, index_end - index_start))

        step = 500
        n_steps = (self.index_end-self.index_start)/step

        # Loada data in batches
        for ii in range(n_steps):
            data = self.mongoCol.find({'index': {'$gte': index_start+ii*step, '$lt': index_start+step+ii*step}},
                                      batch_size=10, cursor_type=cursor.CursorType.EXHAUST)
            print ii*step

            for test in list(data):
                ii = test['index']
                # history lists
                self.price_list_total.append(test['marketsum']['result'][0]['Last'])
                self.quant_list_total.append(test['marketsum']['result'][0]['BaseVolume'])

                # orderbooks
                sell1 = np.array(map(lambda x: (x['Rate'], x['Quantity']), test['orderbook']['result']['sell']))
                buy1 = np.array(map(lambda x: (x['Rate'], x['Quantity']), test['orderbook']['result']['buy']))
                self.order_buy[0:len(buy1), :, ii - self.index_start] = buy1[0:self.book_length, :]
                self.order_sell[0:len(sell1), :, ii - self.index_start] = sell1[0:self.book_length, :]

    def buy(self, market, quantity, rate):
        # Simulate immediate buy order
        quantity_to_buy = abs(quantity)
        quantity_bought = 0
        price = rate
        fee = self.fee  # Bittrex transaction fee
        price_of_buy = 0
        BTC = self.BTC

        sell1 = self.order_sell[:, :, self.index + 1]

        sell1 = sell1[sell1[:, 0] < price]

        for ii in range(len(sell1)):
            quant_buy = min([sell1[ii, 1], quantity_to_buy, BTC / (fee + 1) / sell1[ii, 0]])
            quantity_bought += quant_buy
            quantity_to_buy = quantity_to_buy - quant_buy
            BTC = BTC - quant_buy * sell1[ii, 0] * (fee + 1)
            price_of_buy += quant_buy * sell1[ii, 0] * (fee + 1)
            if quantity_to_buy <= 0:
                break

        # Update values from transaction
        self.coin_quant += quantity_bought
        self.BTC = BTC

    def sell(self, market, quantity, rate):
        # Simulate immediate sell order

        if quantity > self.coin_quant:
            raise Exception('Attempting to sell more coin then you have')

        quantity_to_sell = abs(quantity)  # Ensure that no attempt to sell negative coins
        price = rate
        fee = self.fee  # Bittrex transaction fee

        buy1 = self.order_buy[:, :, self.index + 1]

        total_buyable = sum(buy1[buy1[:, 0] >= price][:, 1])

        sold = min(quantity_to_sell, total_buyable)

        self.coin_quant -= sold
        self.BTC += sold * price * (1 - fee)

    def sell_by_quant(self, quantity):
        price_index = np.argmax(np.cumsum(self.order_buy[:, 1, self.index]) > quantity)
        price = self.order_buy[price_index+1, 0, self.index]
        self.sell('', quantity, price)

    def inc(self):
        self.index += self.step

    def get_BTC(self):
        return self.BTC

    def get_coin(self):
        return self.coin_quant

    def get_last(self):
        return self.price_list_total[self.index]

    def get_history(self):
        return self.price_list_total[self.index + 1 - self.history:self.index + 1]

    def get_quant_hist(self):
        return self.quant_list_total[self.index + 1 - self.history:self.index + 1]

    def clear_orders(self):
        return

    def reset(self):
        # Internal index. Used for syncing up with MongoDB
        # Resets to a random place in the dataset. The end has a buffer so as not to run into the
        # end of the data set while training

        tot_length = self.index_end - self.index_start
        min_index = self.history
        max_index = tot_length-self.end_buffer

        # Set new index as some random time, inbetween buffer
        self.index = np.random.randint(min_index, max_index)

        # Mongo collection that has gathered data
        self.BTC = self.init_BTC
        self.coin_quant = 0  # Current ammount of alt-coin purchased

    def get_max_index(self):
        return self.index_end - self.index_start

    def get_min_index(self):
        return self.history

    def get_index(self):
        return self.index

    def get_Ask(self):
        return np.min(self.order_sell[:, 0, self.index])

    def get_Bid(self):
        return np.max(self.order_buy[:, 0, self.index])

    def get_value(self):
        Btc_before = self.get_BTC()
        coin_before = self.get_coin()
        return Btc_before + coin_before * self.get_Bid()
