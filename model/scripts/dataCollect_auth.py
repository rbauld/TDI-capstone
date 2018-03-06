#!/home/ec2-user/anaconda2/envs/btc-bot/bin/python
"""
Created on Mon Jul 17 20:59:46 2017

@author: Reg

This script is for collecting data from the bittrex servers.

Currently this particular script will pull down pricing and order book data and attempt to store
into a local mongoDB. This script is indended to be run as a cron job at the desired time-stamp frequency.
Ideally this can be done on an aws instance.

The script is configured only for the btc-ltc market. I may expand this in the future.
"""

import urllib2
import time
import hmac, hashlib
import json
from pymongo import MongoClient
from pprint import pprint
from numpy import floor
import sys


class BittrexWrap(object):
    def __init__(self, apikey, apisecret):
        self.apikey = apikey
        self.apisecret = apisecret

    def get_open_orders(self):
        nonce = str(int(time.time() * 1000))
        uri = 'https://bittrex.com/api/v1.1/market/getopenorders?apikey=' + self.apikey + '&nonce=' + nonce
        sign = hmac.new(self.apisecret, uri, hashlib.sha512).hexdigest()
        req = urllib2.Request(uri, None, {'apisign': sign})
        resp = urllib2.urlopen(req)
        return resp.read()

    def get_ticker(self, market):
        uri = 'https://bittrex.com/api/v1.1/public/getticker?market=' + market
        req = urllib2.Request(uri)
        resp = urllib2.urlopen(req)
        return resp.read()

    def get_market_sum(self, market):
        uri = 'https://bittrex.com/api/v1.1/public/getmarketsummary?market=' + market
        req = urllib2.Request(uri)
        resp = urllib2.urlopen(req)
        return resp.read()

    def get_order_book(self, market, Type, depth):
        uri = 'https://bittrex.com/api/v1.1/public/getorderbook?market=' + market + '&type=' + Type + '&depth=' + depth
        req = urllib2.Request(uri)
        resp = urllib2.urlopen(req)
        return resp.read()


def get_data_point():
    # Get last index
    last_index = db.btc_ltc.find_one(sort=[("index", -1)])

    if not last_index:
        ii = 0
    else:
        ii = last_index['index'] + 1

    timestamp = str(int(floor(time.time())))
    orderbook = bit.get_order_book('BTC-LTC', 'both', '20')
    marketsum = bit.get_market_sum('BTC-LTC')
    ticker = bit.get_ticker('BTC-LTC')

    dat = [timestamp, orderbook, marketsum, ticker, str(ii)]
    dat = '{{"timestamp":{0},"orderbook":{1},"marketsum":{2},"ticker":{3},"index":{4} }}'.format(*dat)

    result = db.btc_ltc.insert_one(json.loads(dat))


if __name__ == "__main__":

    apikey = 'nope'
    apisecret = 'nope'
    nonce = str(int(time.time() * 1000))
    uri = 'https://bittrex.com/api/v1.1/market/getopenorders?apikey=' + apikey + '&nonce=' + nonce

    sign = hmac.new(apisecret, uri, hashlib.sha512).hexdigest()

    req = urllib2.Request(uri, None, {'apisign': sign})
    # req.add_header('apisign:', sign)
    resp = urllib2.urlopen(req)
    content = resp.read()

    bit = BittrexWrap(apikey, apisecret)

    # connect to MongoDB,
    client = MongoClient('mongodb://%s:%s@127.0.0.1' % ('username', 'mongodb_password_here'))
    db = client.bittrex
    # Issue the serverStatus command and print the results
    serverStatusResult = db.command("serverStatus")
    pprint(serverStatusResult)

    get_data_point()

    try:
        sys.stdout.close()
    except:
        pass
