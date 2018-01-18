# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 20:59:46 2017

@author: Reg

This script is to syncronize the data between my local mongoDB and the aws instance.
"""

import urllib2
import time
import hmac, hashlib
import json
from pymongo import MongoClient
from pprint import pprint
from numpy import floor
import threading

# connect to MongoDB,
mongoDB_collect = MongoClient('mongodb connect string')
mongoDB_store   = MongoClient('mongodb connect string')

db_remote = mongoDB_collect.bittrex
db_local  = mongoDB_store.bittrex


def dl_500():
    # Get min index
    min_index = db_remote.btc_ltc.find_one(sort=[("index", 1)])['index']
    
    # Get max index
    max_index = db_remote.btc_ltc.find_one(sort=[("index", -1)])['index']
    
    if max_index - min_index > 500:
        
        for ii in range(min_index, min_index+500):
            # get data
            test = db_remote.btc_ltc.find_one({'index':ii})
            
            if test is not None:
                del test['_id']
            
                # load into local
                db_local.btc_ltc.replace_one({'index':ii},test, upsert=True)
                
                # Remove entry
                db_remote.btc_ltc.delete_one({'index':ii})
            
            if ii % 10 == 0 : print ii


# Get min/max index
min_index = db_remote.btc_ltc.find_one(sort=[("index", 1)])['index']
max_index = db_remote.btc_ltc.find_one(sort=[("index", -1)])['index']

while max_index - min_index > 2000:
    dl_500()
    
    min_index = db_remote.btc_ltc.find_one(sort=[("index", 1)])['index']
    max_index = db_remote.btc_ltc.find_one(sort=[("index", -1)])['index']




