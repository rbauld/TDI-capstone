# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 20:59:46 2017

@author: Reg
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

mongoDB_store = MongoClient('mongodb://127.0.0.1:27017')

# start_index = 9999
# end_index = 58000

db_local  = mongoDB_store.bittrex.btc_ltc
   

# Clean between these indexes
start_ii = 980000
end_ii = 981000


for ii in range(start_ii, end_ii):
    test = db_local.find_one({"index":ii})
    
    if test is not None:
        if test['orderbook']['result']['sell']:
            last_working = test
        else:
            test['orderbook']['result']['sell'] = last_working['orderbook']['result']['sell']
            test['orderbook']['result']['buy'] = last_working['orderbook']['result']['buy']
    else:
        last_working['index'] += 1
        test = last_working
    
    if '_id' in test.keys():
        del test['_id']
    # load into local
    db_local.replace_one({'index':ii},test, upsert=True)
    
    test['orderbook']['result']['sell']      

    print ii




