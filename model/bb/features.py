#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 23:16:00 2017

@author: reg

Completed features are packaged here. I am not sure how I want to package features.
"""

import pandas as pd
import numpy as np


def rsi(series):
    """
    Relative strength index
    :param series:
    :return:
    """

    delta = pd.Series(series).diff()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]

    u_mean = np.mean(u)
    d_mean = np.mean(d)

    if d_mean == 0:
        return 100/float(100)
    else:
        rs = u_mean/d_mean
        return (100 - 100 / (1 + rs))/float(100)


def cci_long(high, low, close, series):
    """
    Commodity chanel index
    :param high:
    :param low:
    :param close:
    :param series:
    :return:
    """
    tp_mean = np.mean(high+low+series)
    tp_mad = np.mean(np.absolute(series-tp_mean))

    return (1/0.015)*(series[-1]-tp_mean)/tp_mad/100


def cci_short(series):
    high = np.max(series)
    low = np.min(series)
    tp_mean = np.mean(high+low+series)
    tp_mad = np.mean(np.absolute(series-tp_mean))

    return (1/0.015)*(series[-1]-tp_mean)/tp_mad/100


def stoc_osc(series):
    # Stocastic oscilator (sort of)
    high = np.max(series)
    low = np.min(series)
    return (series[-1]-low)/(high-low)
