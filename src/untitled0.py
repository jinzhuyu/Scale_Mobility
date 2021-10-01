# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:12:45 2021

@author: Jin-Zhu Yu
"""



import pandas as pd
from statsmodels.tsa.stattools import adfuller

def get_adf(pd_series):
    '''test the stationarity of time series using Augmented Dickey-Fuller (ADF), one of the unit root tests.
    # https://machinelearningmastery.com/time-series-data-stationary-python/
    '''
    # series = pd.read_csv('daily-total-female-births.csv', header=0, index_col=0, squeeze=True)
    X = pd_series.values
    result = adfuller(X)    
    adf_value = result[0]
    p_value = result[1] 
    # result[4].items()[result[4].items().keys()[1]]
    is_stat = False    
    if p_value <= 0.05:
        is_stat = True
    return adf_value, p_value, is_stat