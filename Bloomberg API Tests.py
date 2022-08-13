# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:53:40 2022

@author: ngonyo
"""

import pandas as pd
import numpy as np
from xbbg import blp, pipeline
import inspect
pd.options.display.max_columns = None
pd.set_option('display.expand_frame_repr', False)

#=@BDH("SPY US EQUITY", "LAST_Price", $A$2, $A$3, "BarType", "T", "BarSize", "5","cols=2;rows=10012")

cur_dt = pd.Timestamp('today', tz='America/New_York').date()
recent = pd.bdate_range(end=cur_dt, periods=3, tz='America/New_York')
pre_dt = max(filter(lambda dd: dd < cur_dt, recent))
pre_dt.date()
blp.bdib('SPY US Equity', dt=pre_dt, session='day').tail()
blp.bdib(ticker = 'SPY US Equity', dt=pre_dt, session='day').head()

blp.bdib(ticker = 'SPY US Equity', dt = '2022-07-01', session='day').tail()
blp.bdib(ticker = 'SPY US Equity', dt = '2022-07-01', session='day').tail()

df = blp.bdib(ticker = 'SPY US Equity', dt = '2022-07-01', session='day').tail()
df([0,1])
df.iloc[:,1:7]

blp.bdib(ticker = 'SPY US Equity', dt = '2022-01-03', typ = 'TRADE', session = 'day').head()
blp.bdib(ticker = 'SPY US Equity', dt = '2022-01-03', typ = 'TRADE', session = 'day').tail()


#############################

range1 = pd.bdate_range(start = '2022-01-03', end = '2022-07-08')

blp.bdib('SPY US Equity', dt=range1).head(8)
type(range1)
type(cur_dt)
type(recent)



cur_dt = pd.Timestamp('today', tz='America/New_York').date()
recent = pd.bdate_range(start = '2022-07-04', end=cur_dt, tz='America/New_York')
pre_dt = max(filter(lambda dd: dd < cur_dt, recent))
pre_dt.date()
blp.bdib('SPY US Equity', dt=pre_dt).head()

#############################

df1 = blp.bdh(
    tickers='SPY US Equity', flds=['VOLATILITY_90D', 'VOLATILITY_30D', 'last_price'],
    start_date='2022-01-03', end_date='2022-07-08', Per='D', Fill='P', Days='A',
)
df1.head()





type(df1)
df1.size #length of dataframe
df1.shape #dimensions of dataframe
df1.ndim #num of dimensions of dataframe

list(df1.columns.values)
list(df1.columns)
df1.iloc[1,] #first row
df1.iloc[:,1] #first column

df1['returns'] = df1 


