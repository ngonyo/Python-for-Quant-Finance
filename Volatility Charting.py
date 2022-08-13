# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:04:21 2022

@author: ngonyo
"""


import pandas as pd
import numpy as np
from xbbg import blp, pipeline
import inspect
pd.options.display.max_columns = None
pd.set_option('display.expand_frame_repr', False)
import datetime as dt
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pyo.init_notebook_mode(connected = True)
pd.options.plotting.backend = 'plotly'
import plotly.io as pio
pio.renderers.default = 'browser'


##################################################################### BTC 2 LEVEL TIME FRAME VOL CHART


df1 = blp.bdh(
    tickers='BITO US Equity', flds=['VOLATILITY_10D', 'VOLATILITY_30D', 'last_price', 'volume'],
    start_date='2022-01-03', end_date='2022-07-08', Per='D', Fill='P', Days='A',
)
df1.head()
df1.tail()
df1.shape
list(df1.columns)
df1.columns = ['VOLATILITY_10D', 'VOLATILITY_30D', 'last_price', 'volume']
list(df1.columns)


fig = make_subplots(rows = 2, cols = 1, shared_xaxes=True,
                    vertical_spacing = 0.1, subplot_titles = ('BITO, 10D/30D Volatility', 'Volume'),
                    row_width = [0.2,0.7])

fig.add_trace(go.Scatter(x = df1.index, y = df1['VOLATILITY_10D'], marker_color = 'purple', name = 'VOLATILITY_10D'))
fig.add_trace(go.Scatter(x = df1.index, y = df1['VOLATILITY_30D'], marker_color = 'pink', name = 'VOLATILITY_30D'))
fig.add_trace(go.Scatter(x = df1.index, y = df1['last_price'], marker_color = 'lightgrey', name = 'BITO'))
fig.add_trace(go.Bar(x = df1.index, y=df1['volume'], marker_color = 'green', showlegend = False), row = 2, col = 1)
fig.show()


##################################################################### OIL 3 LEVEL TIME FRAME VOL CHART


df2 = blp.bdh(
    tickers='USO US Equity', flds=['VOLATILITY_60D', 'VOLATILITY_30D', 'VOLATILITY_10D', 'last_price', 'volume'],
    start_date='2022-01-03', end_date='2022-07-08', Per='D', Fill='P', Days='A',
)
df2.head()
df2.tail()
df2.shape
list(df2.columns)
df2.columns = ['VOLATILITY_60D', 'VOLATILITY_30D', 'VOLATILITY_10D', 'last_price', 'volume']
list(df2.columns)


fig2 = make_subplots(rows = 2, cols = 1, shared_xaxes=True,
                    vertical_spacing = 0.1, subplot_titles = ('USO, 10D/30D/60D Volatility', 'Volume'),
                    row_width = [0.2,0.7])
fig2.add_trace(go.Scatter(x = df2.index, y = df2['VOLATILITY_60D'], marker_color = 'purple', name = 'VOLATILITY_60D'))
fig2.add_trace(go.Scatter(x = df2.index, y = df2['VOLATILITY_30D'], marker_color = 'violet', name = 'VOLATILITY_30D'))
fig2.add_trace(go.Scatter(x = df2.index, y = df2['VOLATILITY_10D'], marker_color = 'pink', name = 'VOLATILITY_10D'))
fig2.add_trace(go.Scatter(x = df2.index, y = df2['last_price'], marker_color = 'lightgrey', name = 'BITO'))
fig2.add_trace(go.Bar(x = df2.index, y=df2['volume'], marker_color = 'green', showlegend = False), row = 2, col = 1)
fig2.show()



##################################################################### LOADING 90,60,30,10 DAY VOL TODAY FOR ALL TICKERS OF DOW30

        
dow30tickers = [
'MMM',
'AXP',
'AAPL',
'BA',
'CAT',
'CVX',
'CSCO',
'KO',
'DOW',
'XOM',
'GS',
'HD',
'IBM',
'INTC',
'JNJ',
'JPM',
'MCD',
'MRK',
'MSFT',
'NKE',
'PFE',
'PG',
'TRV',
'UNH',
'UTX',
'VZ',
'V',
'WMT',
'WBA',
'DIS',
]    

#add us equity to end of ticker for bloomberg api to read it
mystring = ' US Equity'
dow30tickers = [s + mystring for s in dow30tickers]
print(dow30tickers)

dfvol = pd.DataFrame()
for i in dow30tickers:
    dftemp = blp.bdh(
        tickers= i, flds=['VOLATILITY_90D', 'VOLATILITY_60D', 'VOLATILITY_30D', 'VOLATILITY_10D', 'last_price'],
        start_date='2022-07-08', end_date='2022-07-08', Per='D', Fill='P', Days='A')
    dftemp.columns = ['VOLATILITY_90D', 'VOLATILITY_60D', 'VOLATILITY_30D', 'VOLATILITY_10D', 'last_price']
    dftemp = dftemp.append({'ticker': i}, ignore_index=True)
    dfvol = pd.concat([dfvol,dftemp])
#fixing structure so ticker is on same row as vol info
dfvol['ticker'] = dfvol['ticker'].shift(-1)
dfvol = dfvol.iloc[::2]
#moving ticker column to first column of dataframe, price column to second position
cols = dfvol.columns.tolist()
cols = cols[-1:] + cols[:-1]
dfvol = dfvol[cols]
dfvol = dfvol[['ticker', 'last_price', 'VOLATILITY_90D', 'VOLATILITY_60D', 'VOLATILITY_30D', 'VOLATILITY_10D']]
print(dfvol.head())
print(dfvol.tail())


#Displaying Top least and most volatile tickers at each time level
vol90top = dfvol.sort_values(by=['VOLATILITY_90D'], ascending=False)
vol90bot = dfvol.sort_values(by=['VOLATILITY_90D'], ascending=True)
print(vol90top.head())
print(vol90bot.head())
vol10top = dfvol.sort_values(by=['VOLATILITY_10D'], ascending=False)
vol10bot = dfvol.sort_values(by=['VOLATILITY_10D'], ascending=True)
print(vol10top.head())
print(vol10bot.head())


#Diplaying which tickers are most above or below their 90, 60, 30 day vol at 10D level
volcomp = dfvol
volcomp['VOL_10D/90D'] = volcomp['VOLATILITY_10D'] / volcomp['VOLATILITY_90D']
volcomp['VOL_10D/60D'] = volcomp['VOLATILITY_10D'] / volcomp['VOLATILITY_60D']
volcomp['VOL_10D/30D'] = volcomp['VOLATILITY_10D'] / volcomp['VOLATILITY_30D']
print(volcomp.head())

voltop = dfvol.sort_values(by=['VOL_10D/90D'], ascending=False)
volbot = dfvol.sort_values(by=['VOL_10D/90D'], ascending=True)
buyvol = (volbot.head(10))
sellvol = (voltop.head(10))
print("TICKERS WITH VOL AT LOW RELATIVE LEVELS:", buyvol)
print("TICKERS WITH VOL AT HIGH RELATIVE LEVELS:", sellvol)




##################################################################### LOADING 90,60,30,10 DAY VOL TODAY FOR GLOBAL MACRO
globalmacro = [
'GLD',
'GDX',
'SLV',
'SPY',
'QQQ',
'BITO',
'USO',
'HYG',
'EEM',
'FXI',
'XLE',
'EWZ',
]    

#add us equity to end of ticker for bloomberg api to read it
mystring = ' US Equity'
globalmacro = [s + mystring for s in globalmacro]
print(globalmacro)

dfvol = pd.DataFrame()
for i in globalmacro:
    dftemp = blp.bdh(
        tickers= i, flds=['VOLATILITY_90D', 'VOLATILITY_60D', 'VOLATILITY_30D', 'VOLATILITY_10D', 'last_price'],
        start_date='2022-07-08', end_date='2022-07-08', Per='D', Fill='P', Days='A')
    dftemp.columns = ['VOLATILITY_90D', 'VOLATILITY_60D', 'VOLATILITY_30D', 'VOLATILITY_10D', 'last_price']
    dftemp = dftemp.append({'ticker': i}, ignore_index=True)
    dfvol = pd.concat([dfvol,dftemp])
#fixing structure so ticker is on same row as vol info
dfvol['ticker'] = dfvol['ticker'].shift(-1)
dfvol = dfvol.iloc[::2]
#moving ticker column to first column of dataframe, price column to second position
cols = dfvol.columns.tolist()
cols = cols[-1:] + cols[:-1]
dfvol = dfvol[cols]
dfvol = dfvol[['ticker', 'last_price', 'VOLATILITY_90D', 'VOLATILITY_60D', 'VOLATILITY_30D', 'VOLATILITY_10D']]
print(dfvol.head())
print(dfvol.tail())


#Displaying Top least and most volatile tickers at each time level
vol90top = dfvol.sort_values(by=['VOLATILITY_90D'], ascending=False)
vol90bot = dfvol.sort_values(by=['VOLATILITY_90D'], ascending=True)
print(vol90top.head())
print(vol90bot.head())
vol10top = dfvol.sort_values(by=['VOLATILITY_10D'], ascending=False)
vol10bot = dfvol.sort_values(by=['VOLATILITY_10D'], ascending=True)
print(vol10top.head())
print(vol10bot.head())


#Diplaying which tickers are most above or below their 90, 60, 30 day vol at 10D level
volcomp = dfvol
volcomp['VOL_10D/90D'] = volcomp['VOLATILITY_10D'] / volcomp['VOLATILITY_90D']
volcomp['VOL_10D/60D'] = volcomp['VOLATILITY_10D'] / volcomp['VOLATILITY_60D']
volcomp['VOL_10D/30D'] = volcomp['VOLATILITY_10D'] / volcomp['VOLATILITY_30D']
print(volcomp.head())

voltop = dfvol.sort_values(by=['VOL_10D/90D'], ascending=False)
volbot = dfvol.sort_values(by=['VOL_10D/90D'], ascending=True)
buyvol = (volbot.head(int(len(globalmacro)/2)))
sellvol = (voltop.head(int(len(globalmacro)/2)))
print("TICKERS WITH VOL AT LOW RELATIVE LEVELS:", "\n", buyvol)
print("TICKERS WITH VOL AT HIGH RELATIVE LEVELS:", "\n", sellvol)

#add 1 day vol
