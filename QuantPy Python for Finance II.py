# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 09:56:12 2022

@author: ngonyo
"""

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data as pdr
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pyo.init_notebook_mode(connected = True)
pd.options.plotting.backend = 'plotly'
import plotly.io as pio
pio.renderers.default = 'browser'

################################################################################
#                   Historical Volatility & Risk-Return Ratios                 #
################################################################################

end = dt.datetime.now()
start = dt.datetime(2015,1,1)
df = pdr.get_data_yahoo(['SPY', 'GLD', 'QQQ', 'IEF'], start, end)
Close = df.Close
Close.head()

#computing log returns
log_returns = np.log(df.Close/df.Close.shift(1)).dropna()
log_returns

#daily sd
daily_std = log_returns.std()
daily_std

#annualized sd
annualized_vol = daily_std * np.sqrt(252)
annualized_vol

#plot histogram of log returns with annualized vol
fig = make_subplots(rows = 2, cols = 2)
trace0 = go.Histogram(x=log_returns['SPY'], name = 'SPY')
trace1 = go.Histogram(x=log_returns['GLD'], name = 'GLD')
trace2 = go.Histogram(x=log_returns['QQQ'], name = 'QQQ')
trace3 = go.Histogram(x=log_returns['IEF'], name = 'IEF')
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 2, 2)

fig.show()
fig.update_layout(autosize = False, width = 800, height = 800, title = "Frequency of log returns",
                  xaxis = dict(title = 'SPY Annualized Vol: ' +str(np.round(annualized_vol['SPY']*100,1))),
                  xaxis2 = dict(title = 'GLD Annualized Vol: ' +str(np.round(annualized_vol['GLD']*100,1))),
                  xaxis3 = dict(title = 'QQQ Annualized Vol: ' +str(np.round(annualized_vol['QQQ']*100,1))),
                  xaxis4 = dict(title = 'IEF Annualized Vol: ' +str(np.round(annualized_vol['IEF']*100,1))),
                  )
fig.show()

#Trailing volatility over time
Trading_days = 60
volatility = log_returns.rolling(window = Trading_days).std()*np.sqrt(Trading_days)
volatility.plot().update_layout(autosize = False, width = 600, height = 300)

#sharpe ratio
Rf = 0.01/252
sharpe_ratio = (log_returns.rolling(window=Trading_days).mean() - Rf)*Trading_days/volatility
sharpe_ratio.plot().update_layout(autosize = False, width = 600, height = 300)

#sortino ratio
#sortino ratio only considers downside variance, ignores upside vol
sortino_vol = log_returns[log_returns<0].rolling(window = Trading_days, center = True, min_periods=10).std()*np.sqrt(Trading_days)
sortino_ratio = (log_returns.rolling(window=Trading_days).mean() - Rf)*Trading_days/sortino_vol
sortino_ratio.plot().update_layout(autosize = False, width = 600, height = 300)

#modigliani raio (M2 ratio)
#m2 ratio measures returns of portfolio adjusted for the risk of the portfolio relative to that of some benchmark
m2_ratio = pd.DataFrame()
benchmark_vol = volatility['SPY']
for c in log_returns.columns:
    if c != 'SPY':
        m2_ratio[c] = (sharpe_ratio[c]*benchmark_vol/Trading_days + Rf)*Trading_days
m2_ratio.plot().update_layout(autosize=False, width=600, height=300)

#max drawdown
def max_drawdown(returns):
    cumulative_returns = (1+returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns/peak) - 1
    return drawdown.min()

returns = df.Close.pct_change().dropna()
max_drawdowns = returns.apply(max_drawdown, axis = 0)
max_drawdowns*100

#Calmar ratio
#calmar ratio uses max drawdown in denominator as opposed to standard deviation/volatility
calmars = np.exp(log_returns.mean()*252)/abs(max_drawdowns)
calmars.plot.bar().update_layout(autosize=False, width=600, height=300)







        