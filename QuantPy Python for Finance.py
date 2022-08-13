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

################################################################################
#       Python for Finance: getting stock data with pandas datareader          #
################################################################################

end = dt.datetime.now()
start = dt.datetime(2000,1,1)
start, end

stocklist = ['CBA', 'NAB', 'WBC', 'ANZ']
stocks = [i + '.AX' for i in stocklist]
stocks

df = pdr.get_data_yahoo(stocks, start, end)
df.head()
df.columns
df.index

Close=df.Close
Close.head()
Close.describe()
Close.describe(percentiles = [0.1,0.5,0.9])
Close[Close.index > end - dt.timedelta(days=100)].describe(percentiles = [0.1,0.5,0.9]) #descriptive stats for last 100 days

#plotting with matplotlib vs ploty
#matplotlib:
Close.plot(figsize=(12,8))
plt.show()

#plotly:
import plotly.offline as pyo
import plotly.graph_objects as go
import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
pyo.init_notebook_mode(connected = True)
pd.options.plotting.backend = 'plotly'
Close.plot()


#plotting return histogram:
Close['CBA.AX'].pct_change()
Close['CBA.AX'].pct_change().plot(kind = 'hist')

################################################################################
#              candlestick graphs with stock data using plotly                 #
################################################################################
from plotly.subplots import make_subplots

end = dt.datetime.now()
start = dt.datetime(2015,1,1)
df = pdr.get_data_yahoo('SPY', start, end)
df.head()

#creating moving average terms:
#we will use pandas .rolling function and speficy the rolling window parameter. min_periods parameter can be used to aboid Na data where MA didnt have enough data

df['MA50'] = df['Close'].rolling(window = 50).mean()
df['MA50'].head()
df['MA50'].tail()

#now days less than 50 will use as much data as possible to populate earlier periods
df['MA50'] = df['Close'].rolling(window = 50, min_periods = 0).mean()
df['MA200'] = df['Close'].rolling(window = 200, min_periods = 0).mean()

#creating plotly fig / subplot
fig = make_subplots(rows = 2, cols = 1, shared_xaxes=True,
                    vertical_spacing = 0.1, subplot_titles = ('SPY', 'Volume'),
                    row_width = [0.2,0.7])
fig.add_trace(go.Candlestick(x = df.index, open = df['Open'], high = df['High'], low = df['Low'], close = df['Close'], name = 'OHLC'), row = 1, col = 1)

#adding moving average terms:
fig.add_trace(go.Scatter(x = df.index, y = df['MA50'], marker_color = 'grey', name = 'MA50'))
fig.add_trace(go.Scatter(x = df.index, y = df['MA200'], marker_color = 'lightgrey', name = 'MA200'))

#adding volume bar chart in subplot
fig.add_trace(go.Bar(x = df.index, y=df['Volume'], marker_color = 'red', showlegend = False), row = 2, col = 1)

#update layout with labels and adjust sizes
fig.update_layout(
    title = 'SPY historical price chart',
    xaxis_tickfont_size = 12,
    yaxis = dict(
        title = 'Price ($/share)',
        titlefont_size=14,
        tickfont_size=12
        ),
    autosize = False,
    width = 800,
    height = 500,
    margin = dict(l=50, r=50, b=100, t=100, pad = 5),
    paper_bgcolor = 'lightseagreen'
)
fig.show()


#removing rangeslider from subplot
fig.update(layout_axis_rangeslider_visible = False)
fig.show()

################################################################################
#                    Are stock returns normally distributed                    #
################################################################################
import scipy.stats as stats
import pylab

#recall: SIMPLE RETURNS: the product of normally distributed variables is NOT normally distributed
#recall: LOG RETURNS: the sum of normally disributed variables DOES follow a normal distribution


simple_returns = df.Close.pct_change().dropna()
simple_returns

df.Close.plot()
df.Close.plot().update_layout(autosize = False, width = 500, height=300)

#use simple returns & attempt to compute final price from starting price over time horizon
print('First', df.Close[0], 'Last', df.Close[-1])
simple_returns.mean()
df.Close[0] * (1+simple_returns.mean())**len(simple_returns) #as we can see this doesnt work. accurate method below:
df.Close[0]* np.prod( [(1 + Rt) for Rt in simple_returns] ) #must multply 1+return for every period when working with simple returns

#computing log returns:
log_returns = np.log(df.Close/df.Close.shift(1)).dropna()
log_returns
log_returns.mean()    
df.Close[0]* np.exp(log_returns.mean() * len(log_returns)) #must multply 1+return for every period when working with simple returns

#log return histogram
log_returns.plot(kind = 'hist').update_layout(autosize = False, width = 500, height=300)
#are these log returns normally distributed?
log_returns_sorted = log_returns.tolist()
log_returns_sorted.sort()
worst = log_returns_sorted[0]
best = log_returns_sorted[-1]
std_worst = (worst - log_returns.mean())/log_returns.std()
std_best = (best - log_returns.mean())/log_returns.std()

print('Std dev. worst %.2f best %.2f' %(std_worst, std_best))
print('Probability worst %.23f best %.15f' %(stats.norm(0,1).pdf(std_worst), stats.norm(0,1).pdf(std_best)))
#seems like these returns are not normally distrubited

#log return qq plot to see how data is distributed
stats.probplot(log_returns, dist = 'norm', plot = pylab)
print('Q-Q Plot')
#we can see the presence of heavy tails in the underlying distribution
#if we were only dealing with data in the middle, normal dist would be reasonable

#log return box plots:
log_returns.plot(kind = 'box').update_layout(autosize = False, width = 500, height=300)
#we can see very fat tails to either size yet again

#hypothesis testing for normal dist: using KS test
#KS test computes distances between empiracal dist and theoretical dist and defines test stat as the supremum of the set of these distances
ks_stat, p_value = stats.kstest(log_returns, 'norm')
print(ks_stat, p_value)
if p_value > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')
    
#Shapiro Wilk Test:
#SW test is most powerful test when testing for a normal dist. It cannot be used for testing against other dist like the KS test can 
sw_stat, p_value = stats.shapiro(log_returns)
print(sw_stat, p_value)
if p_value > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')



