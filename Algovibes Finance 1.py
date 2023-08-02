# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 18:37:28 2022

@author: Nathan
"""

import pandas_datareader.data as reader
import matplotlib.pyplot as plt
import yfinance as yf

data = yf.download("SPY", start = '2000-01-01', end = '2020-12-31')
#note yahoo finance datetimes are received as UTC
data
data.to_csv('spy.csv') #create csv file
#note adj close takes dividends into account, if paid on that day

volume = data['Volume']
volume

AdjClose = data['Adj Close']
AdjClose

volume.plot()
plt.show

#lets get daily returns
daily_returns = AdjClose.pct_change()
daily_returns
daily_returns.plot(color = 'purple')

#lets compare two stocks
stocks = ["AAPL", "MSFT"]
data2 = yf.download(stocks, start = '2004-01-01', end = '2020-12-31')
data2

AdjClose2 = data2['Adj Close']
AdjClose2
AdjClose2.plot()

#we need to compare are relative price changes, not absolute price changes
daily_returns2 = AdjClose2.pct_change()
daily_returns2
#define new variable to calculate daily cumulative returns
daily_cum_returns = (daily_returns2+1).cumprod()
daily_cum_returns
daily_cum_returns.plot()
plt.show()
#this shows true magnitude of outperformance. we must always compare relative price changes