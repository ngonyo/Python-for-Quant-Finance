# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:28:11 2022

@author: ngonyo
"""

import pandas_datareader.data as reader
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns

end = dt.datetime.now()
start = dt.date(end.year - 1, end.month,end.day)

portfolio = ['GOOG', 'AAPL', 'MSFT', 'TSLA', '^GSPC']
df = reader.get_data_yahoo(portfolio,start,end)['Adj Close']

returns = df.pct_change()
returns.cov()
returns.var()
returns.corr()

sns.heatmap(returns.corr())
