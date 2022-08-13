# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:42:31 2022

@author: ngonyo
"""

#general framework:
#load max amount of 5 min data
#export max amount of data to csv file
#load in 5 min data from yesterday, add to master 5 min list
#calculate weighted corr for last x 5 min periods, based on half life formula of ~300
#create dict with dates, each 5 min period's historical weighted correlations
#create plot of historical weighted correlations

from scipy.stats import rankdata
import pandas as pd
import numpy as np
import pandas_datareader.data as reader
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from xbbg import blp, pipeline
import inspect
pd.options.display.max_columns = None
pd.set_option('display.expand_frame_repr', False)

df1 = blp.bdh(
    tickers='SPY US Equity', flds=['last_price'],
    start_date='2022-01-03', end_date='2022-07-08', Per='D', Fill='P', Days='A',
)
df1.head()

df2 = blp.bdh(
    tickers='IEF US Equity', flds=['last_price'],
    start_date='2022-01-03', end_date='2022-07-08', Per='D', Fill='P', Days='A',
)
df2.head()

returnsSPY = df1.pct_change().mul(100)
returnsIEF = df2.pct_change().mul(100)
df1['returns_SPY'] = returnsSPY
df2['returns_IEF'] = returnsIEF
df1.columns = ['SPY_Price', 'SPY_returns']
df2.columns = ['IEF_Price', 'IEF_returns']
df1.head()
df2.head()
list(df1.columns)
list(df2.columns)

df3 = df1.join(df2)
df3.head()
del df3['SPY_Price'], df3['IEF_Price']
df3.head()
df3.tail()
df3.size
df3.shape


###creating decay colunmn
df3.insert(0, 'Position', range(0, 0 + len(df3)))
df3.head()
del df3['Decay1']

df4 = df3
df4['Position'] = range(0, 0+len(df3))
df4.tail()
df4['Decay'] = 1/df4['Position']
df4.head()
del df4['Decay_test']



class WeightedCorr:
    def __init__(self, xyw=None, x=None, y=None, w=None, df=None, wcol=None):
        ''' Weighted Correlation class. Either supply xyw, (x, y, w), or (df, wcol). Call the class to get the result, i.e.:
        WeightedCorr(xyw=mydata[[x, y, w]])(method='pearson')
        :param xyw: pd.DataFrame with shape(n, 3) containing x, y, and w columns (column names irrelevant)
        :param x: pd.Series (n, ) containing values for x
        :param y: pd.Series (n, ) containing values for y
        :param w: pd.Series (n, ) containing weights
        :param df: pd.Dataframe (n, m+1) containing m phenotypes and a weight column
        :param wcol: str column of the weight column in the dataframe passed to the df argument.
        '''
        if (df is None) and (wcol is None):
            if np.all([i is None for i in [xyw, x, y, w]]):
                raise ValueError('No data supplied')
            if not ((isinstance(xyw, pd.DataFrame)) != (np.all([isinstance(i, pd.Series) for i in [x, y, w]]))):
                raise TypeError('xyw should be a pd.DataFrame, or x, y, w should be pd.Series')
            xyw = pd.concat([x, y, w], axis=1).dropna() if xyw is None else xyw.dropna()
            self.x, self.y, self.w = (pd.to_numeric(xyw[i], errors='coerce').values for i in xyw.columns)
            self.df = None
        elif (wcol is not None) and (df is not None):
            if (not isinstance(df, pd.DataFrame)) or (not isinstance(wcol, str)):
                raise ValueError('df should be a pd.DataFrame and wcol should be a string')
            if wcol not in df.columns:
                raise KeyError('wcol not found in column names of df')
            self.df = df.loc[:, [x for x in df.columns if x != wcol]]
            self.w = pd.to_numeric(df.loc[:, wcol], errors='coerce')
        else:
            raise ValueError('Incorrect arguments specified, please specify xyw, or (x, y, w) or (df, wcol)')

    def _wcov(self, x, y, ms):
        return np.sum(self.w * (x - ms[0]) * (y - ms[1]))

    def _pearson(self, x=None, y=None):
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        mx, my = (np.sum(i * self.w) / np.sum(self.w) for i in [x, y])
        return self._wcov(x, y, [mx, my]) / np.sqrt(self._wcov(x, x, [mx, mx]) * self._wcov(y, y, [my, my]))

    def _wrank(self, x):
        (unique, arr_inv, counts) = np.unique(rankdata(x), return_counts=True, return_inverse=True)
        a = np.bincount(arr_inv, self.w)
        return (np.cumsum(a) - a)[arr_inv]+((counts + 1)/2 * (a/counts))[arr_inv]

    def _spearman(self, x=None, y=None):
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        return self._pearson(self._wrank(x), self._wrank(y))

    def __call__(self, method='pearson'):
        '''
        :param method: Correlation method to be used: 'pearson' for pearson r, 'spearman' for spearman rank-order correlation.
        :return: if xyw, or (x, y, w) were passed to __init__ returns the correlation value (float).
                 if (df, wcol) were passed to __init__ returns a pd.DataFrame (m, m), the correlation matrix.
        '''
        if method not in ['pearson', 'spearman']:
            raise ValueError('method should be one of [\'pearson\', \'spearman\']')
        cor = {'pearson': self._pearson, 'spearman': self._spearman}[method]
        if self.df is None:
            return cor()
        else:
            out = pd.DataFrame(np.nan, index=self.df.columns, columns=self.df.columns)
            for i, x in enumerate(self.df.columns):
                for j, y in enumerate(self.df.columns):
                    if i >= j:
                        out.loc[x, y] = cor(x=pd.to_numeric(self.df[x], errors='coerce'), y=pd.to_numeric(self.df[y], errors='coerce'))
                        out.loc[y, x] = out.loc[x, y]
            return out
        
WeightedCorr(x=df4['SPY_returns'], y=df4['IEF_returns'], w=df4['Decay'])()

#~~~~~~~~~~~~~~~~

numdays = 100
base = dt.date.today()
date_list = [base - dt.timedelta(days=x) for x in range(numdays)]
df5min = pd.DataFrame()
    
for x in date_list:
    dfx = blp.bdib(ticker = 'SPY US Equity', dt = x, session='day', )
    if dfx.empty:
        continue
    dfx.reset_index(inplace=True)
    dfx.columns = ['datex', 'b', 'c', 'd', 'spy_close', 'f', 'g', 'h']
    dfx = dfx.drop(columns=['b', 'c', 'd', 'f', 'g', 'h'])
    dfx['filter'] = [i.minute % 5 for i in pd.to_datetime(dfx['datex'])]
    dfx = dfx[dfx['filter'] == 0]
    dfx = dfx.drop(columns=['filter'])
    returnsSPY = dfx['spy_close'].pct_change().mul(100)
    dfx['returnsSPY'] = returnsSPY
    dfx.columns = ['date', 'spy_close', 'returns_SPY']
    #print(dfx.tail(3))
    #print(dfx.isna().sum())
    
    dfy = blp.bdib(ticker = 'IEF US Equity', dt = x, session='day', )
    if dfy.empty:
        continue
    dfy.reset_index(inplace=True)
    dfy.columns = ['datey', 'b', 'c', 'd', 'ief_close', 'f', 'g', 'h']
    dfy = dfy.drop(columns=['b', 'c', 'd', 'f', 'g', 'h'])
    dfy['filter'] = [i.minute % 5 for i in pd.to_datetime(dfy['datey'])]
    dfy = dfy[dfy['filter'] == 0]
    dfy = dfy.drop(columns=['filter'])
    returnsIEF = dfy['ief_close'].pct_change().mul(100)
    dfy['returns_IEF'] = returnsIEF
    dfy.columns = ['date', 'ief_close', 'returns_IEF']
    #print(dfy.tail(3))
    #print(dfy.isna().sum())
    
    dff = pd.merge(dfx, dfy, how = 'inner', on = ['date'])
    dff = dff.drop(columns = ['spy_close', 'ief_close'])
    dff = dff.dropna(thresh=2) #keep only rows with 2 non NA values
    print(dff.head(2))
    print(dff.tail(2))
    print(dff.size)
    print(dff.isna().sum())




for x in date_list:     
    if df5min.empty:
        df5min = dff
    else:
        df5min = df5min.append(dff)
    print(df5min.head())
    print(df5min.tail())
    
    
    
    #calc corr for that day
   # corr1d = WeightedCorr(x=dff['returns_SPY'], y=df4['returns_IEF'], w=df4['Decay'])()
    #corrdict = {
     #   "date" : x
      #  "1D_corr" : 
       # }
    
