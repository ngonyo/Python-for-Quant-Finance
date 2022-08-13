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

#do same for 1day correlations? with weighting over last 20 periods, half life of 10 days
#plots of historical vol?

#daily update of master list of 5 min data
#email dimitri 

import os
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
        


#~~~~~~~~~~~~~~~~

numdays = 204
base = dt.date.today()
date_list = [base - dt.timedelta(days=x) for x in range(numdays)]
df5min = pd.DataFrame()#initialize new dataframe
    
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
    #print(dff.head(2))
    #print(dff.tail(2))
    #print(dff.size)
    #print(dff.isna().sum()
    
    df5min = df5min.append(dff) #start adding each day's 5min entries into dataframe
    print(df5min.head(2))
    print(df5min.tail(2))
    print(df5min.size)
    print(df5min.isna().sum())
    print(df5min['date'].duplicated().any())
    print(os.listdir())
    print(os.getcwd())
    df5min.to_csv('SPY_IEF_5MIN JAN22-JULY22.csv', sep='\t')
    
    
#~~~~~~~~~~~~~~~~

#load in todays data
dftx = blp.bdib(ticker = 'SPY US Equity', dt = dt.date(2022, 7, 18), session='day')
dftx.reset_index(inplace=True)
dftx.columns = ['datex', 'b', 'c', 'd', 'spy_close', 'f', 'g', 'h']
dftx = dftx.drop(columns=['b', 'c', 'd', 'f', 'g', 'h'])
dftx['filter'] = [i.minute % 5 for i in pd.to_datetime(dftx['datex'])]
dftx = dftx[dftx['filter'] == 0]
dftx = dftx.drop(columns=['filter'])
returnsSPYtoday = dftx['spy_close'].pct_change().mul(100)
dftx['returnsSPY'] = returnsSPY
dftx.columns = ['date', 'spy_close', 'returns_SPY']
print(dftx.head(3))
print(dftx.tail(3))
print(dftx.shape)
print(dftx.isna().sum())

dfty = blp.bdib(ticker = 'IEF US Equity', dt = dt.date(2022, 7, 18), session='day', )
dfty.reset_index(inplace=True)
dfty.columns = ['datey', 'b', 'c', 'd', 'ief_close', 'f', 'g', 'h']
dfty = dfty.drop(columns=['b', 'c', 'd', 'f', 'g', 'h'])
dfty['filter'] = [i.minute % 5 for i in pd.to_datetime(dfty['datey'])]
dfty = dfty[dfty['filter'] == 0]
dfty = dfty.drop(columns=['filter'])
returnsIEF = dfty['ief_close'].pct_change().mul(100)
dfty['returns_IEF'] = returnsIEF
dfty.columns = ['date', 'ief_close', 'returns_IEF']
print(dfty.head(3))
print(dfty.tail(3))
print(dfty.shape)
print(dfty.isna().sum())

#merge daily data from each security into one dataframe
dfdaily = pd.merge(dftx, dfty, how = 'inner', on = ['date'])
dfdaily = dfdaily.drop(columns = ['spy_close', 'ief_close'])
dfdaily = dfdaily.dropna(thresh=2) #keep only rows with 2 non NA values
print(dfdaily.head(2))
print(dfdaily.tail(2))
print(dfdaily.size)
list(dfdaily.columns)

#~~~~~~~~~~~~~~~~


#read in csv file
#add new day's data to csv file

data5minhistorical = pd.read_csv('SPY_IEF_5MIN JAN22-JULY22.csv', sep='\t')
data5minhistorical.columns = ['a', 'date', 'returns_SPY', 'returns_IEF']
data5minhistorical = data5minhistorical.drop(columns=['a'])
data5minhistorical['date'] = pd.to_datetime(data5minhistorical['date'])
datafull = pd.concat([data5minhistorical,dfdaily]) #add daily 5 min data at bottom of dataframe
datafull = datafull.drop_duplicates(subset=(['date']), keep = False) #if duplicate time data included, remove
datafull = datafull.sort_values(by='date')
#datafull.index = datafull.date
#datafull.reset_index(inplace=True)
print(datafull.head(3))
print(datafull.tail(3))
print(datafull.shape)

###creating decay colunmn for 5min data
decay5Min = pd.DataFrame()#initialize new dataframe
decay5Min.insert(0, '5Min_Position', range(0, 1501))
decay5Min['DecayExp'] = decay5Min['5Min_Position']/390
decay5Min['5Min_Decay'] = np.power(.5, (decay5Min['DecayExp']))
decay5Min.head(3)
decay5Min.tail(3)

###creating decay colunmn for 1day data
decay1D = pd.DataFrame()#initialize new dataframe
decay1D.insert(0, '1D_Position', range(0, 21))
decay1D['DecayExp'] = decay1D['1D_Position']/5
decay5Min['1D__Decay'] = np.power(.5, (decay1D['DecayExp']))
decay1D.head(3)
decay1D.tail(3)

datafull.index
for row in datafull.itertuples():
    datafull["decayedcorr"] = WeightedCorr(x=pd.Series(datafull['returns_SPY'].rolling(1500)), y=pd.Series(datafull['returns_IEF'].rolling(1500)), w=pd.Series(decay5Min['5Min_Decay']))

print(datafull.head(3))
print(datafull.tail(3))





