# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 18:37:28 2022

@author: Nathan
"""

import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.optimize as spop
import matplotlib.pyplot as plt

###specifying parameters###
stocks = ['JPM', 'C']
start = '2019-12-31'
end = '2021-03-08'
fee = 0.001 #specify transaction cost
window = 252 #trading window
#threshhold of t value -- lower is better/more statistically significant
t_threshhold = -2.5

data = pd.DataFrame()
returns = pd.DataFrame()
for stock in stocks:
    prices = yf.download(stock, start, end)
    data[stock] = prices['Close']
    returns[stock] = np.append(data[stock][1:].reset_index(drop=True)/data[stock][:-1].reset_index(drop=True) - 1, 0) #calculates return from each day
    
data 
returns

###initializing arrays###
gross_returns = np.array([]) #start with empty numpy array
net_returns = np.array([]) #start with empty numpy array
t_s = np.array([]) #start with empty numpy array. we will want to see if series is becoming more or less cointegrated 
stock1 = stocks[0]
stock1
stock2 = stocks[1]
stock2

#now moving through the sample:
for t in range(window, len(data)):
    # defining unit root funtion: stock2 = a + b*stock1
    def unit_root(b):
        a = np.average(data[stock2][t-window:t] - b*data[stock1][t-window:t]) #a is average of stock 2 over timeframe
        fair_value = a + b*data[stock1][t-window:t]#calc fair price of stock2 based on above function
        diff = np.array(fair_value - data[stock2][t - window:t])#deviation from fair value
        diff_diff = diff[1:] - diff[:-1]#differences in differences - ie dynamics
        #this difference in differences will be used later to get unit root. this will allow us to calc dickey fuller t stat, compare to threshhold
        reg = sm.OLS(diff_diff, diff[:-1]) #y is difference in differences #x is lagged differences
        res = reg.fit() 
        return res.params[0]/res.bse[0] #bse = standard error
        #return output of unit root function - dickey fuller t stat
    #now we will vary b to minimize t stat, make as negative as possible
    res1 = spop.minimize(unit_root, data[stock2][t]/data[stock1][t], method='Nelder-Mead')#what to minimize, starting value (we can pick anything here), specify method (here would usuall nelder-meed or powell - nm better for more simple optimization problems)
    t_opt = res1.fun #optimized (min) value of t stat
    b_opt = float(res1.x) #optimized value of b, x variable
    a_opt = np.average(data[stock2][t-window:t] - b_opt*data[stock1][t-window:t])
    fair_value = a_opt + b_opt*data[stock1][t] #tells if stock2 is over or undervalued at time t
    #simulating trading
    if t == window: #very first day of trading
        old_signal = 0
    if t_opt > t_threshhold: #we dont trade if t opt greater than t threshold, sit in cash
        signal = 0
        gross_return = 0
    else:
        signal = np.sign(fair_value - data[stock2][t])
        # positive means stock2 undervalued, long2 short1. negative means 2 overvalued, long1 short2
        gross_return = signal*returns[stock2][t] - signal*returns[stock1][t]
    #calculating trading fees
    fees = fee*abs(signal - old_signal) #if stick with same strategy,no action fee is 0. if trade pay fees once. if go from short to long or vice versa pay fees twice
    net_return = gross_return - fees
    gross_returns = np.append(gross_returns, gross_return)
    net_returns = np.append(net_returns, net_return)
    t_s = np.append(t_s, t_opt)
    #interface: reporting daily positions and realised returns
    print('day'+str(data.index[t]))
    print('t')
    if signal == 0:
        print('no trading')
    elif signal ==1:
        print('long position in '+stock2+' and short position in '+stock1)
    else: 
        print('long position in '+stock1+' and short position in '+stock2)
    print('gross daily return: '+str(round(gross_return*100,2))+'%')
    print('net daily return: '+str(round(net_return*100,2))+'%')
    print('cumulative net return so far: '+str(round(np.prod(1+net_returns)*100-100,2))+'%')
    print('')
    old_signal = signal
#plotting equity curves
plt.plot(np.append(1, np.cumprod(1+gross_returns)))    
plt.plot(np.append(1, np.cumprod(1+net_returns)))         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    