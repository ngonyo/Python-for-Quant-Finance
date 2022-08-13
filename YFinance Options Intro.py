# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:23:59 2022

@author: ngonyo
"""

# dependencies
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import mpl_toolkits
from mpl_toolkits import mplot3d
from datetime import datetime
from itertools import chain
from matplotlib import cm

# choose a ticker and get data via yfinance
ticker = "SPY"
stock = yf.Ticker(ticker)
# store maturities
lMaturity = list(stock.options)
print(lMaturity) #print option expiries


'''
lets convert the option expiries into days to expiration. this will be helpful if we wish to price the options later. 
'''

# get current date
today = datetime.now().date()
# empty list for days to expiration
lDTE = []
# empty list to store data for calls
lData_calls = []
# loop over maturities
for maturity in lMaturity:
    # maturity date
    maturity_date = datetime.strptime(maturity, '%Y-%m-%d').date()
    # DTE: difference between maturity date and today
    lDTE.append((maturity_date - today).days)
    # store call data
    lData_calls.append(stock.option_chain(maturity).calls)
    
#now doing same for puts:    
lData_puts = []
# loop over maturities
for maturity in lMaturity:
    # maturity date
    maturity_date = datetime.strptime(maturity, '%Y-%m-%d').date()
    # DTE: difference between maturity date and today
    lDTE.append((maturity_date - today).days)
    # store call data
    lData_puts.append(stock.option_chain(maturity).puts)

lData_calls
lData_puts

#going forward we will only be writing code for calls but puts can be done the same way.


################################################################################################################################################################

'''
Remember that plot_trisurf requires 3 vectors? 
We’re currently stuck with a list of options data and days to expiration, so we need to unlist the options data, 
get the data we want (strike and implied volatility), and since the vectors must be the same length, 
we must extend the list of days to expiration appropriately. 

Currently, in the list of days to expiration, 
each element is unique, but as there are several strikes for each maturity date, we need to create a new list, 
where each unique date is repeated by how many strikes are available for that maturity date.
'''

# create empty lists to contain unlisted data
lStrike = []
lDTE_extended = []
lImpVol = []
for i in range(0,len(lData_calls)):
    # append strikes to list
    lStrike.append(lData_calls[i]["strike"])
    # repeat DTE so the list has same length as the other lists
    lDTE_extended.append(np.repeat(lDTE[i], len(lData_calls[i])))
    # append implied volatilities to list
    lImpVol.append(lData_calls[i]["impliedVolatility"])
    
# unlist list of lists
lStrike = list(chain(*lStrike))
lDTE_extended = list(chain(*lDTE_extended))
lImpVol = list(chain(*lImpVol))

lStrike
lDTE_extended
lImpVol     

'''
Now we can plot the volatility surface
'''

# initiate figure
fig = plt.figure(figsize=(7,7))
# set projection to 3d
axs = plt.axes(projection="3d")
# use plot_trisurf from mplot3d to plot surface and cm for color scheme
axs.plot_trisurf(lStrike, lDTE_extended, lImpVol, cmap=cm.jet)
# change angle
axs.view_init(30, 65)
# add labels
plt.xlabel("Strike")
plt.ylabel("DTE")
plt.title("Volatility Surface for $"+ticker+": IV as a Function of K and T")
plt.show()

#Note: Only the 11 closest maturities are included in the plot.


