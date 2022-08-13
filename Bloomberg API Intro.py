# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from xbbg import blp, pipeline

#checking version of blp
blp.__version__

#simple stocks and info
blp.bdp('AAPL US Equity', flds=['Security_Name', 'Last_Price'])
blp.bdp('6758 JP Equity', flds='Crncy_Adj_Mkt_Cap', Eqy_Fund_Crncy='USD')
blp.bdp(tickers='NVDA US Equity', flds=['Security_Name', 'GICS_Sector_Name'])
blp.bdp('AAPL US Equity', 'Eqy_Weighted_Avg_Px', VWAP_Dt='20181224')

holders = blp.bds('AMZN US Equity', flds='All_Holders_Public_Filings', cache=True)
(
    holders
    .loc[:, ~holders.columns.str.contains(
        f'holder_id|portfolio_name|change|number|'
        f'metro|percent_of_portfolio|source'
    )]
    .rename(
        index=lambda tkr: tkr.replace(' Equity', ''),
        columns={
            'holder_name_': 'holder',
            'position_': 'position',
            'filing_date__': 'filing_dt',
            'percent_outstanding': 'pct_out',
            'insider_status_': 'insider',
        }
    )
).head()

#showing dividends and earnings data
blp.dividend('SPY US Equity', start_date='2019')
blp.earning('FB US Equity', Eqy_Fund_Year=2018, Number_Of_Periods=2)

#Historical data with Excel compatible overrides
blp.bdh(
    tickers='SHCOMP Index', flds=['high', 'low', 'last_price'],
    start_date='2019-11', end_date='2020', Per='W', Fill='P', Days='A',
)

#Dividend / split adjustments
pd.concat([
    blp.bdh(
        'AAPL US Equity', 'Px_Last', '20140605', '20140610',
        CshAdjNormal=True, CshAdjAbnormal=True, CapChg=True
    ).rename(columns={'Px_Last': 'Px_Adj'}),
    blp.bdh(
        'AAPL US Equity', 'Px_Last', '20140605', '20140610',
        CshAdjNormal=False, CshAdjAbnormal=False, CapChg=False
    ).rename(columns={'Px_Last': 'Px_Raw'}),
], axis=1)




### Intraday Bars ###
cur_dt = pd.Timestamp('today', tz='America/New_York').date()
recent = pd.bdate_range(end=cur_dt, periods=2, tz='America/New_York')
pre_dt = max(filter(lambda dd: dd < cur_dt, recent))
pre_dt.date()

blp.bdib('QQQ US Equity', dt=pre_dt, session='day').tail()
blp.bdib('388 HK Equity', dt=pre_dt, session='am_open_7')

###Intraday tick data
blp.bdtick('QQQ US Equity', dt=pre_dt).tail(6)

###Equity Screen BEQS
blp.beqs('Core Capital Ratios', typ='GLOBAL').iloc[:5, :6]


#Subscription
#blp.live will yield market data as dict
async for snap in blp.live(['ESA Index', 'NQA Index'], max_cnt=2):
    print(snap)


### Pipelines ###
cur_dt = pd.Timestamp('today', tz='America/New_York').date()
recent = pd.bdate_range(end=cur_dt, periods=2, tz='America/New_York')
pre_dt = max(filter(lambda dd: dd < cur_dt, recent))

fx = blp.bdib('JPY Curncy', dt=pre_dt)
jp = pd.concat([
    blp.bdib(ticker, dt=pre_dt, session='day')
    for ticker in ['7974 JP Equity', '9984 JP Equity']
], axis=1)
jp.tail()


#get close prices and convert to USD
prices = (
    jp
    .pipe(pipeline.get_series, col='close')
    .pipe(pipeline.apply_fx, fx=fx)
    .tz_convert('Asia/Tokyo')
)
prices.tail()


### Customized Pipelines ###
#VWAP for intraday bar data

def vwap(data: pd.DataFrame, fx=None, name=None) -> pd.Series:
    return pd.Series({
        ticker: (
            data[ticker][['close', 'volume']].prod(axis=1).sum()
            if fx is None else (
                data[ticker].close
                .pipe(pipeline.apply_fx, fx)
                .close
                .mul(data[ticker].volume)
                .sum()
            )
        ) / data[ticker].volume.sum()
        for ticker in data.columns.get_level_values(0).unique()
    }, name=name)

#VWAP in local currency
jp.pipe(vwap, name=jp.index[-1].date())

#VWAP in USD
jp.pipe(vwap, fx=fx, name=jp.index[-1].date())



#######Total traded volume as of time in day for past few days
jp_hist = pd.concat([
    pd.concat([
        blp.bdib(ticker, dt=dt, session='day')
        for ticker in ['7974 JP Equity', '9984 JP Equity']
    ], axis=1)
    for dt in pd.bdate_range(end='today', periods=10)[:-1]
], sort=False)
#unique(jp_hist.index.date)


def drop_zeros(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data
        .replace(0, np.nan)
        .dropna(how='all')
        .replace(np.nan, 0)
    )

def traded_volume(data: pd.DataFrame, asof: str) -> pd.DataFrame:
    return (
        data
        .pipe(pipeline.get_series, col='volume')
        .between_time('0:00', asof)
        .resample('B')
        .sum()
        .pipe(drop_zeros)
    )
jp_hist.pipe(traded_volume, asof='10:00')

