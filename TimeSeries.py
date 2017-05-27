# -*- coding: utf-8 -*-
"""
Created on Fri May 26 23:59:41 2017

@author: AM
"""
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_model import ARIMA


def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

def plot(df1):
    lag_acf = acf(df1, nlags=30)
    lag_pacf = pacf(df1, nlags=30, method='ols')
    #Plot ACF: 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(df1)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(df1)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(df1)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(df1)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
df1 = pd.read_csv('merge_power.csv', parse_dates = ['record_date'], 
                  index_col = 'record_date', date_parser = dateparse)
df1 = df1.iloc[:,1]
#df1.to_csv('merge_power2.csv')
df1 = df1.astype('float64')
model = ARIMA(df1, order=(0, 1, 99))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(df1)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-df1)**2))
