# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:11:43 2022

@author: Bence Mány

Introduction to Financial Engineering

Portfolio Management Project
"""

import sys
sys.path.append('../../')
from Exercises import finance_functions as ff
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.stats import gmean
import pandas as pd
import statistics as stat
from scipy import stats
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from arch import arch_model
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

TICKERS = sorted(["GOOGL", "KO", "AAPL", "AMZN", "PFE", "AMT", "XOM", "JPM"])
START = "2016-01-01"
END = "2022-01-01"
INTERVAL = "1d"


def historical_data(tickers, stock_data, daily_returns):
    for i in tickers:
        plt.plot(stock_data["Adj Close"][i], label = i)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('USD $', fontsize=20)
    plt.grid()
    plt.legend(fontsize = 15)
    plt.title("Historical daily prices", fontsize = 30)
    
    fig2 = plt.figure(2)
    for i in tickers:
        plt.plot(daily_returns[i], label = i)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('USD $', fontsize=20)
    plt.grid()
    plt.legend(fontsize = 15)
    plt.title("Daily returns", fontsize = 30)
    return

def return_analysis(tickers, daily_returns):
    
    #Calculate mean annual returns  
    mean_daily_returns = gmean(daily_returns + 1) - 1
    mean_annual_returns = (1 + mean_daily_returns)**252 - 1   

    print("\n=== Mean annual returns ===")       
    for etf, geomean_annual in zip(tickers, mean_annual_returns):
        print("{}: {}%".format(etf, round(geomean_annual*100, 2)))

    
    #Calculate standard deviation of returns
    sd_daily = []
    for i in tickers:
        sd_daily.append(stat.stdev(daily_returns[i]))
    sd_annual = [np.sqrt(252) * item for item in sd_daily]
    
    print('\n===Annualised standard deviation of returns===')
    for i in range(len(tickers)):
        print(f"{tickers[i]}: {round(sd_annual[i]*100, 2)}%")
    
    
    #Calculate correlation between returns
    corrmat = daily_returns.corr()
    
    print('\n===Correlation matrix of daily returns===')
    print(corrmat)
    
    
    #Histogram of returns to show distribution
    figure = plt.figure()
    #plt.hist(daily_returns, density = True, bins=30)
    ax = sns.histplot(daily_returns, kde = True, stat = "density", bins = 40)
    plt.title('Histogram of Weekly Returns', fontsize=30)
    plt.xlabel('Return', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    
    #Do we need anything else here?
    
    #Calculating Sharpe ratio for the stocks
    sharpe = mean_annual_returns / sd_annual
    print("\n=== Sharpe ratio of stocks ===")
    for i in range(len(tickers)):
        print(f"{tickers[i]}: {round(sharpe[i], 3)}")
        
    print("\nCopy to overleaf")
    print("=== Stock \t Annual return \t St dev \t Sharpe r ===")
    for i in range(len(tickers)):
        print(f"{tickers[i]}: &\t {round(mean_annual_returns[i]*100, 2)} &\t {round(sd_annual[i]*100, 2)} &\t {round(sharpe[i], 3)} \\")
        
        
        
    #Calculate range, skewness, kurtosis of returns
    ranges = [max(daily_returns[stock]) - min(daily_returns[stock]) for stock in tickers]
    kurtosis = [stats.kurtosis(daily_returns[stock]) for stock in tickers]
    skewness = [stats.skew(daily_returns[stock]) for stock in tickers]
    
    print("\n=== Stock \t Range \t Kurtosis \t Skewness ===")
    for i in range(len(tickers)):
        print(f"{tickers[i]}: \t {round(ranges[i]*100, 2)}% \t {round(kurtosis[i], 3)} \t {round(skewness[i], 3)}")
        
    return

def autocorrelation(tickers, time_series):
    location = "pictures\\acf_pacf\\"
    
    # #Plotting auto-correlation and partial auto-correlation of selected stocks then saving them to file
    # for stock in tickers:
    #     fig1 = plot_acf(time_series[stock], lags = 30, title = f"Auto-correlation of {stock}")
    #     fig1.savefig(location + "acf_" + stock)   
    #     plt.close(fig1) 
        
    #     fig2 = plot_pacf(time_series[stock], lags = 30, title = f"Partial auto-correlation of {stock}", method='ywm')
    #     fig2.savefig(location + "pacf_" + stock)   
    #     plt.close(fig2)    
        
        
       
    selected_stocks = ["GOOGL", "PFE"]
    
    for stock in selected_stocks:
        
        #Seasonal decomposition
        result=seasonal_decompose(time_series[stock], period=12)
        result.plot()
        
        #Stationarity test
        p = adfuller(time_series[stock])[1]
        print(f"\n{stock}: p-value is {p}")
        if p > 0.05:
            print("Therefore the time series is not stationary")
        else:
            print("Therefore the time series is stationary")
        
    return

def build_model(daily_returns):
    
    stock1 = "GOOGL"
    stock2 = "PFE"
    


            
    model = ARIMA(daily_returns[stock1], order = (1,1,0))  #AR = 1, diff = 1, MA=0
    results = model.fit()
    print("AIC: ", round(results.aic, 3), "\nBIC: ", round(results.bic, 3))
    
    
    #TODO: Estimation for GARCH model
    #model_ARMA_train_residuals = daily_returns[stock1] - model.predict()
    #model_GARCH = arch_model(model_ARMA_train_residuals, p = 1, q = 1, mean = 'zero', dist = 'normal')
    #model_GARCH = model_GARCH.fit()
    
    return


def main(test=False):
    
    stock_data = yf.download(
    tickers = TICKERS,
    start = START,
    end = END,
    interval = INTERVAL
    ).dropna()
    stock_data.tail()
    
    """------------------------------ DATA PRESENTATION ----------------------------------------"""

    #Calculate daily returns
    daily_returns = stock_data["Adj Close"].pct_change().dropna()
    
    #Plotting historical prices, returns
    #historical_data(TICKERS, stock_data, daily_returns)
    
    #Calculate annualised mean and covariance
    #return_analysis(TICKERS, daily_returns)

    
    
    """------------------------------ DATA ANALYSIS ----------------------------------------"""
    
    autocorrelation(TICKERS, daily_returns)
    
    build_model(daily_returns)


    print("merging test")
    


if __name__ == "__main__":
    main()
    


