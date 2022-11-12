# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:39:37 2022

@author: Bence Mány

Introduction to Financial Engineering

Functions for statistical analysis and data presentation
"""

import matplotlib.pyplot as plt
from scipy.stats import gmean
import statistics as stat
import numpy as np
import seaborn as sns
import statsmodels.graphics.gofplots as sm
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


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


def return_analysis(tickers, daily_returns, display = True):
    
    #Calculate mean annual returns  
    mean_daily_returns = gmean(daily_returns + 1) - 1
    mean_annual_returns = (1 + mean_daily_returns)**252 - 1   


    # Covariance matrix
    daily_cov_matrix = daily_returns.cov()
    annual_cov_matrix = daily_cov_matrix * 252


    #Calculate standard deviation of returns
    sd_daily = []
    for i in tickers:
        sd_daily.append(stat.stdev(daily_returns[i]))
    sd_annual = [np.sqrt(252) * item for item in sd_daily]
    
    
    #Calculate correlation between returns
    corrmat = daily_returns.corr()
    
    
    #Calculating Sharpe ratio for the stocks
    sharpe = mean_annual_returns / sd_annual
    
        
    #Calculate range, skewness, kurtosis of returns
    ranges = [max(daily_returns[stock]) - min(daily_returns[stock]) for stock in tickers]
    kurtosis = [stats.kurtosis(daily_returns[stock]) for stock in tickers]
    skewness = [stats.skew(daily_returns[stock]) for stock in tickers]
            
    if display:
        print("\n=== Mean annual returns ===")       
        for etf, geomean_annual in zip(tickers, mean_annual_returns):
            print("{}: {}%".format(etf, round(geomean_annual*100, 2)))
        
        print("=== Annual Covariance Matrix ===")
        print(annual_cov_matrix)
            
        print('\n===Annualised standard deviation of returns===')
        for i in range(len(tickers)):
            print(f"{tickers[i]}: {round(sd_annual[i]*100, 2)}%")
     
        print('\n===Correlation matrix of daily returns===')
        print(corrmat)
        fig_heat = plt.figure(3)
        plt.title("Correlation of returns")
        sns.heatmap(corrmat)
        
        figure = plt.figure()
        ax = sns.histplot(daily_returns, kde = True, stat = "density", bins = 40)
        plt.title('Histogram of Weekly Returns', fontsize=30)
        plt.xlabel('Return', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()
        
        #QQ plots for the stocks
        location = "pictures\\qq_plots\\"
        for stock in tickers:
            fig = sm.ProbPlot(daily_returns[stock]).qqplot(line='s')
            plt.title(f"QQ plot of {stock}")
            fig.savefig(location + "qq_" + stock)
            plt.close(fig)
            
            
        print("\n=== Sharpe ratio of stocks ===")
        for i in range(len(tickers)):
            print(f"{tickers[i]}: {round(sharpe[i], 3)}")
            
        print("\nCopy to overleaf")
        print("=== Stock \t Annual return \t St dev \t Sharpe r ===")
        for i in range(len(tickers)):
            print(f"{tickers[i]}: &\t {round(mean_annual_returns[i]*100, 2)} &\t {round(sd_annual[i]*100, 2)} &\t {round(sharpe[i], 3)} \\")
        
        print("\n=== Stock \t Range \t Kurtosis \t Skewness ===")
        for i in range(len(tickers)):
            print(f"{tickers[i]}: \t {round(ranges[i]*100, 2)}% \t {round(kurtosis[i], 3)} \t {round(skewness[i], 3)}")
     
        
    return mean_annual_returns, annual_cov_matrix, sd_annual


def autocorrelation(tickers, time_series, selected_stocks):
    location = "pictures\\acf_pacf\\"
    
    #Plotting auto-correlation and partial auto-correlation of selected stocks then saving them to file
    for stock in tickers:
        fig1 = plot_acf(time_series[stock], lags = 30, title = f"Auto-correlation of {stock}")
        fig1.savefig(location + "acf_" + stock)   
        plt.close(fig1) 
        
        fig2 = plot_pacf(time_series[stock], lags = 30, title = f"Partial auto-correlation of {stock}", method='ywm')
        fig2.savefig(location + "pacf_" + stock)   
        plt.close(fig2)    
        
    
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
     