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


TICKERS = ["GOOGL", "KO", "AAPL", "AMZN", "PFE", "AMT", "XOM", "JPM"]
START = "2016-01-01"
END = "2022-01-01"
INTERVAL = "1d"

def download_data(name, plotting=False, interval='1d'):
    """
    Downloading and plotting stock data

    Parameters
    ----------
    name : str
        Code of the selected company.
    plotting : Bool
        Plots the historical Adjusted Closing Price if True is selected. The default is False.
    interval : str
        Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

    Returns
    -------
    stock : Ticker object
        Contains all the important data related to the selected stock

    """
    start = datetime.strptime("2016-01-01", "%Y-%m-%d")
    end = datetime.strptime("2022-01-01", "%Y-%m-%d")
    stock = yf.download(name, start=start, end=end, interval=interval)
    stock = stock.dropna(axis = 0)  #Dropping NaN values (corporate events)
    stock = stock.reset_index()     #Makes the "Date" column accessible
        
    
    if plotting:
        fig1 = plt.figure(1)
        plt.plot(stock["Date"], stock["Adj Close"], label = name)
        plt.xlabel('Year', fontsize=20)
        plt.ylabel('USD $', fontsize=20)
        plt.grid()
        plt.legend(fontsize = 15)
        plt.title("Historical prices", fontsize = 30)
    
    return stock

def check_correlation(stocks):
    print("Test started: Correlation checking")
    error = False
    
    for i in range(len(stocks)):
        for j in range(i + 1, len(stocks)):
            r = np.corrcoef(stocks[i]["Returns"][1:], stocks[j]["Returns"][1:])[1][0]
            if (abs(r) > 0.7):
                print(f"Too high correlation between {i, j}! r = {round(r, 3)}")
                error = True 
    
    if not error:
        print("Test passed. No strong correlation found.")
    return

def check_sectors(companies):
    print("Test started: Sector checking")
    sectors = []
    error = False
    
    for i in companies:
        tickerdata = yf.Ticker(i)
        sectors.append(tickerdata.info['sector'])
        print(f"{i} - {tickerdata.info['sector']}")
    
    for i in range(len(sectors)):
        for j in range(i + 1, len(sectors)):
            if sectors[i] == sectors[j]:
                print(f"Error, same sector! {i, j}, ({sectors[i]})")
                error = True
                
    if not error:
        print("Test passed. All companies are from different sectors.")
    return

def annual_return(ticker):
    returns = []
    for i in range(1, len(ticker["Date"])):
            if (i + 1) % 252 == 0:
                returns.append((ticker["Adj Close"][i] - ticker["Adj Close"][i-251]) / ticker["Adj Close"][i-251])    
    return returns

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
    mean_annual_returns = (1 + mean_daily_returns)**len(daily_returns) - 1

    #TODO: Google 632% ?? looks wrong
    print("\n=== Mean annual returns ===")
    for i in range(len(tickers)):
        print(f"{tickers[i]}: {round(mean_annual_returns[i] * 100, 2)}%")

    
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
    
    return

def main(test=False):
    
    #Download and plot stock data
    stock_data = yf.download(
    tickers = TICKERS,
    start = START,
    end = END,
    interval = INTERVAL
    ).dropna()
    stock_data.tail()
    

    #Calculate daily returns
    daily_returns = stock_data["Adj Close"].pct_change().dropna()
    daily_returns.tail()
    
    #Plotting historical prices, returns
    historical_data(TICKERS, stock_data, daily_returns)
    
    #Calculate annualised mean and covariance
    return_analysis(TICKERS, daily_returns)
    
    
    #Calculate standard deviation of returns
    #TODO
    
    
    #Calculate correlation of returns
    #TODO
    
    #Distribution of returns
    #TODO
    
    #Calculate range, skewness, kurtosis of returns
    #TODO
    
    #Calculate Sharpe ratio
    #TODO
    


if __name__ == "__main__":
    main()
    


