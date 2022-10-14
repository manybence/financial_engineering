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

def download_data(name, plotting=False, interval='1d'):
    """
    Downloading and plottting stock data

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

def main(test=False):
    
    #Download and plot stock data
    companies = ["GOOGL", "KO", "AAPL", "AMZN", "PFE", "AMT", "XOM", "JPM"]
    stocks = []
    for i in companies:
        stocks.append(download_data(i, True))
    
    #Test if the given stock returns have high correlation or from same sector
    if test:
        check_correlation(stocks)
        check_sectors(companies)
    
    
    #Calculate and plot daily returns
    fig2 = plt.figure(2)
    for i in range(len(stocks)):
        stocks[i]["Returns"] = stocks[i]["Adj Close"].diff()
        plt.plot(stocks[i]["Date"], stocks[i]["Returns"], label = companies[i])
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Return', fontsize=20)
    plt.grid()
    plt.legend(fontsize = 15)
    plt.title("Daily returns", fontsize = 30)
        
    
    
    #Calculating annual historical returns
    fig3 = plt.figure(3)
    for i in range(len(stocks)):
        annual = annual_return(stocks[i])
        plt.plot(range(2016, 2016 + len(annual)), annual, label = companies[i])
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Return', fontsize=20)
    plt.grid()
    plt.legend(fontsize = 15)
    plt.title("Annual returns", fontsize = 30)
    
    
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
    


