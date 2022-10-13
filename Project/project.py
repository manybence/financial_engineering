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

def download_data(name):
    start = datetime.strptime("2016-01-01", "%Y-%m-%d")
    end = datetime.strptime("2022-01-01", "%Y-%m-%d")
    stock = yf.download(name, start=start, end=end, interval='1d')
    stock = stock.dropna(axis = 0)  #Dropping NaN values (corporate events)
    
    # fig1 = plt.figure(1)
    # plt.plot(stock["Adj Close"], label = name)
    # plt.xlabel('Year', fontsize=20)
    # plt.ylabel('USD $', fontsize=20)
    # plt.grid()
    # plt.legend()
    # plt.title("Closing prices")
    
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

#Download stock data
companies = ["GOOGL", "KO", "AAPL", "TSLA", "PFE", "AMT", "XOM", "JPM"]
stocks = []
for i in companies:
    stocks.append(download_data(i))

for i in stocks:
    i["Returns"] = i["Adj Close"].diff()
    


#Test if the given stock returns have high correlation or from same sector
check_correlation(stocks)
check_sectors(companies)



