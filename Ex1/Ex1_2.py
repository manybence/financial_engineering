# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 20:38:23 2022

@author: Bence Mány

Introduction to Financial Engineering

1. Exercise/2: ETFs
"""
import yfinance as yf
import matplotlib.pyplot as plt
import csv
import statistics
import math
from datetime import datetime
from scipy.stats.mstats import gmean
import numpy as np
  

# Downloading stock data
start = datetime.strptime("2005-01-01", "%Y-%m-%d")
end = datetime.today()
stocks = yf.download(["SPY", "XLF", "EEM"], start=start, end=end, interval='1d')
stocks = stocks.dropna(axis = 0)  #Dropping NaN values (corporate events)


plt.plot(stocks['Adj Close']['SPY'], label = "SPY")
plt.plot(stocks['Adj Close']['XLF'], label = "XLF")
plt.plot(stocks['Adj Close']['EEM'], label = "EEM")
plt.xlabel('Year', fontsize=20)
plt.ylabel('USD $', fontsize=20)
plt.grid()
plt.legend()
plt.show()



