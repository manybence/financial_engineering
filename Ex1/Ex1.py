# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:13:00 2022

@author: Bence Mány

Introduction to Financial Engineering

1. Exercise: Financial returns
"""

import yfinance as yf
import matplotlib.pyplot as plt
import csv
import statistics
import math
from datetime import datetime
from scipy.stats.mstats import gmean
import numpy as np
  

def calculate_returns(prices):
    return (prices / np.roll(prices, 1) - 1)[1:]

def calculate_log_returns(returns):
    return [(np.log(i + 1)) for i in returns]


def geometric_average(returns):
    average = 1
    for i in returns:
        average *= (1 + i)
    return average**(1/len(returns)) - 1


# Downloading stock data
start = datetime.strptime("2005-01-01", "%Y-%m-%d")
end = datetime.strptime("2022-08-30", "%Y-%m-%d")
xom = yf.download('XOM', start=start, end=end, interval='1wk')
xom = xom.dropna(axis = 0)  #Dropping NaN values (corporate events)
#print(xom.tail(10))

        
#Displaying closing and adjusted closing prices
fig1 = plt.figure(1)
plt.plot(xom["Close"], label = "Closing prices")
plt.plot(xom["Adj Close"], label = "Adjusted closing prices")
plt.xlabel('Year', fontsize=20)
plt.ylabel('USD $', fontsize=20)
plt.grid()
plt.legend()
plt.title("XOM closing prices")

      

#Calculating weekly returns
xom["Returns"] = calculate_returns(xom["Adj Close"])


#Calculating (geometric) average weekly return
average = geometric_average(xom["Returns"][1:])
print("The average weekly return is: ", round(average*100, 4), "%")


#Calculating standard deviation of weekly returns
deviation = statistics.stdev(xom["Returns"][1:])
print("The standard deviation of weekly returns is: ", round(deviation*100, 4), "%")


#Calculating weekly log returns
xom["Log returns"] = calculate_log_returns(xom["Returns"])

#Calculating average weekly log return    
average = sum(xom["Log returns"][1:])/len(xom["Log returns"][1:])
print("The average weekly log return is: ", round(average*100, 4), "%")


#Displaying weekly returns and weekly log returns
fig2 = plt.figure(2)
plt.plot(xom["Returns"], label = "Weekly returns")
plt.plot(xom["Log returns"], label = "Weekly log returns")
plt.title('Returns and log(Returns) Comparison', fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.grid()
plt.legend()
plt.show()



