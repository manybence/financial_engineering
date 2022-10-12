# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:13:17 2022

@author: Bence Mány

Introduction to Financial Engineering

7. Exercise: Portfolio construction
"""

import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import sys
sys.path.append('../../')
from Exercises import finance_functions as ff


def download_data(name):
    start = datetime.strptime("2011-01-01", "%Y-%m-%d")
    end = datetime.strptime("2021-01-01", "%Y-%m-%d")
    stock = yf.download(name, start=start, end=end, interval='1wk')
    stock = stock.dropna(axis = 0)  #Dropping NaN values (corporate events)
    
    fig1 = plt.figure(1)
    plt.plot(stock["Adj Close"], label = name)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('USD $', fontsize=20)
    plt.grid()
    plt.legend()
    plt.title("Closing prices")
    
    return stock



# Downloading stock data
mcd = download_data("MCD")
ko = download_data("KO")
msft = download_data("MSFT")


#Calculating weekly returns
mcd["Returns"] = mcd["Adj Close"].diff()
ko["Returns"] = ko["Adj Close"].diff()
msft["Returns"] = msft["Adj Close"].diff()

fig1 = plt.figure(2)
plt.plot(mcd["Returns"], label = "MCD")
plt.plot(ko["Returns"], label = "KO")
plt.plot(msft["Returns"], label = "MSFT")
plt.xlabel('Year', fontsize=20)
plt.ylabel('USD $', fontsize=20)
plt.grid()
plt.legend()
plt.title("Weekly returns")


#Calculating annualised mean and covariance of returns
annual_mcd = ff.annual_return(mcd["Adj Close"])
mean_mcd = ff.geometric_average(annual_mcd)
print(f"MCD annual return: {round(mean_mcd*100, 3)} %")
    
annual_ko = ff.annual_return(ko["Adj Close"])
mean_ko = ff.geometric_average(annual_ko)
print(f"KO annual return: {round(mean_ko*100, 3)} %")

annual_msft = ff.annual_return(msft["Adj Close"])
mean_msft = ff.geometric_average(annual_msft)
print(f"MSFT annual return: {round(mean_msft*100, 3)} %")

print("Covariance matrix of annualised returns: \n", np.cov([annual_mcd, annual_ko, annual_msft]))
print("\n\n")

#TODO: Compare portfolios
#TODO: Which portfolio has the maximum mean?
#TODO: Which portfolio has the minimum deviation?
#TODO: Which portfolio has highest mean-deviation ratio?


