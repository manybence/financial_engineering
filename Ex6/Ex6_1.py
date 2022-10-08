# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:47:03 2022

@author: Bence Mány

Introduction to Financial Engineering

Ex 5:  Time series analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima

#Uncomment the parts you'd like to plot

df = pd.read_excel("WILL5000PRFC.xls")






# Problem 1: Augmented Dickey-Fuller test -> Is it stationary?

plt.plot(df["Date"], df["Price"])

print("Dickey-Fuller test p-value: ", adfuller(df["Price"])[1])
if (adfuller(df["Price"])[1] < 0.05):
    print("Time series is stationary")
else:
    print("Time series is non-stationary")
    
    
    
    
    
    
    
#Problem 2: Differencing the data, seasonality

plt.plot(df["Date"], df["Price"].diff())

print("Dickey-Fuller test p-value: ", adfuller(df["Price"].diff()[1:])[1])
if (adfuller(df["Price"].diff()[1:])[1] < 0.05):
    print("Time series is stationary")
else:
    print("Time series is non-stationary")


#Seasonal decomposition
result=seasonal_decompose(df['Price'], model='multiplicable', period=12)
result.plot()







#Problem 3: Auto-ARIMA forecast

model = auto_arima(df['Price'].diff()[1:], start_p=0, start_q=0)
print(model.summary())
model.plot_diagnostics()




    
