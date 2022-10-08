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
from statsmodels.tsa.arima.model import ARIMA

#Uncomment the parts you'd like to plot


# Exercise 1: Plotting Autocorrelation and Partial Autocorrelation

df = pd.read_excel("WILL5000PRFC.xls")

# plt.plot(df["Date"], df["Price"])
# plot_acf(df["Price"], lags = 50)
# plot_pacf(df["Price"], lags = 50)




# Exercise 2: Calculating ACF and PACF with differenced data
    
# plt.plot(df["Date"], df["Price"].diff())
# plot_acf(df["Price"].diff()[1:], lags = 50)
# plot_pacf(df["Price"].diff()[1:], lags = 50)
    




# Exercise 3: AIC and BIC on differential data

model = ARIMA(df["Price"].diff()[1:], order = (1,1,1))  #AR = 1, diff = 1, MA=1
results = model.fit()
print("AIC: ", round(results.aic, 3), "\nBIC: ", round(results.bic, 3))
#results.plot_predict(dynamic = False)

# 5. Forecast based on model
y_pred = results.forecast(3)
print("Forecast: ", y_pred)

    
