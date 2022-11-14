# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:18:01 2022

@author: Bence Mány

Introduction to Financial Engineering

Functions for building models
"""

from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from arch import arch_model
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import numpy as np
import pandas as pd


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


def calc_ADF(daily_returns):
    """Using The Augmented Dickey-Fuller tests"""
    p_value = adfuller(daily_returns)[1]
    print(f"The p-value from ADF-test for the log-transform: {p_value}")
    
def recomended_model(daily_returns):
    stock1 = "GOOGL"
    stock2 = "PFE"
    print("For GOOGLE: \n")
    model_stock1 = auto_arima(daily_returns[stock1],start_p=0,start_q=0,seasonal=False,trace=True)
    print("\nFor PFE: \n")
    model_stock2 = auto_arima(daily_returns[stock2],start_p=0,start_q=0,seasonal=False,trace=True)

def model_auto_arima(daily_returns):
    """Building a model recomended by auto_arima"""
    stock1 = "GOOGL"
    stock2 = "PFE"
    #model = auto_arima(daily_returns,start_p=0,start_q=0,seasonal=False,trace=True)
    model_GOOGL = ARIMA(daily_returns[stock1], order = (1,0,0))
    model1 = model_GOOGL.fit()
    model_PFE = ARIMA(daily_returns[stock2], order = (2,0,2))
    model2 = model_PFE.fit()
    
    
    plt.figure(figsize=(17,10))
    plt.plot(model1.predict(), color = "orange")
    plt.plot(model2.predict(), color = "red")
    plt.show()
       
def model_choice(daily_returns):
    """This model is the preffered one by testing different ARIMA models"""
    stock1 = "GOOGL"
    stock2 = "PFE"
    #model = auto_arima(daily_returns,start_p=0,start_q=0,seasonal=False,trace=True)
    model_GOOGL = ARIMA(daily_returns[stock1], order = (1,1,0))
    model1 = model_GOOGL.fit()
    model_PFE = ARIMA(daily_returns[stock2], order = (2,1,0))
    model2 = model_PFE.fit()
    plt.figure(figsize=(17,10))
    plt.plot(model1.predict(), color = "orange")
    plt.plot(model2.predict(), color = "red")
    plt.show()
    
def work_around_for_arma_garch(daily_returns, stock):
    """This calculetes predicted values by using ARIMA, residuals of ARIMA and GARCH on residuals"""
    
    #Making a training set
    train_lein = int(daily_returns[stock]*0.66) #training set should be around 66%-70%
    train = daily_returns[stock].iloc[:train_lein].values
    test = daily_returns[stock].iloc[train_lein:].values
    
    #Create an ARMA model based on training data
    model_ARMA = ARIMA(train, order = (1,0,0), trend = [0,0,0,0])
    model_ARMA = model_ARMA.fit()
    
    #Computing the residuals
    model_ARMA_train_residuals = train - model_ARMA.predict()
    
    #Making a GARCH model fit the residuals
    model_GARCH = arch_model(model_ARMA_train_residuals, p = 1, q = 1, mean = 'zero', dist = 'normal')
    model_GARCH = model_GARCH.fit()
    
    #Using GARCH to predict the residuals
    garch_forecast = model_GARCH.forecast(horizon=1)
    
    #Use ARIMA to predict my
    predicted_mu = model_ARMA.predict()

    #Use GARCH toe predict epsilon (white noise)
    predicted_et = garch_forecast.mean['h.1'].iloc[-1]
    
    #Combine both models' output: yt = mu + et (the predicted y-hat)
    prediction = predicted_mu + predicted_et
     
def grid_search_model(daily_returns):
    
    AR = [0, 1, 2, 3, 4]
    MA = [0, 1, 2, 3, 4]
    I = [1]
    grid = []

    values = []
    for i in range(5):
       for j in range(5):
           model = ARIMA(daily_returns.dropna(), order = (i,1,j))
           results = model.fit()
           values.append(round(results.aic, 3))
           for k in range(len(I)):
                grid.append((AR[i], I[k], MA[j]))   
    models = []

    AIC = []
    BIC = []
    LL = []

    for i in range(len(grid)):

        model = ARIMA(daily_returns.dropna(), order = grid[i])
        models.append(model.fit())

        print('ORDER: {}'.format(grid[i]))
        AIC.append(models[i].aic)
        print('AIC: {}'.format(models[i].aic))
        BIC.append(models[i].bic)
        print('BIC: {}'.format(models[i].bic))
        LL.append(models[i].llf)
        print('LL: {}'.format(models[i].llf))   


    #print(f"The optimal model's fit is: {min([abs(i) for i in values])}")
    print("The best model according to Akaike Information Criterion (AIC) is at index: {}".format(AIC.index(min(AIC))))
    print("The best model according to Bayesian Information Criterion (BIC) is at index: {}".format(BIC.index(min(BIC))))
    print("The best model according to Log Likelihood is at index: {}".format(LL.index(min(LL))))

    #Created the model and show its value
    Candidate_model = models[AIC.index(min(AIC))]
    print(Candidate_model.summary())

    print("The best model according to Akaike Information Criterion (AIC) is at index: {}".format(AIC.index(min(AIC))))
    print("The best model according to Bayesian Information Criterion (BIC) is at index: {}".format(BIC.index(min(BIC))))
    print("The best model according to Log Likelihood is at index: {}".format(LL.index(min(LL))))
    
    Candidate_model = models[AIC.index(min(AIC))]
    print(Candidate_model.summary())

def show_observed_predict(daily_returns, stock , p,d,q):
    
    #Creates the model
    model = ARIMA(daily_returns[stock].dropna(), order=(p,d,q))
    model_fit = model.fit()
    
    #show figure
    plt.rc("figure", figsize=(15,6))
    plt.title(stock)
    plt.plot(daily_returns[stock].dropna(), color = 'blue', label = "Observed")
    plt.plot(model_fit.predict(), color = 'red', label = "Predicted")
    plt.legend()
    plt.show()

def forecast_model(TICKER, p, d, q):
    #Getting the training set and the observed set
    start = datetime.datetime(2016,1,1)
    end = datetime.datetime(2021,12,1)
    end2 = datetime.datetime(2022,12,1)
    stock_observed = yf.download(TICKER , start=end, end=end2, interval="1d")
    stock_training = yf.download(TICKER, start=start, end=end, interval="1d")
    
    #Calculating simple-return for the training set
    stock_training["Returns"] = stock_training['Adj Close'].dropna().pct_change()

    #Calculating the simple-return from the beginning of 2022 to this day
    stock_observed["Returns"] = stock_observed['Adj Close'].dropna().pct_change()
    
    #Create the model 
    model = ARIMA(stock_training["Returns"].dropna(), order=(p,d,q))
    model_fit = model.fit()
    
    #Forcasting 12 step
    forecast = np.array(model_fit.forecast(12).values)
    
    #Adding the date
    DATE = np.array([datetime.datetime(2022,i,1) for i in [1,2,3,4,5,6,7,8,9,10,11,12]])
    
    #Adjusting date
    forecast_df = pd.DataFrame(data = {'DATE': DATE, "Forecast": forecast})
    start = 240
    end = stock_observed.shape[0]
    
    #Creating the conf interval
    conf_ins = model_fit.get_forecast(12).summary_frame()
    
    #Calculating the confidence interval
    plt.figure(figsize=(9,3), dpi = 200)
    plt.title(TICKER)
    plt.plot(stock_observed.index, stock_observed["Returns"], color = 'blue', label = "Observed")
    plt.plot(forecast_df['DATE'], forecast_df['Forecast'], color= 'red', label = "Forecasted")
    plt.plot(forecast_df['DATE'],conf_ins['mean_ci_lower'] , color = 'grey', label = "lower confidence")
    plt.plot(forecast_df['DATE'],conf_ins['mean_ci_upper'] , color = 'grey', label = "higher confidence")
    plt.legend(loc = "lower left")
    plt.show()
    
    