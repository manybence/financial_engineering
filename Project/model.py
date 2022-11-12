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

def build_model(daily_returns):
    
    stock1 = "GOOGL"
    stock2 = "PFE"
    


            
    model = ARIMA(daily_returns[stock1], order = (1,1,0))  #AR = 1, diff = 1, MA=0
    results = model.fit()
    print("AIC: ", round(results.aic, 3), "\nBIC: ", round(results.bic, 3))
    
    
    #TODO: Estimation for GARCH model
    #model_ARMA_train_residuals = daily_returns[stock1] - model.predict()
    #model_GARCH = arch_model(model_ARMA_train_residuals, p = 1, q = 1, mean = 'zero', dist = 'normal')
    #model_GARCH = model_GARCH.fit()
    
    return