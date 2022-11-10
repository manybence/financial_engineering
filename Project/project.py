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
from scipy import stats, optimize
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from arch import arch_model
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

TICKERS = sorted(["GOOGL", "KO", "AAPL", "AMZN", "PFE", "AMT", "XOM", "JPM"])
START = "2016-01-01"
END = "2022-01-01"
INTERVAL = "1d"


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
    mean_annual_returns = (1 + mean_daily_returns)**252 - 1   

    print("\n=== Mean annual returns ===")       
    for etf, geomean_annual in zip(tickers, mean_annual_returns):
        print("{}: {}%".format(etf, round(geomean_annual*100, 2)))

    

    # Covariance matrix
    daily_cov_matrix = daily_returns.cov()
    annual_cov_matrix = daily_cov_matrix * 252
    print("=== Annual Covariance Matrix ===")
    print(annual_cov_matrix)

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
    
    
    #Histogram of returns to show distribution
    figure = plt.figure()
    #plt.hist(daily_returns, density = True, bins=30)
    ax = sns.histplot(daily_returns, kde = True, stat = "density", bins = 40)
    plt.title('Histogram of Weekly Returns', fontsize=30)
    plt.xlabel('Return', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    
    
    #Calculating Sharpe ratio for the stocks
    sharpe = mean_annual_returns / sd_annual
    print("\n=== Sharpe ratio of stocks ===")
    for i in range(len(tickers)):
        print(f"{tickers[i]}: {round(sharpe[i], 3)}")
        
    print("\nCopy to overleaf")
    print("=== Stock \t Annual return \t St dev \t Sharpe r ===")
    for i in range(len(tickers)):
        print(f"{tickers[i]}: &\t {round(mean_annual_returns[i]*100, 2)} &\t {round(sd_annual[i]*100, 2)} &\t {round(sharpe[i], 3)} \\")
        
        
        
    #Calculate range, skewness, kurtosis of returns
    ranges = [max(daily_returns[stock]) - min(daily_returns[stock]) for stock in tickers]
    kurtosis = [stats.kurtosis(daily_returns[stock]) for stock in tickers]
    skewness = [stats.skew(daily_returns[stock]) for stock in tickers]
    
    print("\n=== Stock \t Range \t Kurtosis \t Skewness ===")
    for i in range(len(tickers)):
        print(f"{tickers[i]}: \t {round(ranges[i]*100, 2)}% \t {round(kurtosis[i], 3)} \t {round(skewness[i], 3)}")
        
    return mean_annual_returns, annual_cov_matrix, sd_annual

def GMV(mus, stds, weights):
   
    idx = np.argmin(stds)
    min_std = stds[idx]
    min_mu = mus[idx]
    min_weight = weights[idx]
    
    return min_mu, min_std, min_weight

def compute_EF_no_bds(mu, Sigma, mu_p):
    def object_function(weights):
        return weights.T @ Sigma @ weights
    
    def constraint_sum_to_1(weights):
        return 1 - np.sum(weights)
    
    def constraint_return_mup(weights):
        return mu_p - np.sum(weights*mu)
    
    cons = ({'type':'eq','fun':constraint_sum_to_1},
            {'type':'eq','fun':constraint_return_mup})
    
    # initialize weights, uniform distribution
    w0 = np.ones(len(mu))/len(mu)
    
    # minimize variance
    sol = optimize.minimize(object_function, w0, constraints = cons)
    
    # extract weights from solution
    w_p = np.array(sol.x)
    std_p = np.sqrt(w_p.T @ Sigma @ w_p)

    if sol.success:
        return w_p, mu_p, std_p
    else:
        return None

def compute_EF(mu, Sigma, mu_p, low_bd = 0.0, up_bd = 1.0):
    
    def object_function(weights):
        return weights.T @ Sigma @ weights
    
    def constraint_sum_to_1(weights):
        return 1 - np.sum(weights)
    
    def constraint_return_mup(weights):
        return mu_p - np.sum(weights*mu)
    
    cons = ({'type':'eq','fun':constraint_sum_to_1},
            {'type':'eq','fun':constraint_return_mup})
    
    # initialize weights, uniform distribution
    w0 = np.ones(len(mu))/len(mu)
    
    # By default
    # lower bound = 0: no short selling
    # upper bound = 1: no lending
    bds = ((low_bd, up_bd) ,)*len(mu)
    
    # minimize variance
    sol = optimize.minimize(object_function, w0, bounds = bds, constraints = cons)
    
    # extract weights from solution
    w_p = np.array(sol.x)
    std_p = np.sqrt(w_p.T @ Sigma @ w_p)

    if sol.success:
        return w_p, mu_p, std_p
    else:
        return None
    
def GMV_shorting(mu_assets, cov_mat_assets):
    mu_ps = np.arange(start = 0.0, stop = 0.8, step = 0.001)
    returns = np.array([])
    stds = np.array([])
    weights = []
    for mu_p in mu_ps:
        solution = compute_EF_no_bds(mu_assets, cov_mat_assets, mu_p)
        if solution is not None:
            weight, mu, std = solution
            weights.append(weight)
            returns = np.append(returns, mu)
            stds = np.append(stds, std)
    mu_gmv, std_gmv, w_gmv = GMV(returns, stds, weights)
    
    return stds, returns, std_gmv, mu_gmv, w_gmv

def GMV_no_shorting(mu_assets, cov_mat_assets):
    # mu_assets: mean annual return of the N assets
    # cov_mat_assets: covariance matrix of the N assets
    mu_ps = np.arange(start = 0.0, stop = 0.8, step = 0.001) # evaluated returns
    
    weights_no_short = []
    ret_no_short = []
    stds_no_short = []
    
    weights_max20 = []
    ret_max20 = []
    stds_max20 = []
    
    weights_min8 = []
    ret_min8 = []
    stds_min8 = []
        
    for mu_p in mu_ps:
        EF_no_short = compute_EF(mu_assets, cov_mat_assets, mu_p)
        EF_max20 = compute_EF(mu_assets, cov_mat_assets, mu_p, up_bd = 0.2)
        EF_min8 = compute_EF(mu_assets, cov_mat_assets, mu_p, low_bd = 0.08)
        
        if EF_no_short is not None:
            w, mu, std = EF_no_short
            weights_no_short.append(w)
            ret_no_short.append(mu)
            stds_no_short.append(std)
            
        if EF_max20 is not None:
            w, mu, std = EF_max20
            weights_max20.append(w)
            ret_max20.append(mu)
            stds_max20.append(std)
            
        if EF_min8 is not None:
            w, mu, std = EF_min8
            weights_min8.append(w)
            ret_min8.append(mu)
            stds_min8.append(std)
    
    no_short = pd.DataFrame((np.stack((ret_no_short, stds_no_short), axis = 0)).T,
                        columns = ['Return','STD'])
    
    max20 = pd.DataFrame((np.stack((ret_max20, stds_max20), axis = 0)).T,
                        columns = ['Return','STD'])
    
    min8 = pd.DataFrame((np.stack((ret_min8, stds_min8), axis = 0)).T,
                        columns = ['Return','STD'])

    mu_mv_no_short, std_mv_no_short, w_mv_no_short = GMV(ret_no_short, stds_no_short, weights_no_short)
    mu_mv_max20, std_mv_max20, w_mv_max20 = GMV(ret_max20, stds_max20, weights_max20)
    mu_mv_min8, std_mv_min8, w_mv_min8 = GMV(ret_min8, stds_min8, weights_min8)
    
    mu_mv = [mu_mv_no_short, mu_mv_max20, mu_mv_min8]
    std_mv = [std_mv_no_short, std_mv_max20, std_mv_min8]
    w_mv =  [w_mv_no_short, w_mv_max20, w_mv_min8]
    
    return no_short, max20, min8, std_mv, mu_mv, w_mv

def equal_weights(N, mu_assets, cov_mat_assets):
    # N is number of assets
    w_eq = (1/N)*np.ones(N)
    mu_eq = w_eq @ mu_assets
    std_eq = np.sqrt(w_eq @ cov_mat_assets @ w_eq.T).squeeze()
    
    return std_eq, mu_eq, w_eq

def tangent(mu_assets, RFR, cov_mat_assets):
    cov_mat_inv = np.linalg.inv(cov_mat_assets)
    mu_e = mu_assets - RFR
    mu_e_tan = (mu_e.T @ cov_mat_inv @ mu_e) / (np.ones_like(mu_e).T @ cov_mat_inv @ mu_e)
    mu_tan = mu_e_tan + RFR
    std_tan = np.sqrt(mu_e.T @ cov_mat_inv @ mu_e) / (np.ones_like(mu_e).T @ cov_mat_inv @ mu_e)
    
    slope = mu_e_tan / std_tan
    
    stds = np.linspace(0, 0.5)
    returns = (stds*slope + RFR).squeeze()
    mu_e = mu_assets - RFR # RFR is risk-free rate of returns
    
    return stds, returns, std_tan, mu_tan

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


def main(test=False):
    
    stock_data = yf.download(
    tickers = TICKERS,
    start = START,
    end = END,
    interval = INTERVAL
    ).dropna()
    stock_data.tail()
    
    """------------------------------ DATA PRESENTATION ----------------------------------------"""

    #Calculate daily returns
    daily_returns = stock_data["Adj Close"].pct_change().dropna()
    
    #Plotting historical prices, returns
    historical_data(TICKERS, stock_data, daily_returns)
    
    #Calculate annualised mean and covariance
    annual_return, annual_cov_mat, annual_std = return_analysis(TICKERS, daily_returns)

    
    
    """------------------------------ DATA ANALYSIS ----------------------------------------"""
    
    autocorrelation(TICKERS, daily_returns)
    
    build_model(daily_returns)



    """------------------------------ PORTFOLIO THEORY -------------------------------------"""

    stds_shorting, ret_shorting, std_gmv_shorting, mu_gmv_shorting, w_gmv_shorting = GMV_shorting(annual_return, annual_cov_mat)
    std_eq, mu_eq, weights_eq = equal_weights(8, annual_return, annual_cov_mat)
    stds_cm, returns_cm, std_tan, mu_tan = tangent(annual_return, 0.02, annual_cov_mat)
    no_short, max20, min8, std_mv, mu_mv, w_mv = GMV_no_shorting(annual_return, annual_cov_mat)
    
    print(['Shorting', 'Equal weights', 'Tangent', 'No shorting', 'Max20', 'Min8', ])
    print('Return')
    print(mu_gmv_shorting, mu_eq, mu_tan, mu_mv)
    print('STD')
    print(std_gmv_shorting, std_eq, std_tan, std_mv)
    print('Weights')
    
    
    plt.figure(figsize=(15,10))
    plt.plot(annual_std, annual_return, 'kx', label = 'Assets', ms = 5)
    plt.plot(stds_shorting, ret_shorting, label='EF with shorting', lw = 2, alpha = 0.75)
    plt.plot(std_gmv_shorting, mu_gmv_shorting, 'o', label = 'GMV-Shorting', ms = 2, mew = 2)
    plt.plot(std_eq, mu_eq, 'o', label = 'Equal weights', ms = 2, mew = 1)
    plt.plot(stds_cm, returns_cm, '--', label="Capital Market Line", lw = 2, alpha = 0.75)
    plt.plot(std_tan, mu_tan, 'o', label="Tangent Portfolio", ms = 2, mew = 2)
    plt.plot(no_short['STD'], no_short['Return'], label='EF No shorting', lw = 2, alpha = 0.75)
    plt.plot(std_mv[0], mu_mv[0], 'o', label = 'MV-No shorting', ms = 2, mew = 2)
    plt.plot(max20['STD'], max20['Return'], label='EF No shorting - Max 20%', lw = 2, alpha = 0.75)
    plt.plot(std_mv[1], mu_mv[1], 'o', label = 'MV-Max 20%', ms = 2, mew = 2)
    plt.plot(min8['STD'], min8['Return'], label='EF No shorting - Min 8%', lw = 2, alpha = 0.75)
    plt.plot(std_mv[2], mu_mv[2], 'o', label = 'MV-Min 8%', ms = 2, mew = 2)

    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Return (Annualized Geometric Mean)')
    plt.title('Efficient Frontier (No Lending and No Shorting)')
    plt.xlim(left = 0)
    plt.ylim(bottom = 0)
    plt.legend()
    plt.grid()
    plt.show()
    
    


if __name__ == "__main__":
    main()
    


