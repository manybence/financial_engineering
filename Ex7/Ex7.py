# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:13:17 2022

@author: Bence Mány

Introduction to Financial Engineering

7. Exercise: Portfolio construction

    1. Find optimal combination of 3 different stocks
    
    2. Calculating the Efficient Frontier for two assets. Effect of correlation between assets.
"""

import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import sys
sys.path.append('../../')
from Exercises import finance_functions as ff
from scipy.stats import gmean
import pandas as pd

def efficient_frontier(mean_annual_returns, annual_cov_matrix):
    mu = mean_annual_returns.reshape(-1,1)
    Sigma = annual_cov_matrix
    Sigma_inv = np.linalg.inv(Sigma)
    a = mu.T @ Sigma_inv @ mu
    b = mu.T @ Sigma_inv @ np.ones_like(mu)
    c = np.ones_like(mu).T @ Sigma_inv @ np.ones_like(mu)
    mu_gmv = b/c
    var_gmv = 1/c
    
    mus = np.linspace(np.min(mu) - np.ptp(mu), np.max(mu) + np.ptp(mu))
    sigmas = np.sqrt((c*mus**2 - 2*b*mus + a) / (a*c - b**2)).squeeze()
    
    # Plots
    plt.plot(sigmas, mus, label="Efficient Frontier")
    plt.legend()
    
    return(sigmas, mus)


exercise_1 = True
exercise_2 = False
exercise_3 = True

def main():

    if exercise_1:
    
        # Downloading stock data
        
        TICKERS = ["MCD","KO","MSFT"]
        START = "2011-01-01"
        END = "2021-01-01"
        INTERVAL = "1wk"
        
        
        stock_data = yf.download(
        tickers = TICKERS,
        start = START,
        end = END,
        interval = INTERVAL
        ).dropna()
        stock_data.tail()
        
        #Calculate weekly returns
        weekly_returns = stock_data["Adj Close"].pct_change().dropna()
        weekly_returns.tail()
        
        
        #Calculate annualised mean and covariance
        mean_weekly_returns = gmean(weekly_returns + 1) - 1
        mean_annual_returns = (1 + mean_weekly_returns)**52 - 1
        weekly_cov_matrix = weekly_returns.cov()
        annual_cov_matrix = weekly_cov_matrix * 52
        print("\n=== Mean weekly returns ===")
        print(pd.Series(mean_weekly_returns, weekly_returns.columns))
        print("\n=== Mean annual returns ===")
        print(pd.Series(mean_annual_returns, weekly_returns.columns))
        print("\n=== Annual Covariance Matrix ===")
        print(annual_cov_matrix)
        
        
        #Combine stocks with different weights
        portfolio_means = []
        portfolio_stds = []
        portfolio_weights = []
        for x in range(0, 10, 1):
            for y in range(0, 10 - x, 1):
                i = x/10
                j = y/10
                k = 1 - i - j
                w = np.array([[i, j, k]])
                portfolio_means.append((i * mean_annual_returns[0] + j * 
                mean_annual_returns[1] + k * mean_annual_returns[2]).squeeze())
                portfolio_stds.append(np.sqrt(w @ annual_cov_matrix @ w.T).squeeze())
                portfolio_weights.append(w)
        portfolio_means = np.array(portfolio_means)
        portfolio_stds = np.array(portfolio_stds)
        plt.scatter(portfolio_stds, portfolio_means)
        plt.xlabel("Annualised standard deviation")
        plt.ylabel("Annualised mean return")
    
    
    
        #Finding the portfolio with the maximum mean
        idx = np.argmax(portfolio_means)
        max_mean = portfolio_means[idx]
        max_weights = portfolio_weights[idx]
        print("\n=== Highest return portfolio ===")
        print(f"Return: {max_mean}")
        print(f"Weights: {max_weights}")
        
        
        #Finding the portfolio with the minimum deviation
        idx = np.argmin(portfolio_stds)
        min_std = portfolio_stds[idx]
        min_weights = portfolio_weights[idx]
        print("\n=== Lowest deviation portfolio ===")
        print(weekly_returns.columns.values)
        print(min_weights)
        print(min_std)
        
        
        #Finding the portfolio with highest mean-deviation (Sharpe) ratio
        idx = np.argmax(portfolio_means / portfolio_stds)
        max_sharpe = (portfolio_means / portfolio_stds)[idx]
        max_weights = portfolio_weights[idx]
        print("\n=== Highest Sharpe ratio portfolio ===")
        print(weekly_returns.columns.values)
        print(max_weights)
        print(max_sharpe)
    
    """----------------------------------------------------------------------------"""
    if exercise_2:
        
        rho = -0.5   #Correlation between assets
        
        mu = np.array([[0.1, 0.2]]).T
        Sigma = np.array([[0.1**2, 0.1*0.2*rho], [0.1*0.2*rho, 0.2**2]])
        Sigma_inv = np.linalg.inv(Sigma)
        
        # Solve GMV
        a = mu.T @ Sigma_inv @ mu
        b = mu.T @ Sigma_inv @ np.ones_like(mu)
        c = np.ones_like(mu).T @ Sigma_inv @ np.ones_like(mu)
        
        mu_gmv = b/c
        var_gmv = 1/c
        w_gmv = 1/c * Sigma_inv * np.ones_like(mu)
        
        # Calculate Efficient Frontier
        mus = np.linspace(np.min(mu) - np.ptp(mu), np.max(mu) + np.ptp(mu))
        sigmas = np.sqrt((c*mus**2 - 2*b*mus + a) / (a*c - b**2)).squeeze()
        
        # Plots
        plt.plot(sigmas, mus)
        plt.plot(0.1, 0.1, "x", ms=12, mew=6, label="Asset 1")
        plt.plot(0.2, 0.2, "x", ms=12, mew=6, label="Asset 2")
        plt.plot(var_gmv**0.5, mu_gmv, "x", ms=12, mew=6, label="GMV")
        plt.legend()
        
    
    """------------------------------------------------------------------------------"""
    if exercise_3:
        
        efficient_frontier(mean_annual_returns, annual_cov_matrix)
        plt.plot(portfolio_stds, portfolio_means, "x", label="Portfolios")
        plt.legend()
        
    
    return
    
    
if __name__ == "__main__":
    main()
