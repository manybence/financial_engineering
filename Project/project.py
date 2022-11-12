# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:11:43 2022

@author: Bence Mány

Introduction to Financial Engineering

Portfolio Management Project
"""

import yfinance as yf
import warnings
import analysis
import portfolio
import model

warnings.filterwarnings("ignore")

TICKERS = sorted(["GOOGL", "KO", "AAPL", "AMZN", "PFE", "AMT", "XOM", "JPM"])
START = "2016-01-01"
END = "2022-01-01"
INTERVAL = "1d"

    

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
    #analysis.historical_data(TICKERS, stock_data, daily_returns)
    
    #Calculate annualised mean and covariance
    annual_return, annual_cov_mat, annual_std = analysis.return_analysis(TICKERS, daily_returns, display = False)

    
    
    """------------------------------ DATA ANALYSIS ----------------------------------------"""
  
    selected_stocks = ["GOOGL", "PFE"]
    
    #analysis.autocorrelation(TICKERS, daily_returns, selected_stocks)
    
    #build_model(daily_returns)



    """------------------------------ PORTFOLIO THEORY -------------------------------------"""

    #portfolio.calculate_portfolios(annual_return, annual_cov_mat, annual_std, TICKERS)
    
    """------------------------------ PORTFOLIO PERFORMANCE -------------------------------------""" 





if __name__ == "__main__":
    main()
    


