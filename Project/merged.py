# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:50:43 2022

@author: Bence Mány

Introduction to Financial Engineering

Portfolio management project
"""

import sys
sys.path.append('../../')
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import analysis
import model
import portfolio

warnings.filterwarnings("ignore")

TICKERS = sorted(["GOOGL", "KO", "AAPL", "AMZN", "PFE", "AMT", "XOM", "JPM"])
START = "2016-01-01"
END = "2022-01-01"
INTERVAL = "1d"


    
    
"""------------------------------ DATA PRESENTATION ----------------------------------------"""
stock_data = yf.download(tickers = TICKERS,start = START,end = END,interval = INTERVAL).dropna()
#Calculate daily returns
daily_returns = stock_data["Adj Close"].pct_change().dropna()

#Plotting historical prices, returns
analysis.historical_data(TICKERS, stock_data, daily_returns)

#Calculate annualised mean and covariance
annual_return, annual_cov_mat, annual_std = analysis.return_analysis(TICKERS, daily_returns)

"""------------------------------ DATA ANALYSIS ----------------------------------------"""
    
model.autocorrelation(TICKERS, daily_returns)

model.show_observed_predict(daily_returns, "GOOGL" , 1,1,0)

model.show_observed_predict(daily_returns, "PFE" , 2,1,0)

model.forecast_model("GOOGL", 1,1,0)

model.forecast_model("PFE" , 2,1,0)
    
    
"""------------------------------ PORTFOLIO THEORY -------------------------------------"""

#Historical data portfolios
stds_shorting, ret_shorting, std_gmv_shorting, mu_gmv_shorting, w_gmv_shorting = portfolio.GMV_shorting(annual_return, annual_cov_mat)
std_eq, mu_eq, w_eq = portfolio.equal_weights(8, annual_return, annual_cov_mat)
stds_cm, returns_cm, std_tan, mu_tan, w_tan = portfolio.tangent(annual_return, 0.02, annual_cov_mat)
no_short, max20, min8, std_mv, mu_mv, w_mv = portfolio.GMV_no_shorting(annual_return, annual_cov_mat)


print('=== Shorting ===', '\nReturns - STD - ', TICKERS)
print(mu_gmv_shorting, std_gmv_shorting, w_gmv_shorting)
print('=== No shorting ===', '\nReturns - STD - ', TICKERS)
print(mu_mv[0], std_mv[0], w_mv[0])
print('=== No shorting - Max 20% ===', '\nReturns - STD - ', TICKERS)
print(mu_mv[1], std_mv[1], w_mv[1])
print('=== No shorting - Min 8% ===', '\nReturns - STD - ', TICKERS)
print(mu_mv[2], std_mv[2], w_mv[2])
print('=== Equal weights ===', '\nReturns - STD - ', TICKERS)
print(std_eq, mu_eq, w_eq)
print('=== Tangent ===', '\nReturns - STD - ', TICKERS)
print(mu_tan, std_tan, w_tan)

plt.figure(figsize=(12,8))
plt.plot(annual_std, annual_return, 'kx', label = 'Assets', ms = 5)
plt.plot(stds_cm, returns_cm, 'b', label="Capital Market Line", lw = 1)
plt.plot(stds_shorting, ret_shorting, 'm--', label='EF with shorting', lw = 1)
plt.plot(no_short['STD'], no_short['Return'], 'k--', label='EF No shorting', lw = 1)
plt.plot(max20['STD'], max20['Return'], 'g--', label='EF No shorting - Max 20%', lw = 1)
plt.plot(min8['STD'], min8['Return'], 'r--', label='EF No shorting - Min 8%', lw = 1)
plt.plot(std_gmv_shorting, mu_gmv_shorting, 'mo', label = 'GMV-Shorting', ms = 4, mew = 2)
plt.plot(std_eq, mu_eq, 'co', label = 'Equal weights', ms = 4, mew = 2)
plt.plot(std_mv[0], mu_mv[0], 'ko', label = 'MV-No shorting', ms = 4, mew = 2)
plt.plot(std_tan, mu_tan, 'bo', label="Tangent Portfolio", ms = 4, mew = 2)
plt.plot(std_mv[2], mu_mv[2], 'ro', label = 'MV-Min 8%', ms = 4, mew = 2)
plt.plot(std_mv[1], mu_mv[1], 'go', label = 'MV-Max 20%', ms = 4, mew = 2)
plt.xlabel('Risk (STD)')
plt.ylabel('Return (annualized geometric mean)')
plt.title('Portfolios and Efficient frontiers')
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.legend()
plt.show()

"_______________________"
portfolios_return = portfolio.porfolio_returns_def(daily_returns, w_gmv_shorting,w_mv,w_tan,w_eq)

#Downloading new data (Jan 2022 - today)

TICKERS = sorted(["GOOGL", "KO", "AAPL", "AMZN", "PFE", "AMT", "XOM", "JPM"])
START = "2022-01-01"
END = "2022-11-12"
INTERVAL = "1d"

nw_stock_data = yf.download(tickers = TICKERS,start = START,end = END,interval = INTERVAL).dropna()


#Calculate daily returns
nw_daily_returns = nw_stock_data["Adj Close"].pct_change().dropna()

#Plotting historical prices, returns
analysis.historical_data(TICKERS, nw_stock_data, nw_daily_returns)

#Calculate annualised mean and covariance
nw_annual_return, nw_annual_cov_mat, nw_annual_std = analysis.return_analysis(TICKERS, nw_daily_returns)

#Calculate returns of portfolios
portfolios_return = portfolio.porfolio_returns_def(nw_daily_returns,w_gmv_shorting,w_mv,w_tan,w_eq)

#Evaluate portfolios based on recent data
nwport_annual_return, nwport_annual_cov_mat, nwport_annual_std = analysis.return_analysis(portfolios_return.columns,portfolios_return)