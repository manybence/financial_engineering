# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:38:45 2022

@author: Bence Mány

Introduction to Financial Engineering

Functions for all the calculations
"""

import yfinance as yf
import matplotlib.pyplot as plt
import csv
import statistics
import math
from datetime import datetime
from scipy.stats.mstats import gmean
import numpy as np




def geometric_average(returns):
    average = 1
    for i in returns:
        average *= (1 + i)
    return average**(1/len(returns)) - 1


def differentiate(prices):
    return (prices / np.roll(prices, 1) - 1)[1:]


def calculate_log_returns(returns):
    return [(np.log(i + 1)) for i in returns]


def annual_return(ticker):
    returns = []
    last = 0
    for i in range(1, len(ticker["Date"])):
        if i % 252 == 0:
            returns.append((ticker[i] - ticker[i-252]) / ticker[i-252])
    return returns