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


def annual_return(prices):
    returns = []
    for i in range(len(prices)):
        if i%52 == 0 and i > 0:
            returns.append((prices[i] - prices[i-52]) / prices[i-52])
    return returns