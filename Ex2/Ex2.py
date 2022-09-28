# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:22:37 2022

@author: Bence Mány

Introduction to Financial Engineering

2. Exercise: Danish bonds
"""

import csv
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt


def calculate_cashflow (coupon, maturity):
    cashflow = []
    years = maturity - pd.to_datetime("today")
    years = int(years.days / 365)
    for i in range(years + 1):
        cashflow.append(coupon)
    cashflow[-1] += 100
    return cashflow
        
def calculate_ytm (price, maturity, cashflow):
    days = (maturity - pd.to_datetime("today")).days
    for r in range(100):
        npv = 0
        for i in range(len(cashflow)):
            npv += cashflow[i] / (1 + r / 100)**((days%365) / 365 + i)
        if round(price - npv) == 0:
            return r / 100
    

# Import bonds data        
bonds = pd.read_csv("bonds.csv", delimiter = ";")      
bonds["Maturity"] = pd.to_datetime(bonds["Maturity"])



# Cash flow calculation
bonds["CF"] = [[], [], [], [], []]

for i in range(len(bonds["Maturity"])):
    bonds["CF"][i].append(calculate_cashflow(bonds["Coupon"][i], bonds["Maturity"][i]))
    
print(bonds["CF"])
    
# Yield to maturity calculation
bonds["YtM"] = 0
bonds["YtM"][0] = calculate_ytm(bonds["Price"][0], bonds["Maturity"][0], bonds["CF"][0][0])
    
print(bonds)








