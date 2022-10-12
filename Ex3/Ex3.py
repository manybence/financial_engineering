# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:47:03 2022

@author: Bence Mány

Introduction to Financial Engineering

Ex 3:  Time series analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
from scipy.stats import shapiro
import seaborn as sns
import statsmodels.graphics.gofplots as sm


df = pd.read_excel("WILL5000.xls")
plt.plot(df["Date"], df["Index"])

mean = stat.mean(df["Index"])
med = stat.median(df["Index"])
dev = stat.stdev(df["Index"])
print(mean, med, dev)
        


#Plotting normal probability
fig, ax = plt.subplots(1, 2, figsize=(12, 7))
sns.histplot(df["Index"],kde=True, color ='blue',ax=ax[0])
sm.ProbPlot(df["Index"]).qqplot(line='s', ax=ax[1])
plt.show()

#Executing Shapiro test
print(shapiro(df["Index"]))

