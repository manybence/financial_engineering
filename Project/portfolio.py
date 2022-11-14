# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:50:43 2022

@author: Bence Mány

Introduction to Financial Engineering

Functions for Portfolio assembly and evaluation
"""

import numpy as np
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
import analysis
import yfinance as yf


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
    
def GMV_shorting(mu_assets, cov_mat_assets, start = 0.0, stop = 0.8):
    mu_ps = np.arange(start = start, stop = stop, step = 0.001)
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

def GMV_no_shorting(mu_assets, cov_mat_assets, start = 0.0, stop = 0.8):
    # mu_assets: mean annual return of the N assets
    # cov_mat_assets: covariance matrix of the N assets
    mu_ps = np.arange(start = start, stop = stop, step = 0.001) # evaluated returns
    
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

def tangent(mu_assets, RFR, cov_mat_assets, up_bd=0.5):
    cov_mat_inv = np.linalg.inv(cov_mat_assets)
    mu_e = mu_assets - RFR
    mu_e_tan = (mu_e.T @ cov_mat_inv @ mu_e) / (np.ones_like(mu_e).T @ cov_mat_inv @ mu_e)
    mu_tan = mu_e_tan + RFR
    std_tan = np.sqrt(mu_e.T @ cov_mat_inv @ mu_e) / (np.ones_like(mu_e).T @ cov_mat_inv @ mu_e)
    w_tan = compute_EF_no_bds(mu_assets, cov_mat_assets, mu_tan)
    
    slope = mu_e_tan / std_tan
    
    stds = np.linspace(0, up_bd)
    returns = (stds*slope + RFR).squeeze()
    mu_e = mu_assets - RFR # RFR is risk-free rate of returns
    
    return stds, returns, std_tan, mu_tan, w_tan

def porfolio_returns_def(returns,w_gmv_shorting,w_mv,w_tan,w_eq):
    w_noshort = w_mv[0]
    w_max20 = w_mv[1]
    w_min8 = w_mv[2]
    w_tangent = w_tan[0]

    dates = []
    short = []
    no_short = []
    tangent = []
    equal = []
    min8 = []
    max20 = []



    for row in returns.iterrows():
        dates.append(row[0])
        short.append(row[1].dot(w_gmv_shorting))
        no_short.append(row[1].dot(w_noshort))
        tangent.append(row[1].dot(w_tangent))
        equal.append(row[1].dot(w_eq))
        min8.append(row[1].dot(w_min8))
        max20.append(row[1].dot(w_max20))


    portfolios_return = pd.DataFrame()

    portfolios_return["Date"] = dates
    portfolios_return["Shorting"] = short
    portfolios_return["No Shorting"] = no_short
    portfolios_return["Tangent"] = tangent
    portfolios_return["Equal wgt"] = equal
    portfolios_return["Min 8%"] = min8
    portfolios_return["Max 20%"] = max20


    portfolios_return.set_index("Date", inplace=True)

    return portfolios_return

