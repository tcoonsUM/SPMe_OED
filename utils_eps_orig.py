#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(utils.py function code)
----------
Bayesian Model-Based Optimal Experimental Design Code
Author: Thomas Coons

Date: Summer 2023
"""

import numpy as np
import random
from scipy import stats

def sample_prior(n_sample, n_param, lb, ub, seed=3142):   
    np.random.seed(seed)
    thetas = np.random.uniform(lb, ub, (n_sample, n_param))
    return thetas

def sample_epsilon(n_samples, n_y, mean=np.zeros((100,)), cov=np.diag(np.ones(100,)*3e-3**2), seed=3142):
    # note: sd is standard deviation of Gaussian noise term 
    np.random.seed(seed)
    epsilons = stats.multivariate_normal.rvs(mean=mean,cov=cov,size=n_samples)
    return epsilons

def evaluate_log_epsilon(epsilon, mean=np.zeros((100,)), cov=np.diag(np.ones(100,)*3e-3**2)):
    eps_pdf = stats.multivariate_normal.logpdf(epsilon,mean,cov)
    return eps_pdf

def eig_eps_fast(epsilons, n_out, n_in, y_inner, y_outer, mean, cov):
    u_d = 0
    for i in range(n_out):
        eps_log_pdf = evaluate_log_epsilon(epsilons[i],mean=mean,cov=cov)
            
        eps_inners=y_outer[i]+epsilons[i]-y_inner
        evidence=np.exp(evaluate_log_epsilon(eps_inners,mean,cov))
        evidence=np.sum(evidence)/n_in
        u_d+=eps_log_pdf - np.log(evidence)
        
    u_d/=n_out
    return u_d

def eig_eps_fast_nd(epsilons, n_out, n_in, y_inner, y_outer, mean, cov):
    # version for nY>1 where epsilons are not in list but np.array format
    u_d = 0
    for i in range(n_out):
        eps_log_pdf = evaluate_log_epsilon(epsilons[i,:],mean,cov)
            
        eps_inners=y_outer[i]+epsilons[i,:]-y_inner
        evidence=np.exp(evaluate_log_epsilon(eps_inners,mean,cov))
        evidence=np.sum(evidence)/n_in
        u_d+=eps_log_pdf - np.log(evidence) 
    
    u_d/=n_out
    return u_d

def eig_eps_fast_vec(epsilons, n_out, n_in, y_inner, y_outer):
    u_d = np.empty((n_out,))
    u=0
    for i in range(n_out):
        eps_log_pdf = evaluate_log_epsilon(epsilons[i])
            
        eps_inners=y_outer[i]+epsilons[i]-y_inner
        evidence=np.exp(evaluate_log_epsilon(eps_inners))
        evidence=np.sum(evidence)/n_in
        u+=eps_log_pdf - np.log(evidence)
        
        u_d[i]=u
        u=0
    return u_d