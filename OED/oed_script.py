#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:02:19 2024

@author: me-tcoons
"""

import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import utils as ute
import torch
import g
import time

if __name__ == '__main__':    
        #% load in files
    x_scalers_npz = np.load("x_scalers.npz")
    y_scalers_npz = np.load("y_scalers.npz")
    corrs_clustered_npz = np.load("corrs_clustered_list.npz")
    clusters_inds_npz = np.load("clusters_list.npz") # can access keywords via clusters_npz.files    
    rel_std = np.load("relative_stds.npy")
    
    # for testing only
    y_test = np.load("y_cleaned.npy")
    
    #% define OED parameters
    n_in = 1e4
    n_out = 1e4
    n_theta = 8
    n_y = 135
    
    # lb = np.array([0.1, -np.inf, 100., 0.1,  1.,  10., 0.001, -np.inf])
    # ub = np.array([2. ,  np.inf, 1800, 2. , 20., 600., 0.01 ,  np.inf])
    
    lb = np.array([0.1, 36.  , 100., 0.1,  1.,  10., 0.001, 25.  ])
    ub = np.array([2. , 3600., 1800, 2. , 20., 600., 0.01 , 1000.])
    
    #%
    # n_test=100
    # thetas_test = torch.tensor(ute.sample_prior(n_test).T)
    # d_test = torch.tensor(np.random.uniform(lb, ub,size=(1,8)))
    # X_test = torch.cat((thetas_test,d_test.repeat(n_test,1)), dim=1)
    # y_test = g.g(X_test)
    #%

    d_test = torch.tensor(np.load("d_test.npy"))
    n_test = 1000
    start_time = time.time()
    utilities = ute.eig_mp(d_test, n_test, n_test, rel_std, clusters_inds_npz, corrs_clustered_npz)
    stop_time = time.time()
    dur = stop_time-start_time
    print(utilities.mean())
    print(utilities.var()/n_test)
    print(dur)

    #%% test functions in utils
    
    import time
    n_test=1000
    thetas_test = torch.tensor(ute.sample_prior(n_test).T)
    d_test = torch.tensor(np.random.uniform(lb, ub,size=(1,8)))
    X_test = torch.cat((thetas_test,d_test.repeat(n_test,1)), dim=1)
    y_test = g.g(X_test)
    y_test_np = y_test.detach().numpy()
    clusters_inds_dict = {key: clusters_inds_npz[key] for key in clusters_inds_npz.files}
    corrs_clustered_dict = {key: corrs_clustered_npz[key] for key in corrs_clustered_npz.files}
    #test, test_cov = ute.sample_epsilon(y_test_np[0,:], rel_std, clusters_inds_npz, corrs_clustered_npz)
    # Timing the eval_log_likelihood function
    start_time_llh = time.time()  # Start the timer
    test, test_cov = ute.sample_epsilon(y_test_np[0,:], rel_std, clusters_inds_npz, corrs_clustered_npz)
    #llh = ute.eval_log_likelihood(y_test_np[0,:]+test, y_test_np[0,:], rel_std, clusters_inds_npz, corrs_clustered_npz)
    end_time_llh = time.time()  # End the timer
    time_llh = end_time_llh - start_time_llh  # Calculate the time taken for eval_log_likelihood
    
    # Timing the eval_log_likelihood_mvn function
    start_time_llh2 = time.time()  # Start the timer
    #llh2 = ute.eval_log_likelihood_numba(y_test_np[0,:]+test, y_test_np[0,:], rel_std, clusters_inds_npz, corrs_clustered_npz)
    test, test_cov = ute.sample_epsilon_dict(y_test_np[0,:], rel_std, clusters_inds_dict, corrs_clustered_dict)
    end_time_llh2 = time.time()  # End the timer
    time_llh2 = end_time_llh2 - start_time_llh2  # Calculate the time taken for eval_log_likelihood_mvn
    
    # Output the results
    print(f"Time taken for eval_log_likelihood: {time_llh} seconds")
    print(f"Time taken for eval_log_likelihood_numba: {time_llh2} seconds")
    
    # Print the log-likelihood values
    #print(np.exp(llh))
    #print(np.exp(llh2))