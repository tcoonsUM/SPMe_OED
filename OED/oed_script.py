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
import utils_oed as ute
import torch
import g
import time
from bayes_opt import BayesianOptimization, acquisition
from scipy.optimize import NonlinearConstraint

def constraint_func_2dim(d0, d1, d2, d3, d4, d5, d6, d7):
    dim0 = d0*d1
    dim1 = d6*d7
    return np.array([dim0, dim1])

def bo_friendly_objective(d0, d1, d2, d3, d4, d5, d6, d7, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz):
    
    d = torch.tensor([[d0, d1, d2, d3, d4, d5, d6, d7]]) # should have [1,8] shape
    utilities = ute.eig_mp(d, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz)
    
    return utilities.mean()

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
    n_in = int(1e2)
    n_out = int(1e2)
    n_theta = 8
    n_y = 135
    
    # lb = np.array([0.1, -np.inf, 100., 0.1,  1.,  10., 0.001, -np.inf])
    # ub = np.array([2. ,  np.inf, 1800, 2. , 20., 600., 0.01 ,  np.inf])
    
    lb = np.array([0.1, 36.  , 100., 0.1,  1.,  10., 0.001, 25.  ])
    ub = np.array([2. , 3600., 1800, 2. , 20., 600., 0.01 , 1000.])
    
    d_test = ((lb+ub)/2).reshape(1,-1)
    d_test[0,1] = 120/d_test[0,0] # d1 must be in [72/d0, 360/d0]
    d_test[0,7] = 0.67/d_test[0,6] # d7 must be in [.25/d6, 1/d6]
    
    pbounds = { 'd0': (lb[0], ub[0]),
                'd1': (lb[1], ub[1]),
                'd2': (lb[2], ub[2]),
                'd3': (lb[3], ub[3]),
                'd4': (lb[4], ub[4]),
                'd5': (lb[5], ub[5]),
                'd6': (lb[6], ub[6]),
                'd7': (lb[7], ub[7])
                }
    
    # define scipy NonlinearConstraint
    constraint_lower = np.array([72,  0.25])
    constraint_upper = np.array([360, 1.])
    constraint = NonlinearConstraint(constraint_func_2dim, constraint_lower, constraint_upper)
    
    # define optimization parameters
    random_state = 42
    verbose = 1
    init_points = 1
    n_iter = 100
    x0 = d_test
    acq = acquisition.ExpectedImprovement(xi=0.15, exploration_decay=0.15/(1.1*n_iter), random_state=random_state)
    
    optimizer = BayesianOptimization(
        f=lambda d0, d1, d2, d3, d4, d5, d6, d7: 
            bo_friendly_objective(d0, d1, d2, d3, d4, d5, d6, d7, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz),
        constraint=constraint,
        acquisition_function=acq,
        pbounds=pbounds,
        random_state=random_state,
        )
        
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
        )
    
    # NOTE: eig_mp expects d in shape [1,8] as an np.array or tensor
    #%
    # n_test=100
    # thetas_test = torch.tensor(ute.sample_prior(n_test).T)
    # d_test = torch.tensor(np.random.uniform(lb, ub,size=(1,8)))
    # X_test = torch.cat((thetas_test,d_test.repeat(n_test,1)), dim=1)
    # y_test = g.g(X_test)
    #%

    # uncomment to perform N_nmc pilot study
    
    # num_tests = 20
    # utilities_list = []
    # uds = np.zeros((num_tests,))
    # ud_vars = np.zeros((num_tests,))
    # durs = np.zeros((num_tests,))
    # n_tests = np.logspace(2, 3, num_tests)
    # i=0
    # for n_test in n_tests:
    #     n_test_int = int(n_test)
    #     print("Now running for n_test: "+str(n_test_int))
    #     start_time = time.time()
    #     utilities = ute.eig_mp(d_test, n_test_int, n_test_int, rel_std, clusters_inds_npz, corrs_clustered_npz)
    #     stop_time = time.time()
    #     dur = stop_time-start_time
    #     print(utilities.mean())
    #     print(utilities.var()/n_test)
    #     print(dur)
        
    #     utilities_list.append(utilities)
    #     ud_vars[i] = utilities.var()/n_test
    #     uds[i] = utilities.mean()
    #     durs[i] = dur
    #     i+=1
        


    # #%% test functions in utils
    
    # import time
    # n_test=1000
    # thetas_test = torch.tensor(ute.sample_prior(n_test).T)
    # d_test = torch.tensor(np.random.uniform(lb, ub,size=(1,8)))
    # X_test = torch.cat((thetas_test,d_test.repeat(n_test,1)), dim=1)
    # y_test = g.g(X_test)
    # y_test_np = y_test.detach().numpy()
    # clusters_inds_dict = {key: clusters_inds_npz[key] for key in clusters_inds_npz.files}
    # corrs_clustered_dict = {key: corrs_clustered_npz[key] for key in corrs_clustered_npz.files}
    # #test, test_cov = ute.sample_epsilon(y_test_np[0,:], rel_std, clusters_inds_npz, corrs_clustered_npz)
    # # Timing the eval_log_likelihood function
    # start_time_llh = time.time()  # Start the timer
    # test, test_cov = ute.sample_epsilon(y_test_np[0,:], rel_std, clusters_inds_npz, corrs_clustered_npz)
    # #llh = ute.eval_log_likelihood(y_test_np[0,:]+test, y_test_np[0,:], rel_std, clusters_inds_npz, corrs_clustered_npz)
    # end_time_llh = time.time()  # End the timer
    # time_llh = end_time_llh - start_time_llh  # Calculate the time taken for eval_log_likelihood
    
    # # Timing the eval_log_likelihood_mvn function
    # start_time_llh2 = time.time()  # Start the timer
    # #llh2 = ute.eval_log_likelihood_numba(y_test_np[0,:]+test, y_test_np[0,:], rel_std, clusters_inds_npz, corrs_clustered_npz)
    # test, test_cov = ute.sample_epsilon_dict(y_test_np[0,:], rel_std, clusters_inds_dict, corrs_clustered_dict)
    # end_time_llh2 = time.time()  # End the timer
    # time_llh2 = end_time_llh2 - start_time_llh2  # Calculate the time taken for eval_log_likelihood_mvn
    
    # # Output the results
    # print(f"Time taken for eval_log_likelihood: {time_llh} seconds")
    # print(f"Time taken for eval_log_likelihood_numba: {time_llh2} seconds")
    
    # # Print the log-likelihood values
    # #print(np.exp(llh))
    # #print(np.exp(llh2))