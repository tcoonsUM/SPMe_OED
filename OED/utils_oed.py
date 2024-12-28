#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:27:34 2024

@author: me-tcoons
"""

from scipy.stats import uniform, norm, multivariate_normal
import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import g as g_func
from numba import njit, jit
import time
import multiprocessing

def sample_prior(n_samps, seed=42):
    # parameters are, in order:
    # a_nmc, b_nmc, c_nmc, d_nmc (assumed independent)
    # graphite_diff_parameter posterior
    # ep_por, neg_por, pos_por
    # cap_dl_neg
    
    np.random.seed(seed)
    
    sigma = np.array([[ 414.16743966, -439.53022219,  157.53082159,  -36.59908635], \
            [-439.53022219,  475.21734538, -170.37237657,   41.13810463], \
            [ 157.53082159, -170.37237657,   61.12237867,  -14.75655731], \
            [ -36.59908635,   41.13810463,  -14.75655731,    3.8372772 ]])
    mu = np.array([-2.29714210e+01, -1.23599647e-02, -1.09287243e+00,  1.62538939e+00])
    #a_nmc, b_nmc, c_nmc, d_nmc = multivariate_normal.rvs(mu,sigma,size=n_samps)
    nmc_coefs = np.random.multivariate_normal(mu, sigma, size=n_samps) # multivariate_normal.rvs(mu,sigma,size=n_samps)
    # graphite_diff_parameter posterior from inference
    stdev = np.sqrt(3.62066527e-05)
    mu = 1.
    graphite_diff_parameter = np.random.normal(mu, stdev, size=n_samps)[0] #norm.rvs(mu, stdev, size=n_samps)[0]
    # sep_por, neg_por, pos_por
    lb = 0.2
    ub = 0.45
    sep_por = np.random.uniform(lb,ub,size=n_samps)
    neg_por = np.random.uniform(lb,ub,size=n_samps)
    pos_por = np.random.uniform(lb,ub,size=n_samps)
    
    # cap_dl_neg
    mu = 0.2
    stdev = 0.05
    cap_dl_neg = np.random.normal(mu, stdev, size=n_samps)[0] # norm.rvs(mu, stdev, size=1)[0]

    samps_list = [nmc_coefs[:,0], nmc_coefs[:,1], nmc_coefs[:,2], nmc_coefs[:,3], 
                  graphite_diff_parameter, sep_por, neg_por, pos_por, cap_dl_neg]
 
    n_theta = len(samps_list)
    samps = np.zeros((n_theta,n_samps))
    
    for i in range(n_theta):
        samps[i,:] = samps_list[i]
        
    return samps

def sample_epsilon_prime_cov(cov_all, n_samps=1, seed=44):       
    n_stats = 27 # number of summary stats that are repeated
    n_repeats = 5 # number of repeats
    
    bit_gen = np.random.PCG64(seed=seed)
    my_generator = np.random.default_rng(seed=bit_gen)
    
    eps_samp = my_generator.multivariate_normal(np.zeros((n_stats*n_repeats,)), cov_all, size=n_samps, method='cholesky')#multivariate_normal.rvs(cov=cov_all)
    
    return eps_samp


@njit(fastmath=True)
def logpdf_np(x, mean, cov):
    """
    Log of the multivariate normal probability density function.

    Args:
        x (array-like): Value at which to evaluate the logpdf.
        mean (array-like): Mean vector of the distribution.
        cov (array-like): Covariance matrix of the distribution.

    Returns:
        float: Log of the PDF.
    """
    k = len(x)
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    det_cov = np.linalg.det(cov)
    if det_cov == 0:
        raise ValueError("Covariance matrix is singular.")

    inv_cov = np.linalg.inv(cov)

    diff = x - mean
    maha_dist = np.dot(np.dot(diff, inv_cov), diff)

    log_pdf = -0.5 * (k * np.log(2 * np.pi) + np.log(det_cov) + maha_dist)
    return log_pdf

@njit(fastmath=True)
def solve_triangular_numba(L, b):
    """Solves L @ x = b for lower-triangular matrix L."""
    n = L.shape[0]
    x = np.empty_like(b)
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]
    return x

@njit(fastmath=True)
def logpdf_np_chol(x, mean, L):
    """
    Log of the multivariate normal probability density function using Cholesky decomposition.

    Args:
        x (array-like): Value at which to evaluate the logpdf.
        mean (array-like): Mean vector of the distribution.
        L (array-like): Cholesky factor of the covariance matrix (L @ L.T = cov).

    Returns:
        float: Log of the PDF.
    """
    k = len(x)
    x = np.asarray(x)
    mean = np.asarray(mean)

    diff = x - mean
    # Solve for v in L @ v = diff
    v = solve_triangular_numba(L, diff)  #solve(L, diff) 
    maha_dist = np.sum(v * v)  # Equivalent to v.T @ v

    log_det_cov = 2 * np.sum(np.log(np.diag(L)))  # Log-determinant of cov = 2 * sum(log(diag(L)))

    log_pdf = -0.5 * (k * np.log(2 * np.pi) + log_det_cov + maha_dist)
    return log_pdf

@njit(fastmath=True)
def eval_log_likelihood_cov_numba_prime(y, g_eval, L, jitter=1e-4):
    
    # compute g and eps, via y = g(theta, d) * (1 + eps)
    eps = (y - g_eval)/g_eval 
    
    logpdf = logpdf_np_chol(eps, np.zeros(eps.shape), L)
    
    return logpdf

# Worker function to compute log-ratio of likelihood to evidence (utility) for a specific i
@njit(fastmath=True)
def utility_with_reuse_worker_cov(i, y_vals, model_evals, n_in, L):
    evidence = 0
    log_likelihood_ij_same = 0

    # Inner loop: process all j for a given i
    for j in range(n_in):
        log_likelihood = eval_log_likelihood_cov_numba_prime(y_vals, model_evals[j, :], L)
        evidence += np.exp(log_likelihood)
        if i == j:
            log_likelihood_ij_same = log_likelihood

    evidence /= n_in
    utility = log_likelihood_ij_same - np.log(evidence)
    return utility

# Outer function with multiprocessing for the outer loop
def utility_with_reuse_mp_cov(y_vals, model_evals, n_in, n_out, cov_all, n_workers=None):
    
    # Create a pool of workers
    n_workers = n_workers or multiprocessing.cpu_count()-2
    print("Number of cores in use: "+str(n_workers))
    L = np.linalg.cholesky(cov_all)
    
    # for debugging only
    # results2 = []
    # for i in range(n_out):
    #     results2.append(utility_with_reuse_worker_cov(i, y_vals[i,:], model_evals, n_in, L))

    with multiprocessing.Pool(n_workers) as pool:
        # Parallelize the outer loop
        results = pool.starmap(
            utility_with_reuse_worker_cov,
            [(i, y_vals[i,:], model_evals, n_in, L) for i in range(n_out)]
        )

    return np.array(results)

import dask
import dask.array as da
from dask.distributed import Client

def utility_with_reuse_mp_cov_dask(y_vals, model_evals, n_in, n_out, cov_all):
    """
    Calculates utility with reuse using Dask for parallelization.

    Args:
        y_vals: 
        model_evals: 
        n_in: 
        n_out: 
        cov_all: 

    Returns:
        An array of results.
    """

    # Create a Dask client
    client = Client(n_workers=multiprocessing.cpu_count()-2) 

    # Create a list of delayed functions
    L = np.linalg.cholesky(cov_all)
    delayed_results = [dask.delayed(utility_with_reuse_worker_cov)(i, y_vals[i,:], model_evals, n_in, L) for i in range(n_out)]

    # Compute the results using Dask
    results = client.compute(delayed_results)
    results = client.gather(results) 
    
    client.close() 

    return np.array(results)

def eig_mp_cov(d, n_in, n_out, cov_all, jitter=1e-4, seed=42):
    
    n_y = 135
    assert n_in==n_out, "n_in and n_out must take the same value for sample reuse"
    
    # first run g_evals via g_func.g
    # convert inputs to appropriate torch tensors
    # d will be the same (repeats), theta will be n_in/n_out new samples
    thetas = torch.tensor(sample_prior(n_in, seed).T)
    
    if not torch.is_tensor(d):
        d = torch.tensor(d)
    d_repeats = d.repeat(n_in,1)
    
    # torch.tensor concatenation
    X = torch.cat((thetas, d_repeats), dim=1)
    
    # run nn surrogate
    g_evals = g_func.g(X).detach().numpy()
    
    # sample epsilon not in parallel
    eps_all = sample_epsilon_prime_cov(cov_all, n_samps=n_out, seed=seed)
    y_vals = g_evals * (np.ones(eps_all.shape) +  eps_all)
    
    start_time = time.time()
    eig = utility_with_reuse_mp_cov_dask(y_vals, g_evals, n_in, n_out, cov_all)
    stop_time = time.time()
    dur = stop_time-start_time
    print("dur for eig " +str(dur))
    
    return eig

# remaining functions not in use!
def sample_prior_scipy(n_samps, seed=42):
    # parameters are, in order:
    # a_nmc, b_nmc, c_nmc, d_nmc (assumed independent)
    # graphite_diff_parameter posterior
    # ep_por, neg_por, pos_por
    # cap_dl_neg
    
    np.random.seed(seed)
    
    sigma = np.array([[ 414.16743966, -439.53022219,  157.53082159,  -36.59908635], \
            [-439.53022219,  475.21734538, -170.37237657,   41.13810463], \
            [ 157.53082159, -170.37237657,   61.12237867,  -14.75655731], \
            [ -36.59908635,   41.13810463,  -14.75655731,    3.8372772 ]])
    mu = np.array([-2.29714210e+01, -1.23599647e-02, -1.09287243e+00,  1.62538939e+00])
    #a_nmc, b_nmc, c_nmc, d_nmc = multivariate_normal.rvs(mu,sigma,size=n_samps)
    nmc_coefs = multivariate_normal.rvs(mu,sigma,size=n_samps)
    # graphite_diff_parameter posterior from inference
    stdev = np.sqrt(3.62066527e-05)
    mu = 1.
    graphite_diff_parameter = norm.rvs(mu, stdev, size=n_samps)[0]
    # sep_por, neg_por, pos_por
    lb = 0.2
    ub = 0.45
    sep_por = np.random.uniform(lb,ub,size=n_samps)
    neg_por = np.random.uniform(lb,ub,size=n_samps)
    pos_por = np.random.uniform(lb,ub,size=n_samps)
    
    # cap_dl_neg
    mu = 0.2
    stdev = 0.05
    cap_dl_neg = norm.rvs(mu, stdev, size=1)[0]

    samps_list = [nmc_coefs[:,0], nmc_coefs[:,1], nmc_coefs[:,2], nmc_coefs[:,3], 
                  graphite_diff_parameter, sep_por, neg_por, pos_por, cap_dl_neg]
 
    n_theta = len(samps_list)
    samps = np.zeros((n_theta,n_samps))
    
    for i in range(n_theta):
        samps[i,:] = samps_list[i]
        
    return samps

def sample_epsilon(g_eval, rel_std, clusters_inds_npz, corrs_clustered_npz, jitter=1e-4, seed=43):
    
    np.random.seed(seed)
    
    n_stats = 27 # number of summary stats that are repeated
    n_repeats = 5 # number of repeats
    n_clusters = len(clusters_inds_npz)
    rel_stds = np.tile(rel_std, n_repeats)
    
    # construct large (sparse) covariance matrix
    cov_all = np.diag(jitter*np.ones((n_stats*n_repeats,)))
    all_inds = []
    for cluster in range(n_clusters):
        key = clusters_inds_npz.files[cluster]
        inds = clusters_inds_npz[key]
        all_inds+=inds.tolist()
        sds = rel_stds[inds]*np.abs(g_eval[inds])
        corr = corrs_clustered_npz[key]
        # add contribution of correlated mvn to overall cov_all
        cov_term = np.diag(sds) @ corr @ np.diag(sds)
        for i in range(len(inds)):
            cov_all[inds[i],inds] += cov_term[i,:]
        
    # add diagonal contributions to non-correlated stats
    rem_inds = list(set(np.arange(27*5)).difference(all_inds))
    sds = rel_stds[rem_inds]*np.abs(g_eval[rem_inds])
    cov_all[rem_inds,rem_inds] += sds**2
    
    eps_samp = multivariate_normal.rvs(cov=cov_all)
    
    return eps_samp, cov_all

def sample_epsilon_dict(g_eval, rel_std, clusters_inds_npz, corrs_clustered_npz, my_generator, jitter=1e-4, seed=44):
    
    np.random.seed(seed)
    
    n_stats = 27 # number of summary stats that are repeated
    n_repeats = 5 # number of repeats
    rel_stds = np.tile(rel_std, n_repeats)
    
    # construct large (sparse) covariance matrix
    cov_all = np.diag(jitter*np.ones((n_stats*n_repeats,)))
    all_inds = []
    for key in clusters_inds_npz.keys():
        inds = clusters_inds_npz[key]
        all_inds+=inds.tolist()
        sds = rel_stds[inds]*np.abs(g_eval[inds])
        corr = corrs_clustered_npz[key]
        # add contribution of correlated mvn to overall cov_all
        cov_term = np.diag(sds) @ corr @ np.diag(sds)
        for i in range(len(inds)):
            cov_all[inds[i],inds] += cov_term[i,:]
        
    # add diagonal contributions to non-correlated stats
    rem_inds = list(set(np.arange(n_stats*n_repeats)).difference(all_inds))
    sds = rel_stds[rem_inds]*np.abs(g_eval[rem_inds])
    cov_all[rem_inds,rem_inds] += sds**2
    
    bit_gen = np.random.PCG64(seed=seed)
    my_generator = np.random.default_rng(seed=bit_gen)
    eps_samp = my_generator.multivariate_normal(np.zeros((n_stats*n_repeats,)), cov_all, method='cholesky')#multivariate_normal.rvs(cov=cov_all)
    
    return eps_samp, cov_all

def sample_epsilon_prime_dict(rel_std, clusters_inds_npz, corrs_clustered_npz, my_generator, n_samps=1, jitter=1e-4, seed=44):
    
    np.random.seed(seed)
    
    n_stats = 27 # number of summary stats that are repeated
    n_repeats = 5 # number of repeats
    rel_stds = np.tile(rel_std, n_repeats)
    
    # construct large (sparse) covariance matrix
    cov_all = jitter*np.diag(np.ones((n_stats*n_repeats,)))
    all_inds = []
    for key in clusters_inds_npz.keys():
        inds = clusters_inds_npz[key]
        all_inds+=inds.tolist()
        sds = rel_stds[inds]
        corr = corrs_clustered_npz[key]
        # add contribution of correlated mvn to overall cov_all
        cov_term = np.diag(sds) @ corr @ np.diag(sds)
        for i in range(len(inds)):
            cov_all[inds[i],inds] += cov_term[i,:]
        
    # add diagonal contributions to non-correlated stats
    rem_inds = list(set(np.arange(n_stats*n_repeats)).difference(all_inds))
    sds = rel_stds[rem_inds]
    cov_all[rem_inds,rem_inds] += sds**2
    
    eps_samp = my_generator.multivariate_normal(np.zeros((n_stats*n_repeats,)), cov_all, size=n_samps, method='cholesky')#multivariate_normal.rvs(cov=cov_all)
    
    return eps_samp, cov_all
    
def eval_log_likelihood(y, g_eval, rel_std, clusters_inds_npz, corrs_clustered_npz, jitter=1e-4):
    
    n_y = y.shape[0]
    n_stats = 27 # number of summary stats that are repeated
    n_repeats = 5 # number of repeats
    
    # Note: we define y as already having the log applied, so we can comment out below:
    # clean data, apply log
    y_cleaned = y.copy()
    # pos_inds = np.array([0, 1, 3, 5, 8, 9, 12, 14, 15, 18, 20, 21, 24, 26]) #7 8 10 11 removed
    # for j in range(n_y):            
    #     # apply logarithm to strictly positive indices
    #     if (pos_inds == np.remainder(j,n_stats)).sum():
    #         if np.isnan(y_cleaned[j]).sum()<=0:
    #             y_cleaned[j] = np.log(y_cleaned[j])
                
    # load in likelihood parameters
    #rel_std = np.load("relative_stds.npy")
    rel_stds = np.tile(rel_std, n_repeats)
    #clusters_inds_npz = np.load("clusters_list.npz") # can access keywords via clusters_npz.files    
    #corrs_clustered_npz = np.load("corrs_clustered_list.npz")
    n_clusters = len(clusters_inds_npz)
    
    # compute g and eps, via y = g(theta, d) + eps
    eps = y_cleaned - g_eval
    
    # compute likelihood as product of independent clusters (sum of logpdfs)
    # also keep track of which indices are part of a cluster
    logpdf = 0
    all_inds = []
    for cluster in range(n_clusters):
        key = clusters_inds_npz.files[cluster]
        inds = clusters_inds_npz[key]
        all_inds+=inds.tolist()
        sds = rel_stds[inds]*np.abs(g_eval[inds])
        corr = corrs_clustered_npz[key]
        logpdf += multivariate_normal.logpdf(eps[inds], 
                                            cov = np.diag(jitter*np.ones((len(inds),))) + np.diag(sds) @ corr @ np.diag(sds),
                                            allow_singular=False )
        # logpdf += logpdf_np(eps[inds], 
        #                     np.zeros(len(inds)), 
        #                     np.diag(jitter*np.ones((len(inds),))) + np.diag(sds) @ corr @ np.diag(sds))
    
    # the remaining indices are themselves the last cluster of independent gaussians
    rem_inds = list(set(np.arange(27*5)).difference(all_inds))
    sds = rel_stds[rem_inds]*np.abs(g_eval[rem_inds])
    logpdf += multivariate_normal.logpdf(eps[rem_inds], cov = np.diag(sds**2+jitter), allow_singular=False  )
    #logpdf += np.sum(np.array([norm.logpdf(x, scale=np.sqrt(sd**2 + jitter)) for x, sd in zip(eps[rem_inds], sds)]))
    
    return logpdf

def eval_log_likelihood_numba(y, g_eval, rel_std, clusters_inds_npz, corrs_clustered_npz, jitter=1e-4):
    
    n_y = y.shape[0]
    n_stats = 27 # number of summary stats that are repeated
    n_repeats = 5 # number of repeats
    
    # Note: we define y as already having the log applied, so we can comment out below:
    # clean data, apply log
    y_cleaned = y.copy()
    # pos_inds = np.array([0, 1, 3, 5, 8, 9, 12, 14, 15, 18, 20, 21, 24, 26]) #7 8 10 11 removed
    # for j in range(n_y):            
    #     # apply logarithm to strictly positive indices
    #     if (pos_inds == np.remainder(j,n_stats)).sum():
    #         if np.isnan(y_cleaned[j]).sum()<=0:
    #             y_cleaned[j] = np.log(y_cleaned[j])
                
    # load in likelihood parameters
    #rel_std = np.load("relative_stds.npy")
    rel_stds = np.tile(rel_std, n_repeats)
    #clusters_inds_npz = np.load("clusters_list.npz") # can access keywords via clusters_npz.files    
    #corrs_clustered_npz = np.load("corrs_clustered_list.npz")
    n_clusters = len(clusters_inds_npz)
    
    # compute g and eps, via y = g(theta, d) + eps
    eps = y_cleaned - g_eval
    
    # compute likelihood as product of independent clusters (sum of logpdfs)
    # also keep track of which indices are part of a cluster
    logpdf = 0
    all_inds = []
    for cluster in range(n_clusters):
        key = clusters_inds_npz.files[cluster]
        inds = clusters_inds_npz[key]
        all_inds+=inds.tolist()
        sds = rel_stds[inds]*np.abs(g_eval[inds])
        corr = corrs_clustered_npz[key]
        # logpdf += multivariate_normal.logpdf(eps[inds], 
        #                                     cov = np.diag(jitter*np.ones((len(inds),))) + np.diag(sds) @ corr @ np.diag(sds),
        #                                     allow_singular=False )
        logpdf += logpdf_np(eps[inds], 
                            np.zeros(len(inds)), 
                            np.diag(jitter*np.ones((len(inds),))) + np.diag(sds) @ corr @ np.diag(sds))
    
    # the remaining indices are themselves the last cluster of independent gaussians
    rem_inds = list(set(np.arange(27*5)).difference(all_inds))
    sds = rel_stds[rem_inds]*np.abs(g_eval[rem_inds])
    logpdf += multivariate_normal.logpdf(eps[rem_inds], cov = np.diag(sds**2+jitter), allow_singular=False  )
    #logpdf += np.sum(np.array([norm.logpdf(x, scale=np.sqrt(sd**2 + jitter)) for x, sd in zip(eps[rem_inds], sds)]))
    
    return logpdf

# def mvn_logpdf(obs, covariance, allow_singular=False):
    
#     return multivariate_normal.logpdf(obs, cov=covariance, allow_singular=allow_singular )

def eval_log_likelihood_dict(y, g_eval, rel_std, clusters_inds_npz, corrs_clustered_npz, jitter=1e-4):
    
    n_stats = 27 # number of summary stats that are repeated
    n_repeats = 5 # number of repeats
    
    # Note: we define y as already having the log applied, so we can comment out below:
    # clean data, apply log
    y_cleaned = y.copy()
    rel_stds = np.tile(rel_std, n_repeats)
    
    # compute g and eps, via y = g(theta, d) + eps
    eps = y_cleaned - g_eval
    
    # compute likelihood as product of independent clusters (sum of logpdfs)
    # also keep track of which indices are part of a cluster
    logpdf = 0
    all_inds = []
    for key in clusters_inds_npz.keys():
        inds = clusters_inds_npz[key]
        all_inds+=inds.tolist()
        sds = rel_stds[inds]*np.abs(g_eval[inds])
        corr = corrs_clustered_npz[key]
        # logpdf += multivariate_normal.logpdf(eps[inds], 
        #                                      cov = np.diag(jitter*np.ones((len(inds),))) + np.diag(sds) @ corr @ np.diag(sds),
        #                                      allow_singular=False )
        logpdf += logpdf_np(eps[inds], 
                            np.zeros(len(inds)), 
                            np.diag(jitter*np.ones((len(inds),))) + np.diag(sds) @ corr @ np.diag(sds))
    
    # the remaining indices are themselves the last cluster of independent gaussians
    rem_inds = list(set(np.arange(n_stats*n_repeats)).difference(all_inds))
    sds = rel_stds[rem_inds]*np.abs(g_eval[rem_inds])
    scales = np.sqrt(sds**2 + jitter)
    #logpdf += np.sum(norm.logpdf_np(eps[rem_inds], scale=scales))#multivariate_normal.logpdf(eps[rem_inds], cov = np.diag(sds**2+jitter), allow_singular=False  ) #
    logpdf += logpdf_np(eps[rem_inds], np.zeros(len(rem_inds)), np.diag(scales))
    
    #logpdf += multivariate_normal.logpdf(eps[rem_inds], cov = np.diag(sds**2+jitter), allow_singular=False  ) 
    
    return logpdf

@njit(fastmath=True)
def eval_log_likelihood_tuple_numba(y, g_eval, rel_std, clusters_inds_npz, corrs_clustered_npz, rem_inds, all_inds, jitter=1e-4):
    
    # n_stats = 27 # number of summary stats that are repeated
    n_repeats = 5 # number of repeats
    
    # Note: we define y as already having the log applied, so we can comment out below:
    # clean data, apply log
    y_cleaned = y.copy()
    #rel_stds = np.tile(rel_std, n_repeats)
    rel_stds = np.repeat(rel_std,n_repeats).reshape(-1,n_repeats).T.flatten()
    
    # compute g and eps, via y = g(theta, d) + eps
    eps = y_cleaned - g_eval
    
    # compute likelihood as product of independent clusters (sum of logpdfs)
    # also keep track of which indices are part of a cluster
    logpdf = 0
    # all_inds = []
    # for key in clusters_inds_npz.keys():
    #    inds = clusters_inds_npz[key]
    #    corr = corrs_clustered_npz[key]
    n_clusters = len(clusters_inds_npz)
    for i in range(n_clusters): #inds, corr in zip(clusters_inds_npz, corrs_clustered_npz): 
        inds = clusters_inds_npz[i]
        corr = corrs_clustered_npz[i]
        # all_inds+=inds.tolist()
        sds = rel_stds[inds]*np.abs(g_eval[inds])
        # logpdf += multivariate_normal.logpdf(eps[inds], 
        #                                      cov = np.diag(jitter*np.ones((len(inds),))) + np.diag(sds) @ corr @ np.diag(sds),
        #                                      allow_singular=False )
        logpdf += logpdf_np(eps[inds], 
                            np.zeros(len(inds)), 
                            np.diag(jitter*np.ones((len(inds),))) + np.diag(sds) @ corr @ np.diag(sds))
    
    # the remaining indices are themselves the last cluster of independent gaussians
    # rem_inds = list(set(np.arange(n_stats*n_repeats)).difference(all_inds))
    # rem_inds = np.array(rem_inds)
    sds = rel_stds[rem_inds]*np.abs(g_eval[rem_inds])
    scales = np.sqrt(sds**2 + jitter)
    #logpdf += np.sum(norm.logpdf_np(eps[rem_inds], scale=scales))#multivariate_normal.logpdf(eps[rem_inds], cov = np.diag(sds**2+jitter), allow_singular=False  ) #
    logpdf += logpdf_np(eps[rem_inds], np.zeros(len(rem_inds)), np.diag(scales))
    
    #logpdf += multivariate_normal.logpdf(eps[rem_inds], cov = np.diag(sds**2+jitter), allow_singular=False  ) 
    
    return logpdf

@njit(fastmath=True)
def eval_log_likelihood_tuple_numba_prime(y, g_eval, rel_std, clusters_inds_npz, corrs_clustered_npz, rem_inds, all_inds, jitter=1e-4):
    
    # n_stats = 27 # number of summary stats that are repeated
    n_repeats = 5 # number of repeats
    
    # Note: we define y as already having the log applied, so we can comment out below:
    # clean data, apply log
    y_cleaned = y.copy()
    #rel_stds = np.tile(rel_std, n_repeats)
    rel_stds = np.repeat(rel_std,n_repeats).reshape(-1,n_repeats).T.flatten()
    
    # compute g and eps, via y = g(theta, d) * (1 + eps)
    eps = (y_cleaned - g_eval)/g_eval 
    
    # compute likelihood as product of independent clusters (sum of logpdfs)
    # also keep track of which indices are part of a cluster
    logpdf = 0
    # all_inds = []
    # for key in clusters_inds_npz.keys():
    #    inds = clusters_inds_npz[key]
    #    corr = corrs_clustered_npz[key]
    n_clusters = len(clusters_inds_npz)
    for i in range(n_clusters): #inds, corr in zip(clusters_inds_npz, corrs_clustered_npz): 
        inds = clusters_inds_npz[i]
        corr = corrs_clustered_npz[i]
        # all_inds+=inds.tolist()
        sds = rel_stds[inds]
        # logpdf += multivariate_normal.logpdf(eps[inds], 
        #                                      cov = np.diag(jitter*np.ones((len(inds),))) + np.diag(sds) @ corr @ np.diag(sds),
        #                                      allow_singular=False )
        logpdf += logpdf_np(eps[inds], 
                            np.zeros(len(inds)), 
                            np.diag(sds) @ corr @ np.diag(sds) + np.diag(np.ones(len(inds))*jitter))
    
    # the remaining indices are themselves the last cluster of independent gaussians
    # rem_inds = list(set(np.arange(n_stats*n_repeats)).difference(all_inds))
    # rem_inds = np.array(rem_inds)
    sds = rel_stds[rem_inds]
    scales = np.sqrt(sds**2)
    #logpdf += np.sum(norm.logpdf_np(eps[rem_inds], scale=scales))#multivariate_normal.logpdf(eps[rem_inds], cov = np.diag(sds**2+jitter), allow_singular=False  ) #
    logpdf += logpdf_np(eps[rem_inds], np.zeros(len(rem_inds)), np.diag(scales))
    
    #logpdf += multivariate_normal.logpdf(eps[rem_inds], cov = np.diag(sds**2+jitter), allow_singular=False  ) 
    
    return logpdf


# Outer function with multiprocessing for the outer loop
def utility_with_reuse_mp(y_vals, model_evals, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz, n_workers=None):
    # Convert npz files to serializable dictionaries
    clusters_inds_dict = {key: clusters_inds_npz[key] for key in clusters_inds_npz.files}
    corrs_clustered_dict = {key: corrs_clustered_npz[key] for key in corrs_clustered_npz.files}
    clusters_inds_tuple = tuple(clusters_inds_dict.values())
    corrs_clustered_tuple = tuple(corrs_clustered_dict.values())
    
    rem_inds = np.load("rem_inds.npy")
    all_inds = np.load("all_inds.npy")

    # Create a pool of workers
    n_workers = n_workers or multiprocessing.cpu_count()-2
    print("Number of cores in use: "+str(n_workers))
    
    # for debugging only
    # results2 = []
    # for i in range(n_out):
    #     results2.append(utility_with_reuse_worker(i,y_vals, model_evals, n_in, rel_std, clusters_inds_tuple, corrs_clustered_tuple, rem_inds, all_inds))

    with multiprocessing.Pool(n_workers) as pool:
        # Parallelize the outer loop
        results = pool.starmap(
            utility_with_reuse_worker,
            [(i, y_vals, model_evals, n_in, rel_std, clusters_inds_tuple, corrs_clustered_tuple, rem_inds, all_inds) for i in range(n_out)]
        )

    return np.array(results)

# Worker function to compute log-ratio of likelihood to evidence (utility) for a specific i
def utility_with_reuse_worker(i, y_vals, model_evals, n_in, rel_std, clusters_inds_tuple, corrs_clustered_tuple, rem_inds, all_inds):
    evidence = 0
    log_likelihood_ij_same = 0

    # Inner loop: process all j for a given i
    for j in range(n_in):
        log_likelihood = eval_log_likelihood_tuple_numba_prime(
            y_vals[i, :], model_evals[j, :], rel_std, clusters_inds_tuple, corrs_clustered_tuple, rem_inds, all_inds )
        evidence += np.exp(log_likelihood)
        if i == j:
            log_likelihood_ij_same = log_likelihood

    evidence /= n_in
    utility = log_likelihood_ij_same - np.log(evidence)
    return utility

def sample_task(i, g_evals, rel_std, clusters_inds_dict, corrs_clustered_dict, jitter):
    """Worker function for parallel execution."""
    #print(f"Processing sample_task for i={i}")
    eps, _ = sample_epsilon_dict(
        g_evals, rel_std, clusters_inds_dict, corrs_clustered_dict, jitter, seed=i )
    #print(f"Completed task {i}")
    return g_evals + eps

def eig_mp(d, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz, jitter=1e-4, seed=42):
    
    n_y = 135
    assert n_in==n_out, "n_in and n_out must take the same value for sample reuse"
    
    # first run g_evals via g_func.g
    # convert inputs to appropriate torch tensors
    # d will be the same (repeats), theta will be n_in/n_out new samples
    thetas = torch.tensor(sample_prior(n_in, seed).T)
    
    if not torch.is_tensor(d):
        d = torch.tensor(d)
    d_repeats = d.repeat(n_in,1)
    
    # torch.tensor concatenation
    X = torch.cat((thetas, d_repeats), dim=1)
    
    # run nn surrogate
    g_evals = g_func.g(X).detach().numpy()
    
    # add noise to sample y_vals
    y_vals = np.zeros((n_in, n_y))
    
    # convert npz files to dicts
    clusters_inds_dict = {key: clusters_inds_npz[key] for key in clusters_inds_npz.files}
    corrs_clustered_dict = {key: corrs_clustered_npz[key] for key in corrs_clustered_npz.files}

    print("starting sampling epsilon loop")
    start_time = time.time()
    # for i in range(n_in):
    #     eps, _ = sample_epsilon_dict(g_evals[i,:], rel_std, clusters_inds_npz, corrs_clustered_npz, jitter)
    #     y_vals[i,:] = g_evals[i,:] + eps
    
    # set n_workers
    # n_workers = multiprocessing.cpu_count()-2
    # print("n_workers = "+str(n_workers))
    
    # sample epsilon in parallel
    # with multiprocessing.get_context("spawn").Pool(1) as pool:
    #     y_vals_list = pool.starmap( sample_task, 
    #                                [(i, g_evals[i,:], rel_std, clusters_inds_dict, corrs_clustered_dict, jitter) for i in range(n_in)] )

    
    # # Convert the list of arrays back to a NumPy array
    # y_vals = np.vstack(y_vals_list)
    
    # sample epsilon not in parallel
    bit_gen = np.random.PCG64(seed=seed)
    my_generator = np.random.default_rng(seed=bit_gen)
    eps_all, cov_all = sample_epsilon_prime_dict(rel_std, clusters_inds_dict, corrs_clustered_dict, my_generator, n_samps=n_out)
    y_vals = g_evals * (np.ones(eps_all.shape) +  eps_all)

    # for i in range(n_in):
    #     eps, eps_cov = sample_epsilon_dict(g_evals[i,:], rel_std, clusters_inds_dict, corrs_clustered_dict, jitter)
    #     y_vals[i,:] = g_evals[i,:] + eps
    
    stop_time = time.time()
    dur = stop_time-start_time
    print("dur for sample eps: " +str(dur))
    
    start_time = time.time()
    eig = utility_with_reuse_mp(y_vals, g_evals, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz)
    stop_time = time.time()
    dur = stop_time-start_time
    print("dur for eig " +str(dur))
    
    return eig

def utility_with_reuse(y_vals, model_evals, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz):
    u_d = np.zeros((n_out,))
    assert n_in==n_out, "n_in and n_out must take the same value for sample reuse"
    for i in range(n_out):
        # if i%20==0:
        #     print(i)
        evidence = 0
        
        for j in range(n_in):   
            log_likelihood = eval_log_likelihood(y_vals[i,:], model_evals[j,:], rel_std, clusters_inds_npz, corrs_clustered_npz)#(y_vals[j,:], model_evals[j,:], eps_mean, eps_cov)
            evidence += np.exp(log_likelihood)
            if i==j:
                log_likelihood_ij_same = log_likelihood
            
        evidence /= n_in
        u_d[i] += log_likelihood_ij_same - np.log(evidence)
    return u_d

def eig(d, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz, jitter=1e-4, seed=42):
    
    n_y = 135
    assert n_in==n_out, "n_in and n_out must take the same value for sample reuse"
    
    # first run g_evals via g_func.g
    # convert inputs to appropriate torch tensors
    # d will be the same (repeats), theta will be n_in/n_out new samples
    thetas = torch.tensor(sample_prior(n_in, seed).T)
    if not torch.is_tensor(d):
        d = torch.tensor(d)
    d_repeats = d.repeat(n_in,1)
    
    # torch.tensor concatenation
    X = torch.cat((thetas, d_repeats), dim=1)
    
    # run nn surrogate
    g_evals = g_func.g(X).detach().numpy()
    
    # add noise to sample y_vals
    y_vals = np.zeros((n_in, n_y))
    for i in range(n_in):
        eps, eps_cov = sample_epsilon(g_evals[i,:], rel_std, clusters_inds_npz, corrs_clustered_npz, jitter)
        y_vals[i,:] = g_evals[i,:] + eps
        
    eig = utility_with_reuse(y_vals, g_evals, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz)
    
    return eig

# def eval_log_likelihood_mvn(y, g_eval, cov_all):
    
#     return multivariate_normal.logpdf(y-g_eval,cov=cov_all)

# def utility_with_reuse_mvn(y_vals, model_evals, n_in, n_out, eps_covs):
#     u_d = np.zeros((n_out,))
#     assert n_in==n_out, "n_in and n_out must take the same value for sample reuse"
#     for i in range(n_out):
#         # if i%20==0:
#         #     print(i)
#         evidence = 0
        
#         for j in range(n_in):   
#             log_likelihood = eval_log_likelihood_mvn(y_vals[i,:], model_evals[j,:], eps_covs[i,:,:])#(y_vals[j,:], model_evals[j,:], eps_mean, eps_cov)
#             evidence += np.exp(log_likelihood)
#             if i==j:
#                 log_likelihood_ij_same = log_likelihood
            
#         evidence /= n_in
#         u_d[i] += log_likelihood_ij_same - np.log(evidence)
#     return u_d

# def eig_mvn(d, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz, jitter=1e-4, seed=42):
    
#     n_y = 135
#     assert n_in==n_out, "n_in and n_out must take the same value for sample reuse"
    
#     # first run g_evals via g_func.g
#     # convert inputs to appropriate torch tensors
#     # d will be the same (repeats), theta will be n_in/n_out new samples
#     thetas = torch.tensor(sample_prior(n_in, seed).T)
#     if not torch.is_tensor(d):
#         d = torch.tensor(d)
#     d_repeats = d.repeat(n_in,1)
    
#     # torch.tensor concatenation
#     X = torch.cat((thetas, d_repeats), dim=1)
    
#     # run nn surrogate
#     g_evals = g_func.g(X).detach().numpy()
    
#     # add noise to sample y_vals
#     y_vals = np.zeros((n_in, n_y))
#     eps_covs = np.zeros((n_in, n_y, n_y))
#     for i in range(n_in):
#         eps, eps_cov = sample_epsilon(g_evals[i,:], rel_std, clusters_inds_npz, corrs_clustered_npz, jitter)
#         y_vals[i,:] = g_evals[i,:] + eps
#         eps_covs[i, :, :] = eps_cov
        
#     eig = utility_with_reuse_mvn(y_vals, g_evals, n_in, n_out, eps_covs)
    
#     return eig
