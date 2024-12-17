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

def sample_epsilon(g_eval, rel_std, clusters_inds_npz, corrs_clustered_npz, jitter=1e-3, seed=42):
    
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

def eval_log_likelihood_mvn(y, g_eval, cov_all):
    
    return multivariate_normal.logpdf(y-g_eval,cov=cov_all)
    

def eval_log_likelihood(y, g_eval, rel_std, clusters_inds_npz, corrs_clustered_npz, jitter=1e-3):
    
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
    
    # the remaining indices are themselves the last cluster of independent gaussians
    rem_inds = list(set(np.arange(27*5)).difference(all_inds))
    sds = rel_stds[rem_inds]*np.abs(g_eval[rem_inds])
    logpdf += multivariate_normal.logpdf(eps[rem_inds], cov = np.diag(sds**2+jitter), allow_singular=False  )
    
    return logpdf

def eval_log_likelihood_dict(y, g_eval, rel_std, clusters_inds_npz, corrs_clustered_npz, jitter=1e-3):
    
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
    for key in clusters_inds_npz.keys():
        inds = clusters_inds_npz[key]
        all_inds+=inds.tolist()
        sds = rel_stds[inds]*np.abs(g_eval[inds])
        corr = corrs_clustered_npz[key]
        logpdf += multivariate_normal.logpdf(eps[inds], 
                                             cov = np.diag(jitter*np.ones((len(inds),))) + np.diag(sds) @ corr @ np.diag(sds),
                                             allow_singular=False )
    
    # the remaining indices are themselves the last cluster of independent gaussians
    rem_inds = list(set(np.arange(27*5)).difference(all_inds))
    sds = rel_stds[rem_inds]*np.abs(g_eval[rem_inds])
    logpdf += multivariate_normal.logpdf(eps[rem_inds], cov = np.diag(sds**2+jitter), allow_singular=False  )
    
    return logpdf

import multiprocessing

# Worker function to compute utility for a specific i
def utility_with_reuse_worker(i, y_vals, model_evals, n_in, rel_std, clusters_inds_dict, corrs_clustered_dict):
    evidence = 0
    log_likelihood_ij_same = 0

    # Inner loop: process all j for a given i
    for j in range(n_in):
        log_likelihood = eval_log_likelihood_dict(
            y_vals[i, :], model_evals[j, :], rel_std, clusters_inds_dict, corrs_clustered_dict
        )
        evidence += np.exp(log_likelihood)
        if i == j:
            log_likelihood_ij_same = log_likelihood

    evidence /= n_in
    utility = log_likelihood_ij_same - np.log(evidence)
    return utility

# Outer function with multiprocessing for the outer loop
def utility_with_reuse_mp(y_vals, model_evals, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz, n_workers=None):
    # Convert npz files to serializable dictionaries
    clusters_inds_dict = {key: clusters_inds_npz[key] for key in clusters_inds_npz.files}
    corrs_clustered_dict = {key: corrs_clustered_npz[key] for key in corrs_clustered_npz.files}

    # Create a pool of workers
    n_workers = n_workers or multiprocessing.cpu_count()-1
    print(n_workers)
    test_list = [(i, y_vals, model_evals, n_in, rel_std, clusters_inds_dict, corrs_clustered_dict) for i in range(n_out)]
    utility_with_reuse_worker(test_list[0])
    with multiprocessing.Pool(n_workers) as pool:
        # Parallelize the outer loop
        results = pool.starmap(
            utility_with_reuse_worker,
            [(i, y_vals, model_evals, n_in, rel_std, clusters_inds_dict, corrs_clustered_dict) for i in range(n_out)]
        )

    return np.array(results)

def eig_mp(d, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz, jitter=1e-3, seed=42):
    
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
        
    eig = utility_with_reuse_mp(y_vals, g_evals, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz)
    
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

def eig(d, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz, jitter=1e-3, seed=42):
    
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

def utility_with_reuse_mvn(y_vals, model_evals, n_in, n_out, eps_covs):
    u_d = np.zeros((n_out,))
    assert n_in==n_out, "n_in and n_out must take the same value for sample reuse"
    for i in range(n_out):
        # if i%20==0:
        #     print(i)
        evidence = 0
        
        for j in range(n_in):   
            log_likelihood = eval_log_likelihood_mvn(y_vals[i,:], model_evals[j,:], eps_covs[i,:,:])#(y_vals[j,:], model_evals[j,:], eps_mean, eps_cov)
            evidence += np.exp(log_likelihood)
            if i==j:
                log_likelihood_ij_same = log_likelihood
            
        evidence /= n_in
        u_d[i] += log_likelihood_ij_same - np.log(evidence)
    return u_d

def eig_mvn(d, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz, jitter=1e-3, seed=42):
    
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
    eps_covs = np.zeros((n_in, n_y, n_y))
    for i in range(n_in):
        eps, eps_cov = sample_epsilon(g_evals[i,:], rel_std, clusters_inds_npz, corrs_clustered_npz, jitter)
        y_vals[i,:] = g_evals[i,:] + eps
        eps_covs[i, :, :] = eps_cov
        
    eig = utility_with_reuse_mvn(y_vals, g_evals, n_in, n_out, eps_covs)
    
    return eig
