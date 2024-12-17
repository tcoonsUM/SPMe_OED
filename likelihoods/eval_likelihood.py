#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:07:41 2024

@author: me-tcoons
"""

import numpy as np
from scipy.stats import multivariate_normal
from numpy import linalg as la

# def isPD(B):
#     """Returns true when input is positive-definite, via Cholesky"""
#     try:
#         _ = la.cholesky(B)
#         return True
#     except la.LinAlgError:
#         return False

# def nearestPD(A):
#     """Find the nearest positive-definite matrix to input
#     A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
#     credits [2].
#     [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
#     [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
#     matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
#     """
#     B = (A + A.T) / 2
#     _, s, V = la.svd(B)
#     H = np.dot(V.T, np.dot(np.diag(s), V))
#     A2 = (B + H) / 2
#     A3 = (A2 + A2.T) / 2
#     if isPD(A3):
#         return A3
#     spacing = np.spacing(la.norm(A))
#     # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
#     # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
#     # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
#     # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
#     # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
#     # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
#     # `spacing` will, for Gaussian random matrixes of small dimension, be on
#     # othe order of 1e-16. In practice, both ways converge, as the unit test
#     # below suggests.
#     I = np.eye(A.shape[0])
#     k = 1
#     while not isPD(A3):
#         mineig = np.min(np.real(la.eigvals(A3)))
#         A3 += I * (-mineig * k**2 + spacing)
#         k += 1
#     return A3

def eval_log_likelihood(y, theta, d):
    
    # clean data, apply log
    pos_inds = np.array([0, 1, 3, 5, 8, 9, 12, 14, 15, 18, 20, 21, 24, 26]) #7 8 10 11 removed
    # original positive indices: 0, 1, 3, 5, 7(X), 9(-1), 10(X), 12(-4 from here onwards), 13, 16, 18, 19, 22, 24, 25, 28, 30
    n_y = y.shape[0]
    n_stats = 27 # number of summary stats that are repeated
    n_repeats = 5 # number of repeats
    y_cleaned = y.copy()
    for j in range(n_y):            
        # apply logarithm to strictly positive indices
        if (pos_inds == np.remainder(j,n_stats)).sum():
            if np.isnan(y_cleaned[j]).sum()<=0:
                y_cleaned[j] = np.log(y_cleaned[j])
                
    # load in likelihood parameters
    rel_std = np.load("relative_stds.npy")
    rel_stds = np.tile(rel_std, n_repeats)
    clusters_inds_npz = np.load("clusters_list.npz") # can access keywords via clusters_npz.files    
    corrs_clustered_npz = np.load("corrs_clustered_list.npz")
    n_clusters = len(clusters_inds_npz)
    
    # compute g and eps, via y = g(theta, d) + eps
    jitter = 1e-4
    g_eval = y_cleaned + multivariate_normal.rvs(
        cov = np.diag((rel_stds*np.abs(y_cleaned))**2))#g(theta,d)
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
                                             allow_singular=True )
    
    # the remaining indices are themselves the last cluster of independent gaussians
    rem_inds = list(set(np.arange(27*5)).difference(all_inds))
    sds = rel_stds[rem_inds]*np.abs(g_eval[rem_inds])
    logpdf += multivariate_normal.logpdf(eps[rem_inds], cov = np.diag(sds**2+jitter), allow_singular=True  )
    
    return logpdf
    
def g(theta, d):
    # placeholder for now
    return theta.sum() + d.sum()

test_y = np.load("test.npy")
log_pdf_test = eval_log_likelihood(test_y, 0., 0.)