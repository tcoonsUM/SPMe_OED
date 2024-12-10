#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:33:12 2024

@author: me-tcoons
"""

#%%
import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

#%% read in files, convert to torch tensor
df_legacy = pd.read_csv("likelihoods_cleaned.csv")

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu') 

y_legacy = df_legacy.iloc[2:,:].to_numpy(dtype=float)
n_y = y_legacy.shape[1] # dimension of summary stats
n_obs = y_legacy.shape[0] # number of observations

#%% cleaning y data
pos_inds = np.array([0, 1, 3, 5, 8, 9, 12, 14, 15, 18, 20, 21, 24, 26]) #7 8 10 11 removed
nan_inds = [];
# original positive indices: 0, 1, 3, 5, 7(X), 9(-1), 10(X), 12(-4 from here onwards), 13, 16, 18, 19, 22, 24, 25, 28, 30
n_stats = 27 # number of summary stats that are repeated
n_repeats = 5 # number of repeats
y_cleaned = y_legacy.copy()
for j in range(n_y):    
    # finding nan indices
    if np.isnan(y_cleaned[:,j]).sum()>0:
        nan_inds.append(j)
    
    # apply logarithm to strictly positive indices
    if (pos_inds == np.remainder(j,n_stats)).sum():
        if np.isnan(y_cleaned[:,j]).sum()<=0:
            y_cleaned[:,j] = np.log(y_cleaned[:,j])

#%% computing relative stdevs for repeated stats
rel_stds = np.zeros((n_stats,))
for j in range(n_stats):
    j_inds = np.arange(j, j+n_stats*n_repeats, n_stats)
    all_rel_stds = y_cleaned[:,j_inds].std(axis=1)/y_cleaned[:,j_inds].mean(axis=1)
    rel_stds[j] = np.abs(all_rel_stds).mean()
            
#%% filling in NaN stats with similar stats' relative stdevs

# index 1, 2, 3, 4, 5, 6 can be computed using other repeats
nan_first_inds = np.array([1,2,3,4,5,6])
for ind in nan_first_inds:
    next_inds = np.arange(ind+n_stats, ind+n_stats*n_repeats, n_stats)
    all_rel_stds = y_cleaned[:,next_inds].std(axis=1)/y_cleaned[:,next_inds].mean(axis=1)
    rel_stds[ind] = np.abs(all_rel_stds).mean()
   
# index 8 can be computed from later instances of final V's within given repeat
# 8 (final V) corresponds to 14 and 20
rel_stds[8] = np.mean([rel_stds[14],rel_stds[20]])

# leaves 0 and 7, which have no repeats or anything
# we will resort to the average relative stdev's of all other stats
rel_stds[0] = rel_stds[7] = np.nanmean(rel_stds)
np.save("relative_stds.npy", rel_stds)

#%% plotting results
plt.rcParams['figure.dpi'] = 600
plt.style.use('bmh')

plt.bar(np.arange(0,n_stats),100*rel_stds)
plt.xlabel("Stat Index")
plt.ylabel("Relative Stdev (%)")

#%% investigating correlation structures
#y_simulated = y_cleaned.copy()
# filling in with "identical" later stats
# for ind in nan_first_inds:
#     y_simulated[:,ind] = y_cleaned[:,ind+n_stats]

# since we are only using correlations,
# we replace all missing values with 0s and call them uncorrelated
y_simulated = np.nan_to_num(y_cleaned)

# correlation structure
corr_test = np.abs(np.corrcoef(y_simulated.transpose()))
corr_test = np.nan_to_num(corr_test)

#%% counting up correlated stats
corr_counter = []
for i in range(n_y):
    correlated_inds = np.where(corr_test[i,:]>0.9)[0]
    if correlated_inds.shape[0] == 0:
        correlated_inds = np.array([i])
    corr_counter.append(correlated_inds)

#%% manually determining correlated clusters of stats
clusters = []
clusters.append(np.array([9, 10, 11]))
clusters.append(np.array([12,  14,  15,  16,  17,  18,  21,  22,  23,  26,  36,  37,  38,\
         41,  42,  43,  44,  45,  53,  63,  64,  65,  68,  69,  72,  80,\
         95, 122, 126, 129, 134]))
clusters.append(np.array([28, 30, 32, 55, 57, 59, 84, 86, 111, 113, 114]))
clusters.append(np.array([29, 31]))
clusters.append(np.array([49, 50]))
clusters.append(np.array([63, 64, 65]))
clusters.append(np.array([70, 71]))
clusters.append(np.array([75, 76, 77]))
clusters.append(np.array([78, 79]))
clusters.append(np.array([82, 84]))
clusters.append(np.array([90, 91,  92,  99, 107, 123, 124, 125]))
clusters.append(np.array([93, 94]))
clusters.append(np.array([100, 101]))
clusters.append(np.array([103, 104]))
clusters.append(np.array([84, 86, 109, 111, 113, 114]))
clusters.append(np.array([109, 111]))
clusters.append(np.array([110, 112]))
clusters.append(np.array([118, 119]))
clusters.append(np.array([130, 131]))
np.savez("clusters_list.npz", *clusters)

#%% producing correlation matrices and saving
corrs_clustered = []
for cluster in clusters:
    corrs_clustered.append(corr_test[:,cluster][cluster,:])
np.savez("corrs_clustered_list.npz", *corrs_clustered)

#%% can also be repeated with absolute, not relative, stdevs

stds = np.zeros((n_stats,))
for j in range(n_stats):
    j_inds = np.arange(j, j+n_stats*n_repeats, n_stats)
    all_stds = y_cleaned[:,j_inds].std(axis=1)
    stds[j] = np.abs(all_stds).mean()
            
#%% filling in NaN stats with similar stats' relative stdevs

# index 1, 2, 3, 4, 5, 6 can be computed using other repeats
nan_first_inds = np.array([1,2,3,4,5,6])
for ind in nan_first_inds:
    next_inds = np.arange(ind+n_stats, ind+n_stats*n_repeats, n_stats)
    all_stds = y_cleaned[:,next_inds].std(axis=1)
    stds[ind] = np.abs(all_stds).mean()
   
# index 8 can be computed from later instances of final V's within given repeat
# 8 (final V) corresponds to 14 and 20
stds[8] = np.mean([stds[14],stds[20]])

# leaves 0 and 7, which have no repeats or anything
# we will resort to the average relative stdev's of all other stats
stds[0] = stds[7] = np.nanmean(stds)

#%% plotting results
plt.rcParams['figure.dpi'] = 600
plt.style.use('bmh')

plt.bar(np.arange(0,n_stats),stds)
plt.xlabel("Stat Index")
plt.ylabel("Stdev")
