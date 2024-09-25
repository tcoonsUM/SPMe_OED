#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:47:26 2024

Prior Grid Sampling File for Conference Presentation, simulated posteriors

@author: me-tcoons
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats

#%% load data
def load_files(integer):
    folder_path = "summary_statistics_fixed_design_no_chirp_new"
    file_extension = f"_{integer}.npy"

    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            y = np.load(os.path.join(folder_path, filename))

    return y   

def load_files_sim(integer):
    folder_path = "simulation_results_fixed_design_no_chirp_new"
    file_extension = f"_{integer}.npy"
    date_time_str = ""

    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            # Extract the date and time part from the filename
            parts = filename.split('_')
            date_time_str = f"{parts[1]}_{parts[2]}_{parts[3].split('.')[0]}"

            if filename.startswith("params"):
                theta = np.load(os.path.join(folder_path, filename))
            elif filename.startswith("time_data"):
                t = np.load(os.path.join(folder_path, filename))
            elif filename.startswith("voltage_data"):
                v = np.load(os.path.join(folder_path, filename))
            elif filename.startswith("current_data"):
                current = np.load(os.path.join(folder_path, filename))
    return theta[:9], t, v, current, date_time_str

def evaluate_log_likelihood_w_reuse_iid(y_samples, model_evals, mean, variances):
    epsilon_generated = np.subtract(y_samples,model_evals)
    likelihood_pdf = 0.
    for i in range(len(variances)):
        likelihood_pdf += stats.norm.logpdf(epsilon_generated[i],0.,np.sqrt(variances[i]))
    return likelihood_pdf

y_all = np.load("likelihood_no_chirp/model_evals_10K.npy")
integer=235
y = y_all[integer,:]
n_y = len(y)
theta, t, v, current, date_time_str = load_files_sim(integer)
n_theta = len(theta)
eps_mean = np.zeros((n_y,))
eps_variances = np.load("likelihood_no_chirp/vars.npy")
eps_variances[np.where(eps_variances<1e-9)]=1e-9
#%%
n_samples = 5000
likelihoods_all = np.zeros((n_samples,))
for i in range(n_samples):
    model_eval_sample = y_all[i,:]
    likelihoods_all[i] = evaluate_log_likelihood_w_reuse_iid(y, model_eval_sample, eps_mean, eps_variances)

#%% compute weights and begin sampling
weights = likelihoods_all/np.sum(likelihoods_all)
weights_cumsum = np.cumsum(weights)
n_post_samples = 500
samples_post = np.zeros((n_post_samples,n_theta))
for sample in range(n_post_samples):
    rand = np.random.uniform()
    for j in range(len(weights)):
        if rand <= weights_cumsum[j]:
            samples_post[sample,:] = load_files_sim(j)[0]
            break
            
#%% prior plots
n_prior_samps = 1000
prior_samps = np.zeros((n_prior_samps,n_theta))
for i in range(n_prior_samps):
    prior_samps[i] = load_files_sim(i)[0]