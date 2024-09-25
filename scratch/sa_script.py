#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:49:32 2024

Sensitivity analysis using NN code

@author: me-tcoons
"""
#%% imports
import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib

#%% load data/models and scalers, non chirp

x_scalers = joblib.load('models/x_scalers_9.gz')
y_scalers = joblib.load('models/y_scalers_9.gz')

def full_model_no_chirp(x, n_x=9, n_y=98):
    # x: (n_x, n_samples) of inputs
    # y: (n_y, n_samples) of outputs
    # note that NN torch model takes torch tensor shape (n_samples, n_x) 
    
    class feed_forward(nn.Module): 
      def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.net = nn.Sequential(
            nn.Linear(9, 64), 
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 98*2),
            nn.ReLU(),
            nn.Linear(98*2, 98)
        )
      def forward(self, X):
        return self.net(X)
      def predict(self, X):
        Y_pred = self.forward(X)
        return Y_pred

    model = torch.load("models/no_chirp_98_9.pt")
    
    # start by scaling inputs
    x_scaled = np.zeros(x.shape)
    print(x.shape)
    for i in range(n_x):
        x_scaled[i,:] = x_scalers[i].transform(x[i,:].reshape(-1,1)).flatten()
        
    # reshape (transpose) and make torch tensor of dtype float32
    x_tensor_scaled = torch.tensor(x_scaled.T, dtype=torch.float32)
    
    # make predictions using model
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model.to(device)
    x_tensor_scaled = x_tensor_scaled.to(device)
    model.eval()
    y_tensor_scaled = model(x_tensor_scaled)
    
    # reshape (transpose) and make numpy array from model outputs
    y_np_scaled = torch.transpose(y_tensor_scaled,0,1).cpu().detach().numpy()
    
    # unscale outputs
    y = np.zeros(y_np_scaled.shape)
    print(y_np_scaled.shape)
    for j in range(n_y):
        y[j,:] = y_scalers[j].inverse_transform(y_np_scaled[j,:].reshape(-1,1)).flatten()
     
    return y
        
#%% Sobol index estimation
from scipy.stats import uniform, norm, sobol_indices
rng = np.random.default_rng()

# a_nmc, b_nmc, c_nmc, d_nmc (assumed independent)
# graphite_diff_parameter posterior
# ep_por, neg_por, pos_por
# cap_dl_neg

dists = [norm(-2.29714210e+01,np.sqrt(414.16743966)),
         norm(-1.23599647e-02,np.sqrt(475.21734538)),
         norm(-1.09287243e+00, np.sqrt(61.12237867)),
         norm(1.62538939e+00, np.sqrt(3.8372772)),
         norm(1., np.sqrt(3.62066527e-05)),
         uniform(0.2, 0.45),
         uniform(0.2, 0.45),
         uniform(0.2, 0.45),
         norm(0.2, 0.05)
    ]

indices = sobol_indices(func = full_model_no_chirp,
                        n=2048, dists=dists, random_state=rng
                        )

boot = indices.bootstrap()
#%%
ind_means = np.mean(indices.first_order,axis=0)
boot_low = np.mean(boot.first_order.confidence_interval.low,axis=0)
boot_high = np.mean(boot.first_order.confidence_interval.high,axis=0)

ind_means_t = np.mean(indices.total_order,axis=0)
boot_low_t = np.mean(boot.total_order.confidence_interval.low,axis=0)
boot_high_t = np.mean(boot.total_order.confidence_interval.high,axis=0)

#%% plotting
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
_ = axs[0].errorbar(
    [1, 2, 3, 4, 5, 6, 7, 8, 9], ind_means, fmt='o',
    yerr=[
        ind_means - boot_low,
        boot_high - ind_means
    ],
)
axs[0].set_ylabel("First order Sobol' indices")
axs[0].set_xlabel('Input parameters')
axs[0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
_ = axs[1].errorbar(
    [1, 2, 3, 4, 5, 6, 7, 8, 9], ind_means_t, fmt='o',
    yerr=[
        ind_means_t - boot_low_t,
        boot_high_t - ind_means_t
    ],
)
axs[1].set_ylabel("Total order Sobol' indices")
axs[1].set_xlabel('Input parameters')
axs[1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.tight_layout()
plt.show()

#%% load data/models and scalers, chirp

x_scalers = joblib.load('models/x_scalers_chirp.gz')
y_scalers = joblib.load('models/y_scalers_chirp.gz')

def full_model_no_chirp(x, n_x=9, n_y=184):
    # x: (n_x, n_samples) of inputs
    # y: (n_y, n_samples) of outputs
    # note that NN torch model takes torch tensor shape (n_samples, n_x) 
    
    class feed_forward(nn.Module): 
      def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.net = nn.Sequential(
            nn.Linear(9, 64), 
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 184*2),
            nn.ReLU(),
            nn.Linear(184*2, 184)
        )
      def forward(self, X):
        return self.net(X)
      def predict(self, X):
        Y_pred = self.forward(X)
        return Y_pred

    model = torch.load("models/chirp.pt")
    
    # start by scaling inputs
    x_scaled = np.zeros(x.shape)
    print(x.shape)
    for i in range(n_x):
        x_scaled[i,:] = x_scalers[i].transform(x[i,:].reshape(-1,1)).flatten()
        
    # reshape (transpose) and make torch tensor of dtype float32
    x_tensor_scaled = torch.tensor(x_scaled.T, dtype=torch.float32)
    
    # make predictions using model
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model.to(device)
    x_tensor_scaled = x_tensor_scaled.to(device)
    model.eval()
    y_tensor_scaled = model(x_tensor_scaled)
    
    # reshape (transpose) and make numpy array from model outputs
    y_np_scaled = torch.transpose(y_tensor_scaled,0,1).cpu().detach().numpy()
    
    # unscale outputs
    y = np.zeros(y_np_scaled.shape)
    print(y_np_scaled.shape)
    for j in range(n_y):
        y[j,:] = y_scalers[j].inverse_transform(y_np_scaled[j,:].reshape(-1,1)).flatten()
     
    return y
        
#%% Sobol index estimation
from scipy.stats import uniform, norm, sobol_indices
rng = np.random.default_rng()

# a_nmc, b_nmc, c_nmc, d_nmc (assumed independent)
# graphite_diff_parameter posterior
# ep_por, neg_por, pos_por
# cap_dl_neg

dists = [norm(-2.29714210e+01,np.sqrt(414.16743966)),
         norm(-1.23599647e-02,np.sqrt(475.21734538)),
         norm(-1.09287243e+00, np.sqrt(61.12237867)),
         norm(1.62538939e+00, np.sqrt(3.8372772)),
         norm(1., np.sqrt(3.62066527e-05)),
         uniform(0.2, 0.45),
         uniform(0.2, 0.45),
         uniform(0.2, 0.45),
         norm(0.2, 0.05)
    ]

indices_chirp = sobol_indices(func = full_model_no_chirp,
                        n=2048, dists=dists, random_state=rng
                        )

boot = indices_chirp.bootstrap()
#%%
ind_means = np.mean(indices.first_order,axis=0)
boot_low = np.mean(boot.first_order.confidence_interval.low,axis=0)
boot_high = np.mean(boot.first_order.confidence_interval.high,axis=0)

ind_means_t = np.mean(indices.total_order,axis=0)
boot_low_t = np.mean(boot.total_order.confidence_interval.low,axis=0)
boot_high_t = np.mean(boot.total_order.confidence_interval.high,axis=0)

#%% plotting
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
_ = axs[0].errorbar(
    [1, 2, 3, 4, 5, 6, 7, 8, 9], ind_means, fmt='o',
    yerr=[
        ind_means - boot_low,
        boot_high - ind_means
    ],
)
axs[0].set_ylabel("First order Sobol' indices")
axs[0].set_xlabel('Input parameters')
axs[0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
_ = axs[1].errorbar(
    [1, 2, 3, 4, 5, 6, 7, 8, 9], ind_means_t, fmt='o',
    yerr=[
        ind_means_t - boot_low_t,
        boot_high_t - ind_means_t
    ],
)
axs[1].set_ylabel("Total order Sobol' indices")
axs[1].set_xlabel('Input parameters')
axs[1].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.tight_layout()
plt.show()

#%%
plt.figure()
for i in range(98):
    plt.plot(indices.first_order[i,:],'o',markersize=2,color='blue')
plt.xlabel("Input parameters")
plt.ylabel("First Order Sobol' Indices")
    
#%% 
index = 4
plt.plot(indices_chirp.first_order[index,:])
plt.xlabel("Input parameters")
plt.ylabel("First Order Sobol' Indices")
plt.title("Output "+str(index))