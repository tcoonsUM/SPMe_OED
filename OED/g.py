#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:56:41 2024

@author: me-tcoons
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def g(x):
    
    # initialize device 
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    n_stats = 27 # number of summary stats that are repeated
    n_repeats = 5 # number of repeats
    n_y = n_stats*n_repeats # 135
    n_x = x.shape[1] # 17
    n_obs = x.shape[0]
    
    # load in the necessary scalers
    x_scalers_npz = np.load("x_scalers.npz", allow_pickle=True) 
    y_scalers_npz = np.load("y_scalers.npz", allow_pickle=True) 
    
    # define model and its parameters
    class feed_forward_bn(nn.Module): 
      def __init__(self, n_x, n_hidden, n_y, seed=42, dropout_rate=0.):
        super().__init__()
        torch.manual_seed(seed)
        self.l1 = nn.Linear(n_x, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.l3 = nn.Linear(n_hidden, n_y)
        self.bn3 = nn.BatchNorm1d(n_y)
        self.dropout = nn.Dropout(dropout_rate) 
      def forward(self, X):
          out = F.relu(self.bn1(self.l1(X)))
          out2 = F.relu(self.bn2(self.l2(out)))
          out3 = self.dropout(self.bn3(self.l3(out2)))
          return out3
      def predict(self, X):
        Y_pred = self.forward(X)
        return Y_pred
    
    model = feed_forward_bn(n_x, n_x*8, n_y, seed=43, dropout_rate=0.25).to(device)
    model.load_state_dict(torch.load("model_state_dict.pt", weights_only=False, map_location=device))
    
    # scale x data
    X_scaled = torch.zeros(x.shape)
    for i in range(n_x):
        key = x_scalers_npz.files[i]
        X_scaled[:,i] = torch.tensor(x_scalers_npz[key].item().transform(x[:,i].reshape(-1,1)).flatten())
    
    # evaluate model
    model.eval()
    with torch.no_grad():
        X_scaled = X_scaled.to(device)
        pred = model(X_scaled)
        
    # inverse transform predictions according to y_scalers
    pred_scaled = torch.zeros(pred.shape)
    for j in range(n_y):
        key = y_scalers_npz.files[j]
        pred_scaled[:,j] = torch.tensor(y_scalers_npz[key].item().inverse_transform(pred[:,j].reshape(-1,1)).flatten())
        
    return pred_scaled