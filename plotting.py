#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:55:26 2024

Scripts for plotting EIG results, chirp vs. non-chirp

@author: me-tcoons
"""

#%% imports

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.dpi'] = 600
plt.style.use('bmh')
u_no_chirp = np.load("u_no_chirp_nout1K_nin5K.npy")
u_chirp = np.load("u_chirp_nout1K_nin5K.npy")
u_chirp = u_chirp[np.where(u_chirp>-10.)]

#%% stacked histograms

fig, axs = plt.subplots(2)
axs[0].hist(u_chirp, bins=20,color='skyblue',label='Chirp + HPPC Utilities')
axs[0].set_title('Chirp + HPPC Bayesian Utility')
axs[0].vlines(np.mean(u_chirp),0,300,linestyles='dashed',color='gray')
axs[1].hist(u_no_chirp,bins=20, color='tomato', label='HPPC Utilities')
axs[1].vlines(np.mean(u_no_chirp),0,600,color='gray',linestyles='dashed')
axs[1].set_title('HPPC Bayesian Utility')

#%% overlapping histograms
plt.hist(u_chirp, bins=20,alpha=0.5,color='skyblue',label='Chirp + HPPC Utilities')
plt.vlines(np.mean(u_chirp),0,400,color='blue',linestyles='dashed',label='Chirp + HPPC EIG')
plt.hist(u_no_chirp, bins=20,alpha=0.5,color='tomato',label='HPPC Utilities')
plt.vlines(np.mean(u_no_chirp),0,400,color='red',linestyles='dashed',label='HPPC EIG')
plt.legend()
plt.title('EIG Results for Chirp vs. HPPC + Chirp Input Signals')
plt.xlabel('Utility (log-ratio of likelihood to evidence)')
plt.ylabel('Frequency')

#%% EIG alone w standard error
yerrs = [np.std(u_no_chirp)/np.sqrt(len(u_no_chirp)), np.std(u_chirp)/np.sqrt(len(u_chirp))]
plt.bar([0.2,1],[np.mean(u_no_chirp), np.mean(u_chirp)],0.5,color=['tomato', 'skyblue'],label=['HPPC','HPPC+Chirp'])
#plt.errorbar([0.2,1],[np.mean(u_no_chirp), np.mean(u_chirp)],yerr=yerrs)
plt.errorbar(0.2, np.mean(u_no_chirp),yerr=yerrs[0], capsize=2,color='black',label='MC standard error',markeredgewidth=2)
plt.errorbar(1, np.mean(u_chirp),yerr=yerrs[1], color='black', capsize=2,markeredgewidth=2)
plt.title('EIG Results  and Errors for Chirp vs. HPPC + Chirp Input Signals')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.legend()
plt.ylabel('EIG')

#%% eig efficiency
u_chirp = u_chirp/10500
u_no_chirp = u_no_chirp/10000
#%%
plt.hist(u_chirp, bins=20,alpha=0.5,color='skyblue',label='Chirp + HPPC Utilities')
plt.vlines(np.mean(u_chirp),0,400,color='blue',linestyles='dashed',label='Chirp + HPPC EIG')
plt.hist(u_no_chirp, bins=20,alpha=0.5,color='tomato',label='HPPC Utilities')
plt.vlines(np.mean(u_no_chirp),0,400,color='red',linestyles='dashed',label='HPPC EIG')
plt.legend()
plt.title('EIG Efficiency Results for Chirp vs. HPPC + Chirp Input Signals')
plt.xlabel('Utility Efficiency (log-ratio)')
plt.ylabel('Frequency')
plt.ticklabel_format(axis='x', style='sci',scilimits=(0,0))

#%%
yerrs = [np.std(u_no_chirp)/np.sqrt(len(u_no_chirp)), np.std(u_chirp)/np.sqrt(len(u_chirp))]
plt.bar([0.2,1],[np.mean(u_no_chirp), np.mean(u_chirp)],0.5,color=['tomato', 'skyblue'],label=['HPPC','HPPC+Chirp'])
#plt.errorbar([0.2,1],[np.mean(u_no_chirp), np.mean(u_chirp)],yerr=yerrs)
plt.errorbar(0.2, np.mean(u_no_chirp),yerr=yerrs[0], capsize=2,color='black',label='MC standard error',markeredgewidth=2)
plt.errorbar(1, np.mean(u_chirp),yerr=yerrs[1], color='black', capsize=2,markeredgewidth=2)
plt.title('EIG Efficiency Results and Errors for Chirp vs. HPPC + Chirp Input Signals')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.legend()
plt.ylabel('EIG Efficiency')
plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))