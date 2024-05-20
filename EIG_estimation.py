#%%
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import glob

def evaluate_log_likelihood_w_reuse(y_samples, model_evals, mean, cov):
    # note: cov, mean are of Gaussian noise term
    epsilon_generated = np.subtract(y_samples,model_evals)
    likelihood_pdf = stats.multivariate_normal.logpdf(epsilon_generated,mean,cov,allow_singular=True)
    return likelihood_pdf

def evaluate_log_likelihood_w_reuse_iid(y_samples, model_evals, mean, variances):
    epsilon_generated = np.subtract(y_samples,model_evals)
    likelihood_pdf = 0.
    for i in range(len(variances)):
        likelihood_pdf += stats.norm.logpdf(epsilon_generated[i],0.,np.sqrt(variances[i]))
    return likelihood_pdf
        
def utility_with_reuse_iid(y_vals, model_evals, n_in, n_out, eps_mean, eps_cov):
    u_d = np.zeros((n_out,))
    assert n_in==n_out, "n_in and n_out must take the same value for sample reuse"
    for i in range(n_out):
        if i%20==0:
            print(i)
        evidence = 0
        
        for j in range(n_in):   
            log_likelihood = evaluate_log_likelihood_w_reuse_iid(y_vals[i,:], model_evals[j,:], eps_mean, eps_cov)
            evidence += np.exp(log_likelihood)
            
        evidence /= n_in
        u_d[i] += evaluate_log_likelihood_w_reuse_iid(y_vals[i,:],model_evals[i,:], eps_mean, eps_cov) - np.log(evidence)
        print(u_d[i])
    return u_d

def utility_with_reuse(y_vals, model_evals, n_in, n_out, eps_mean, eps_cov):
    u_d = np.zeros((n_out,))
    assert n_in==n_out, "n_in and n_out must take the same value for sample reuse"
    for i in range(n_out):
        if i%20==0:
            print(i)
        evidence = 0
        
        for j in range(n_in):   
            log_likelihood = evaluate_log_likelihood_w_reuse(y_vals[j,:], model_evals[j,:], eps_mean, eps_cov)
            evidence += np.exp(log_likelihood)
            
        evidence /= n_in
        u_d[i] += evaluate_log_likelihood_w_reuse(y_vals[i,:],model_evals[i,:], eps_mean, eps_cov) - np.log(evidence)
    return u_d

def load_files(integer):
    folder_path = "summary_statistics_fixed_design_no_chirp"
    file_extension = f"_{integer}.npy"

    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            y = np.load(os.path.join(folder_path, filename))

    return y   

def load_files_chirp(integer):
    folder_path = "summary_statistics_fixed_design"
    file_extension = f"_{integer}.npy"

    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            y = np.load(os.path.join(folder_path, filename))
    nanIndices = [5, 7, 50, 52, 94,  95,  96,  97, 139, 140, 141, 142, 184, 185, 186, 187]
    y = np.delete(y, nanIndices)

    return y

#%% starting with non-chirp signal

n_out = 1000 # number of inner and outer loop samples (using reuse)
print("no chirp, nout = "+str(n_out))
# load in likelihood statistics (defining mean and cov of eps ~ MVN(eps_mean, eps_cov) )
n_y = 105 # dimension of y
eps_mean = np.zeros((n_y,))#np.load("likelihood_chirp/means.npy")
eps_variances = np.load("likelihood_no_chirp/vars.npy")
eps_variances[np.where(eps_variances<1e-9)]=1e-9

# load in model evaluations at non-chirp design
y = np.zeros((n_out,n_y))
model_evals = np.copy(y)
for integer in range(n_out):
    if integer%100==0:
        print(integer)
    # Load files based on the provided integer
    g = load_files(integer)
    eps = np.zeros((n_y,))
    for i in range(n_y):
        eps[i] = stats.norm.rvs(loc=0.,scale=np.sqrt(eps_variances[i]))
    #eps = stats.multivariate_normal.rvs(eps_mean_chirp, np.diag(eps_variances_chirp))
    model_evals[integer,:] = g
    y[integer,:] = g + eps

#%% compute utility 
print("Computing utility w IID assumption, no chirp")
#y = np.load("likelihood_no_chirp/y.npy")
model_evals = np.load("likelihood_no_chirp/model_evals.npy")
eps_variances = np.load("likelihood_no_chirp/vars.npy")
eps_variances[np.where(eps_variances<1e-9)]=1e-9
u_no_chirp = utility_with_reuse_iid(y, model_evals, n_out, n_out, eps_mean, eps_variances)

#%% chirp signal

n_out = 1000 # number of inner and outer loop samples (using reuse)
print("chirp, nout = "+str(n_out))
# load in likelihood statistics (defining mean and cov of eps ~ MVN(eps_mean, eps_cov) )
n_y = 209 # dimension of y
eps_mean_chirp = np.zeros((n_y,))#np.load("likelihood_chirp/means.npy")
eps_variances_chirp = np.load("likelihood_chirp/var.npy")

# load in model evaluations at non-chirp design
y = np.zeros((n_out,n_y))
model_evals = np.copy(y)
for integer in range(n_out):
    if integer%5==0:
        print(integer)
    # Load files based on the provided integer
    g = load_files_chirp(integer)
    eps = np.zeros((n_y,))
    for i in range(n_y):
        eps[i] = stats.norm.rvs(loc=0.,scale=np.sqrt(eps_variances_chirp[i]))
    #eps = stats.multivariate_normal.rvs(eps_mean_chirp, np.diag(eps_variances_chirp))
    model_evals[integer,:] = g
    y[integer,:] = g + eps

#%% compute utility 
n_out = 1000
n_y = 209
print("Computing utility w IID assumption, chirp")
y = np.load("likelihood_chirp/y.npy")
model_evals = np.load("likelihood_chirp/model_evals.npy")
eps_variances = np.load("likelihood_chirp/var.npy")
eps_variances[np.where(eps_variances<1e-6)]=1e-6
u_chirp = utility_with_reuse_iid(y, model_evals, n_out, n_out, eps_mean_chirp, eps_variances)

#%% chirp signal

# load in likelihood statistics (defining mean and cov of eps ~ MVN(eps_mean, eps_cov) )
eps_mean_chirp = np.load("likelihood_chirp/means.npy")
eps_cov_chirp = np.diag(np.load("likelihood_chirp/var.npy"))

# load in model evaluations at non-chirp design
n_out = 100 # number of inner and outer loop samples (using reuse)
n_y = 209 # dimension of y
y = np.zeros((n_out,n_y))
model_evals = np.copy(y)
for integer in range(n_out):
    if integer%5==0:
        print(integer)
    # Load files based on the provided integer
    g = load_files_chirp(integer)
    eps = stats.multivariate_normal.rvs(eps_mean_chirp, eps_cov_chirp)
    model_evals[integer,:] = g
    y[integer,:] = g + eps
