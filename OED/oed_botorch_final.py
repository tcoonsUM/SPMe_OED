#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 09:34:57 2025

@author: me-tcoons
"""

import utils_oed as ute
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import numpy as np
from tabulate import tabulate

# Min-max scaling functions
def min_max_scale(X, lower_bounds, upper_bounds):
    return (X - lower_bounds) / (upper_bounds - lower_bounds)

def inverse_min_max_scale(X_scaled, lower_bounds, upper_bounds):
    return X_scaled * (upper_bounds - lower_bounds) + lower_bounds

# Your bo_friendly_objective function
def bo_friendly_objective(d0, d1, d2, d3, d4, d5, d6, d7, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz):
    # IMPORTANT NOTE:
    #  this will accept d0 then d1=d0*d1' as first two inputs
    #               and d6 then d7=d6/d7' as last two inputs
    #
    #  the eig function takes d0, d1', ..., d6, d7' as inputs
    #               where d1' = d1/d0 and d7'=d7/d6
    
    d1prime = d1/d0
    d7prime = d7/d6
    d = torch.tensor([[d0, d1prime, d2, d3, d4, d5, d6, d7prime]])  # Should have [1,8] shape
    utilities = ute.eig_mp(d, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz)
    return utilities.mean()

def sample_uniformly_from_bounds(bounds, n_samples):
    lower_bounds = bounds[0]
    upper_bounds = bounds[1]
    samples = lower_bounds + (upper_bounds - lower_bounds) * torch.rand((n_samples, bounds.size(1)))
    return samples

def evaluate_objective_for_initial_points(initial_points, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz):
    """Evaluate the bo_friendly_objective for initial points."""
    initial_Y = []
    for point in initial_points:
        initial_Y.append(
            bo_friendly_objective(
                *point.tolist(), n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz
            )
        )
    return torch.tensor(initial_Y, dtype=torch.float64).unsqueeze(-1)  # Ensure shape is [n_points, 1]

if __name__ == "__main__":
    # Load the required files
    x_scalers_npz = np.load("x_scalers.npz")
    y_scalers_npz = np.load("y_scalers.npz")
    corrs_clustered_npz = np.load("corrs_clustered_list.npz")
    clusters_inds_npz = np.load("clusters_list.npz")  # can access keywords via clusters_npz.files
    rel_std = np.load("relative_stds.npy")

    # Define OED parameters
    n_in = int(1e5)
    n_out = int(1e5)
    n_theta = 8
    n_y = 135

    # Define bounds
    bounds = torch.tensor([[0.1, 72.0, 100.0, 0.1, 1.0, 10.0, 0.001, 0.25],  # lower bounds
                           [2.0, 360.0, 1800.0, 2.0, 20.0, 600.0, 0.01, 1.0]])  # upper bounds
    
    # Train data
    n_init = 10
    initial_points = sample_uniformly_from_bounds(bounds, n_init)
    initial_Y = evaluate_objective_for_initial_points(
        initial_points, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz
    )
    
    # Apply min-max scaling to train_X
    train_X_scaled = min_max_scale(initial_points, bounds[0], bounds[1]).to(torch.float64)
    train_Y = initial_Y  # Shape [n_points, 1]

    # Define and fit GP model
    gp = SingleTaskGP(train_X_scaled, train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)


    # Optimization loop
    n_iterations = 20
    train_X_unscaled = []
    train_Y_unscaled = []
    for iteration in range(n_iterations):
        # Step 3: Fit a GP model
        gp = SingleTaskGP(train_X_scaled, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
    
        # Step 4: Define the constrained acquisition function
        qEI = qExpectedImprovement(
            model=gp,
            best_f=train_Y.max().item(),
        )
        ei = ExpectedImprovement(model=gp, best_f=train_Y.max().item())
    
        # Step 5: Optimize the acquisition function
        candidate_scaled, _ = optimize_acqf(
            acq_function=ei,
            bounds=torch.stack([torch.zeros(train_X_scaled.size(1)), torch.ones(train_X_scaled.size(1))]),
            q=1,
            num_restarts=10,
            raw_samples=100,
        )
        
        # Scale candidate back to original bounds
        candidate = inverse_min_max_scale(candidate_scaled, bounds[0], bounds[1])
    
        # Evaluate objective
        candidate_y = torch.tensor(bo_friendly_objective(
            *candidate[0], n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz
        ),dtype=torch.float64).reshape(1,1)
        
        # Step 6: Update training data
        train_X_scaled = torch.cat([train_X_scaled, candidate_scaled], dim=0).to(torch.float64)
        train_Y = torch.cat([train_Y, candidate_y], dim=0)
        
        # Save the unscaled candidate (original coordinates)
        candidate_orig = candidate
        candidate_orig[:,1] = candidate_orig[:,1]/candidate_orig[:,0]
        candidate_orig[:,7] = candidate_orig[:,7]/candidate_orig[:,6]
        train_X_unscaled.append(candidate_orig)
        train_Y_unscaled.append(candidate_y)
                
        # Inside the optimization loop (after Step 6: Update training data and before the next iteration)
        headers = [f"x{i}" for i in range(candidate_orig.size(1))] + ["y"]
        data = torch.cat([candidate_orig, candidate_y], dim=1).tolist()  # Combine new candidate X and Y for tabulation
        table = tabulate(data, headers, tablefmt="grid")
        print(table)
        with open("oed_results.txt", "w") as f:
            f.write(table)
    
    # Convert history lists to tensors for saving
    train_X_unscaled_tensor = torch.cat(train_X_unscaled, dim=0)
    train_Y_unscaled_tensor = torch.cat(train_Y_unscaled, dim=0)
    
    # Save train_X and train_Y history
    torch.save(train_X_unscaled_tensor, "train_X_unscaled.pt")
    torch.save(train_Y_unscaled_tensor, "train_Y_unscaled.pt")
    
    print("Optimization complete.")
    
    
