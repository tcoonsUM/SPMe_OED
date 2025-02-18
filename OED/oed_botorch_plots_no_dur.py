#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:55:32 2025

@author: me-tcoons
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import PosteriorMean
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from botorch.models.transforms import Standardize

# Min-max scaling functions
def min_max_scale(X, lower_bounds, upper_bounds):
    return (X - lower_bounds) / (upper_bounds - lower_bounds)

def inverse_min_max_scale(X_scaled, lower_bounds, upper_bounds):
    return X_scaled * (upper_bounds - lower_bounds) + lower_bounds

def sample_uniformly_from_bounds(bounds, n_samples):
    lower_bounds = bounds[0]
    upper_bounds = bounds[1]
    samples = lower_bounds + (upper_bounds - lower_bounds) * torch.rand((n_samples, bounds.size(1)))
    return samples

if __name__ == "__main__":
    # Load the required files
    x_scalers_npz = np.load("x_scalers.npz")
    y_scalers_npz = np.load("y_scalers.npz")
    corrs_clustered_npz = np.load("corrs_clustered_list.npz")
    clusters_inds_npz = np.load("clusters_list.npz")  # can access keywords via clusters_npz.files
    rel_std = np.load("relative_stds.npy")

    # Define OED parameters
    n_theta = 8
    n_y = 135

    # Define bounds
    bounds_orig = torch.tensor([[0.1, 72.0, 100.0, 0.1, 1.0, 10.0, 0.001, 0.25],  # lower bounds
                           [2.0, 360.0, 1800.0, 2.0, 20.0, 600.0, 0.01, 1.0]])  # upper bounds
    
    # Warm start with legacy data
    train_X_unscaled_legacy = torch.load("train_X_unscaled_6e3.pt")
    train_Y_unscaled_legacy = torch.load("train_Y_unscaled_6e3.pt")#torch.load("train_Y_unscaled_legacy_dur.pt")
    
    # Apply min-max scaling to train_X
    train_X_new_coords = train_X_unscaled_legacy.clone()
    train_X_new_coords[:,1] = train_X_new_coords[:,0]*train_X_new_coords[:,1]
    train_X_new_coords[:,7] = train_X_new_coords[:,6]*train_X_new_coords[:,7]
    train_X_scaled = min_max_scale(train_X_new_coords, bounds_orig[0], bounds_orig[1]).to(torch.float64)
    train_Y = train_Y_unscaled_legacy  # Shape [n_points, 1]

    # Define and fit GP model
    gp = SingleTaskGP(train_X_scaled, train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    # Assuming all necessary data (train_X_scaled, train_Y, bounds, etc.) is loaded

    # 1. Find the best observed point
    best_idx = train_Y.argmax()
    best_observed_point = train_X_scaled[best_idx].unsqueeze(0)
    best_observed_value = train_Y[best_idx]
    
    # 2. Optimize the posterior mean to find the predicted maximizer
    posterior_mean_acq = PosteriorMean(gp)
    
    # Define bounds for optimization
    bounds = torch.stack([
        torch.zeros(train_X_scaled.size(1), dtype=torch.float64),  # Since train_X_scaled is in [0, 1]
        torch.ones(train_X_scaled.size(1), dtype=torch.float64)
    ])
    
    # Optimize the posterior mean
    predicted_max_point, predicted_max_value = optimize_acqf(
        acq_function=posterior_mean_acq,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=100,
    )
    
    # 3. Evaluate the GP at the best observed point
    with torch.no_grad():
        posterior_best = gp.posterior(best_observed_point)
        predicted_mean = posterior_best.mean.item()
        predicted_variance = posterior_best.variance.item()
    
        posterior_predicted_max = gp.posterior(predicted_max_point)
        predicted_max_variance = posterior_predicted_max.variance.item()
    
    # 4. Display the comparison
    print("--- Comparison of GP Predictions ---")
    print(f"Best Observed Point (scaled): {best_observed_point}")
    print(f"Observed Value at Best Point: {best_observed_value.item()}")
    print(f"GP Predicted Mean at Best Observed Point: {predicted_mean}")
    print(f"GP Predicted Variance at Best Observed Point: {predicted_variance}")
    
    print("\nPredicted Maximizer (Posterior Mean):")
    print(f"Predicted Max Point (scaled): {predicted_max_point.squeeze()}")
    print(f"Predicted Max Value: {predicted_max_value.item()}")
    print(f"Predicted Variance at Predicted Max Point: {predicted_max_variance}")
    
    # 5. Visualize the effect of each dimension
    num_points = 100  # Number of points to evaluate per dimension
    best_observed_true_x_all = torch.zeros(train_X_scaled.size(1))
    for dim in range(train_X_scaled.size(1)):
        X_grid = torch.linspace(0, 1, num_points, dtype=torch.float64)  # Vary within [0, 1]
        X_eval = best_observed_point.repeat(num_points, 1)
        X_eval[:, dim] = X_grid  # Vary one dimension
    
        with torch.no_grad():
            posterior = gp.posterior(X_eval)
            mean = posterior.mean.squeeze().numpy()
            std = posterior.variance.sqrt().squeeze().numpy()
    
        # Convert scaled values back to original scale for plotting
        X_grid_true = inverse_min_max_scale(X_grid, bounds_orig[0,dim], bounds_orig[1,dim])
        best_observed_true_x = inverse_min_max_scale(best_observed_point[0, dim], bounds_orig[0,dim], bounds_orig[1,dim])
    
        if dim==1:
            best_observed_true_x_full = inverse_min_max_scale(best_observed_point[0, :], bounds_orig[0,:], bounds_orig[1,:])
            X_grid_true = X_grid_true/best_observed_true_x_full[0]
            best_observed_true_x = best_observed_true_x/best_observed_true_x_full[0]
        elif dim==7:
            best_observed_true_x_full = inverse_min_max_scale(best_observed_point[0, :], bounds_orig[0,:], bounds_orig[1,:])
            X_grid_true = X_grid_true/best_observed_true_x_full[6]
            best_observed_true_x = best_observed_true_x/best_observed_true_x_full[6]

        best_observed_true_x_all[dim] = best_observed_true_x
    
        # Plotting
        plt.figure(figsize=(8, 4))
        plt.plot(X_grid_true.numpy(), mean, label='Posterior Mean', color='blue')
        plt.fill_between(X_grid_true.numpy(), mean - 2 * std, mean + 2 * std, color='blue', alpha=0.2, label='±2 Std Dev')
        plt.scatter(
            best_observed_true_x.item(),  # X-coordinate of the best observed point (true scale)
            best_observed_value.item(),   # Y-coordinate of the best observed value
            color='red', marker='*', s=50, zorder=3, label='Best Observed Point'
        )
        plt.title(f'Effect of Dimension {dim}')
        plt.xlabel(f'Dimension {dim} (true scale)')
        plt.ylabel('GP Prediction')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    #%% all in one subplot
    
    num_points = 100  # Number of points to evaluate per dimension
    best_observed_true_x_all = torch.zeros(train_X_scaled.size(1))
    
    fig1, axes1 = plt.subplots(4, 2, figsize=(12, 16), sharey=True)  # Shared y-axis
    fig2, axes2 = plt.subplots(4, 2, figsize=(12, 16), sharey=False)  # Independent y-axis
    
    axes1 = axes1.flatten()
    axes2 = axes2.flatten()
    
    for dim in range(train_X_scaled.size(1)):
        X_grid = torch.linspace(0, 1, num_points, dtype=torch.float64)  # Vary within [0, 1]
        X_eval = best_observed_point.repeat(num_points, 1)
        X_eval[:, dim] = X_grid  # Vary one dimension
    
        with torch.no_grad():
            posterior = gp.posterior(X_eval)
            mean = posterior.mean.squeeze().numpy()
            std = posterior.variance.sqrt().squeeze().numpy()
    
        # Convert scaled values back to original scale for plotting
        X_grid_true = inverse_min_max_scale(X_grid, bounds_orig[0, dim], bounds_orig[1, dim])
        best_observed_true_x = inverse_min_max_scale(best_observed_point[0, dim], bounds_orig[0, dim], bounds_orig[1, dim])
    
        if dim == 1:
            best_observed_true_x_full = inverse_min_max_scale(best_observed_point[0, :], bounds_orig[0, :], bounds_orig[1, :])
            X_grid_true = X_grid_true / best_observed_true_x_full[0]
            best_observed_true_x = best_observed_true_x / best_observed_true_x_full[0]
        elif dim == 7:
            best_observed_true_x_full = inverse_min_max_scale(best_observed_point[0, :], bounds_orig[0, :], bounds_orig[1, :])
            X_grid_true = X_grid_true / best_observed_true_x_full[6]
            best_observed_true_x = best_observed_true_x / best_observed_true_x_full[6]
    
        best_observed_true_x_all[dim] = best_observed_true_x
    
        # Plot in both figure grids
        for ax in [axes1[dim], axes2[dim]]:
            ax.plot(X_grid, mean, label='Posterior Mean', color='blue')
            ax.fill_between(X_grid, mean - 2 * std, mean + 2 * std, color='blue', alpha=0.2, label='±2 Std Dev')
            ax.scatter(
                best_observed_point[0, dim],#best_observed_true_x.item(),  # X-coordinate of the best observed point (true scale)
                best_observed_value.item(),   # Y-coordinate of the best observed value
                color='red', marker='*', s=50, zorder=3, label='Best Observed Point'
            )
            ax.set_title(f'Effect of Dimension {dim}')
            ax.set_xlabel(f'Dimension {dim} (scaled)')
            ax.set_ylabel('EIG per sec')
            #ax.set_yscale('log')
            ax.legend()
            ax.grid(True)
    
    # Show the figures
    fig1.suptitle("Effect of Individual Design Variables", fontsize=16)
    fig2.suptitle("Effect of Individual Design Variables", fontsize=16)
    
    fig1.tight_layout()
    fig2.tight_layout()

    plt.show()
    #%%    
    # import itertools
    # def plot_marginals(gp, X_train, n_dimensions=8, num_points=100):
    #     # X_train shape: (n_samples, n_dimensions)
    #     n_dimensions = X_train.shape[1]
    #     X_mean = X_train.mean(dim=0)
        
    #     for dim in range(n_dimensions):
    #         # Create a grid for this dimension
    #         dim_min, dim_max = X_train[:, dim].min(), X_train[:, dim].max()
    #         grid = torch.linspace(dim_min, dim_max, num_points)
            
    #         # Create a full grid where this dimension varies and others are fixed at their mean
    #         X_grid = X_mean.expand(num_points, n_dimensions).clone()
    #         X_grid[:, dim] = grid
            
    #         # Get the posterior predictions
    #         posterior = gp.posterior(X_grid)
    #         mean = posterior.mean.detach().numpy()
    #         lower, upper = posterior.mvn.confidence_region()
    
    #         # Plot
    #         plt.figure()
    #         plt.plot(grid.numpy(), mean, label='Mean')
    #         plt.fill_between(grid.numpy(), lower.numpy(), upper.numpy(), alpha=0.2, label='±2 SD')
    #         plt.title(f'Marginal Effect of Dimension {dim+1}')
    #         plt.xlabel(f'Dimension {dim+1} values')
    #         plt.ylabel('GP Mean Prediction')
    #         plt.legend()
    #         plt.show()
    

