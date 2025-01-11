import utils_oed as ute
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import numpy as np
from tabulate import tabulate

# Your bo_friendly_objective function
def bo_friendly_objective(d0, d1, d2, d3, d4, d5, d6, d7, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz):
    # this will accept d0 then d1=d0*d1' as first two inputs
    # the eig function takes d0, d1', ... as inputs
    d1prime = d1/d0
    d = torch.tensor([[d0, d1prime, d2, d3, d4, d5, d6, d7]])  # Should have [1,8] shape
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
    n_in = int(1e2)
    n_out = int(1e2)
    n_theta = 8
    n_y = 135

    # Define bounds
    bounds = torch.tensor([[0.1, 72.0, 100.0, 0.1, 1.0, 10.0, 0.001, 25.0],  # lower bounds
                           [2.0, 360.0, 1800.0, 2.0, 20.0, 600.0, 0.01, 1000.0]])  # upper bounds
    
    # inequality_constraints = [
    # ([0, 1], [1.0, -1.0], 360 - 72),  # d0 * d1 should be between 72 and 360
    # ([6, 7], [1.0, -1.0], 1 - 0.25)  # d6 * d7 should be between 0.25 and 1
    # ]

    # # Generate feasible initial points
    # initial_points = generate_feasible_initial_points(
    #     n_points=2,  # Number of initial points
    #     bounds=bounds,
    #     constraint_function=constraint_function,
    # )
    
    # # Evaluate the objective for the initial points
    # initial_Y = evaluate_objective_for_initial_points(
    #     initial_points, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz
    #)
    
    # Train data
    n_init = 2
    initial_points = sample_uniformly_from_bounds(bounds, n_init)
    initial_Y = evaluate_objective_for_initial_points(
        initial_points, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz
    )
    train_X = initial_points.to( torch.float64 )
    train_Y = initial_Y  # Shape [n_points, 1]
    
    # Define and fit GP model
    gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    #%%
    # Optimization loop
    n_iterations = 5
    for iteration in range(n_iterations):
        # Step 3: Fit a GP model
        gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
    
        # Step 4: Define the constrained acquisition function
        qEI = qExpectedImprovement(
            model=gp,
            best_f=train_Y.max().item(),
        )
        ei = ExpectedImprovement(model=gp, best_f=train_Y.max().item())
    
        # Step 5: Optimize the acquisition function
        candidate, _ = optimize_acqf(
            acq_function=ei,
            bounds=torch.stack([torch.zeros(train_X.size(1)), torch.ones(train_X.size(1))]),
            q=1,
            num_restarts=10,
            raw_samples=100,
        )
    
        # Scale candidate back to original bounds
        candidate = candidate * (bounds[1] - bounds[0]) + bounds[0]
    
        # Evaluate objective
        candidate_y = torch.tensor(bo_friendly_objective(
            *candidate[0], n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz
        ),dtype=torch.float64).reshape(1,1)
        
        # Step 6: Update training data
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, candidate_y], dim=0)
                
        # Print table of results
        headers = [f"x{i}" for i in range(train_X.size(1))] + ["y"]
        data = torch.cat([train_X, train_Y], dim=1).tolist()  # Combine train_X and train_Y for tabulation
        print(tabulate(data, headers, tablefmt="grid"))
    
    print("Optimization complete.")
    
    # # n_init = 5
    # # X_train = torch.rand(n_init, bounds.size(1)) * (bounds[1] - bounds[0]) + bounds[0]
    # # X_train = X_train.to( torch.float64 )
    # # y_train = torch.tensor([bo_friendly_objective(*x, n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz)
    # #                         for x in X_train.numpy()]).unsqueeze(-1)

    # # Define acquisition function
    # # qEI = qExpectedImprovement(
    # #     model=gp,
    # #     best_f=train_Y.max(),
    # #     objective=ConstrainedMCObjective(objective=constrained_obj, constraints=[constraint_function])
    # # )
    
    # # Define the acquisition function using ExpectedImprovement
    # ei = ExpectedImprovement(
    #     model=gp,
    #     best_f=train_Y.max().item()
    # )

    # # Optimize acquisition function
    # batch_size = 1
    # num_restarts = 10
    # raw_samples = 100
    # X_next, _ = optimize_acqf(
    #     acq_function=ei,
    #     bounds=bounds,
    #     q=batch_size,
    #     num_restarts=num_restarts,
    #     raw_samples=raw_samples,
    # )

    # print(f"Next suggested point: {X_next}")

# # Constraint function (must return a tensor of constraints; negative = satisfied)
# def constraint_function(X):
#     d0, d1, d2, d3, d4, d5, d6, d7 = X.unbind(dim=-1)
#     constraint1 = 72 - d0 * d1  # dim0 constraint: d0 * d1 >= 72
#     constraint2 =  d0 * d1 - 360
#     constraint3 = 0.25 - d6 * d7   # dim1 constraint: d6 * d7 >= 0.25
#     constraint4 = d6 * d7 - 1
#     return torch.stack([constraint1, constraint2, constraint3, constraint4], dim=-1)

# # Feasibility function for the ConstrainedMCObjective
# def feasibility(samples):
#     constraints = constraint_function(samples)
#     return (constraints <= 0).all(dim=-1).float()  # Feasible if all constraints are <= 0

# # Define the constrained objective
# def constrained_obj(samples):
#     return feasibility(samples) * torch.tensor([
#         bo_friendly_objective(*sample.tolist(), n_in, n_out, rel_std, clusters_inds_npz, corrs_clustered_npz)
#         for sample in samples
#     ])


# def generate_feasible_initial_points(n_points, bounds, constraint_function,seed=42):
#     """Generate initial points satisfying the constraints."""
#     feasible_points = []
#     np.random.seed(seed)
    
#     while len(feasible_points) < n_points:
#         # Generate random points within bounds
#         random_point = torch.tensor([np.random.uniform(low, high) for low, high in zip(bounds[0], bounds[1])]).unsqueeze(0)
        
#         # Check feasibility
#         if feasibility(random_point):
#             feasible_points.append(random_point.squeeze(0))
    
#     return torch.stack(feasible_points)


