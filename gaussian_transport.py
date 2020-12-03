import numpy as np
import torch
from torch import nn


def root_p_matrix(X, p=0.5):
    U, S, V = torch.svd(X)
    S_root = S.clamp(min=0).pow(p)
    return U.mm(torch.diag(S_root)).mm(V.t())


def plan_gaussian_transport(covariance_1, covariance_2, t=0.5):
    X_root = root_p_matrix(covariance_1, p=0.5)
    X_inv_root = root_p_matrix(covariance_1, p=-0.5)

    G = X_root.matmul(covariance_2.matmul(X_root))
    G_sqrt = root_p_matrix(G, p=t)
    M = X_inv_root.matmul(G_sqrt.matmul(X_inv_root))    
    return M


def l2_transport(x, mean_1, mean_2, covariance_1, covariance_2):
    M = plan_gaussian_transport(covariance_1, covariance_2)
    return mean_2 + (x - mean_1).mm(M)


def bures_Wasserstein(a_cov, b_cov):
    U, S, V = torch.svd(a_cov)
    S_root = S.clamp(min=0).pow(0.5)
    root_a_cov = U.mm(torch.diag(S_root)).mm(V.t())
    # Interactions
    cov_prod = root_a_cov.mm(b_cov).mm(root_a_cov)
    return cov_prod


def bures_Wasserstein_distance(a_cov, b_cov, a_mean=None, b_mean=None):
    mseloss = nn.MSELoss(reduction='sum')
    if (a_mean is not None) and (b_mean is not None): 
        mean_diff_squared = mseloss(a_mean, b_mean)
    else:
        mean_diff_squared = 0.0

    tr_a_cov = torch.trace(a_cov)
    tr_b_cov = torch.trace(b_cov)

    cov_prod = bures_Wasserstein(a_cov, b_cov)

    _, S, _ = torch.svd(cov_prod)
    var_overlap = S.clamp(min=0).pow(0.5).sum()

    dist = (mean_diff_squared + tr_a_cov + tr_b_cov - 2 * var_overlap)

    return dist