import numpy as np 
import torch
from torch import nn
from ot.utils import unif


def sinkhorn_loss(mu, nu, C, epsilon=1e-3, n_iter=100, rho=1, tau=-0.8, thr=1E-6,
                  device=torch.device("cpu"), return_coupling=False):
    """
    Copied from: XXX Not putting names here to avoid automattic rejectwion

    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop

    Parameters
        rho = 1  # (.5) **2          # unbalanced transport
        tau = -.8  # nesterov-like acceleration
        thr:  stopping criterion
    """
    n_a = mu.shape[0]
    n_b = nu.shape[0]

    if n_a != n_a:
        raise IOError("The two input vectors do not have the same size."
                      "a has %s samples, whereas b has %s." % (n_a, n_a))
    else:
        n = n_b

    lam = rho / (rho + epsilon)  # Update exponent

    # # both marginals are fixed with equal weights
    # mu = Variable((1. / n * torch.FloatTensor(n).fill_(1)).type(torch.float64),
    #               requires_grad=False).to(device)
    # nu = Variable((1. / n * torch.FloatTensor(n).fill_(1)).type(torch.float64),
    #               requires_grad=False).to(device)

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1E-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    # Initiallization
    u = torch.zeros_like(mu)#.type(torch.float64)
    v = torch.zeros_like(nu)#.type(torch.float64)
    err = 0.0
    # to check if algorithm terminates because of threshold or max iterations reached
    actual_nits = 0
    for i in range(n_iter):
        # useful to check the update
        u1 = u
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thr).data.numpy():
            break

    # Transport plan pi = diag(a)*K*diag(b)
    Pi = torch.exp(M(u, v))
    # Sinkhorn cost
    cost = torch.sum(Pi.mul(C))

    if return_coupling:
        return Pi, cost
    else:
        return cost

### Aux
def compute_cost_matric(u, v, C, epsilon):
    "Modified cost for logarithmic updates"
    "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
    return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon


def lse(A):
    "log-sum-exp"
    return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN