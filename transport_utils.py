import numpy as np 
import torch
from torch import nn
from ot.lp import emd
from ot.utils import unif
from ot.bregman import (
    sinkhorn_stabilized, 
    greenkhorn, 
    sinkhorn_epsilon_scaling,
    sinkhorn_knopp
)
from manifold import Mobius
from sinkhorn import sinkhorn_loss
from base import acosh, atanh, asinh


def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1) 
    y_lin = y.unsqueeze(0) 
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c


def base_poincare_matrix(u, v):
    norm_u = 1 - cost_matrix(torch.zeros_like(u), u) 
    norm_v = 1 - cost_matrix(torch.zeros_like(v), v) 
    C = cost_matrix(u, v) 
    return 1. + 2 * C / (norm_u * norm_v)


def poincare_matrix(u, v, manifold, s=1.):
    x_col = u.unsqueeze(1) 
    y_lin = v.unsqueeze(0) 
    return 2 * s * atanh(manifold.add(-x_col, y_lin).norm(dim=-1) / s)    


def cost_matrix_hyperbolic(x, y, manifold=None, detach_x=True, detach_y=True):
    u, v = x, y   
    manifold = Mobius() if manifold is not None else manifold
    ## Hack
    try:
        if detach_x:
            u.detach_()
    except:
        pass
    try:
        if detach_y:
            v.detach_() 
    except:
        pass           
 
    s = manifold.s
    #M = acosh(base_poincare_matrix(u, v, s=s))
    M = poincare_matrix(u, v, manifold=manifold, s=s)             
    return M 


def cost_normalization(M, normalization='max'):
    if normalization == 'max':
        return M / M.max()
    elif normalization == 'median':
        return M / M.median()
    elif normalization == 'log':
        return torch.log(1. + M)
    elif normalization == "loglog":
        return torch.log(1 + torch.log(1 + M))
    else:
        return M


def sinkhorn_cost(x, y, reg_ot=1., nx=None, ny=None, 
                  ys=None, yt=None,
                  n_iter=100, manifold=None, 
                  normalization='max', wrapped_function=None, 
                  detach_x=False, detach_y=True,
                  is_hyperbolic=False, match_targets=False):
    
    nx = len(x) if nx is None else nx
    ny = len(y) if ny is None else ny
    
    a = torch.FloatTensor(unif(nx)).detach()
    b = torch.FloatTensor(unif(ny)).detach()
    
    M = compute_cost(x, y, manifold=manifold, 
                     ys=ys, yt=yt, match_targets=match_targets,
                     normalization=normalization, wrapped_function=wrapped_function, 
                     detach_x=detach_x, detach_y=detach_y,
                     is_hyperbolic=is_hyperbolic)
    return sinkhorn_loss(a, b, M, epsilon=reg_ot, 
                         n_iter=n_iter, return_coupling=True)


def sinkhorn_normalized(x, y, reg_ot, nx=None, ny=None, n_iter=100, 
                        ys=None, yt=None, match_targets=False,
                        normalization='max', wrapped_function=None, 
                        detach_x=False, detach_y=True,
                        manifold=None, is_hyperbolic=False):
    Gxy, Wxy = sinkhorn_cost(x, y, reg_ot, 
                             nx=nx, ny=ny, n_iter=n_iter, ys=ys, yt=yt,
                             match_targets=match_targets,
                             normalization=normalization, 
                             wrapped_function=wrapped_function, 
                             detach_x=detach_x, detach_y=detach_y,                             
                             manifold=manifold, is_hyperbolic=is_hyperbolic)
    Gxx, Wxx = sinkhorn_cost(x, x, reg_ot, 
                             nx=nx, ny=ny, n_iter=n_iter, ys=ys, yt=yt,
                             match_targets=match_targets,
                             normalization=normalization, 
                             wrapped_function=wrapped_function, 
                             detach_x=detach_x, detach_y=detach_y,                             
                             manifold=manifold, is_hyperbolic=is_hyperbolic)
    Gyy, Wyy = sinkhorn_cost(y, y, reg_ot, 
                             nx=nx, ny=ny, n_iter=n_iter, ys=ys, yt=yt,
                             match_targets=match_targets,
                             normalization=normalization, 
                             wrapped_function=wrapped_function, 
                             detach_x=detach_x, detach_y=detach_y,                             
                             manifold=manifold, is_hyperbolic=is_hyperbolic)
    return 2 * Wxy - Wxx - Wyy


def compute_cost(Xs, Xt, ys=None, yt=None, manifold=None, is_hyperbolic=True,  
                 normalization='max', wrapped_function=None, 
                 detach_x=False, detach_y=True, limit_max=1e15, 
                 match_targets=False):
    """
        Cost used in the OT problem
    """
    if is_hyperbolic:
        wrapped_function = ((lambda x: torch.cosh(x).log()) 
                            if wrapped_function is None else wrapped_function)         
        manifold = Mobius() if manifold is None else manifold
        M_0 = cost_matrix_hyperbolic(
            Xs, Xt, 
            manifold=manifold, 
            detach_x=detach_x, detach_y=detach_y
        )
        M_0 = wrapped_function(M_0)    
    else: 
        M_0 = cost_matrix(Xs, Xt)
    
    M_0 = cost_normalization(M_0, normalization)

    if ((ys is not None) and (yt is not None)) and match_targets:
        limit_max_ = limit_max * M_0.max()
        ys_ = ys.data.numpy()
        yt_ = yt.data.numpy()
        classes = [c for c in np.unique(ys_) if c != -1]
        # assumes labeled source samples occupy the first rows
        # and labeled target samples occupy the first columns
        for c in classes:
            idx_s = np.where((ys_ != c) & (ys_ != -1))
            idx_t = np.where(yt_ == c)
            # all the coefficients corresponding to a source sample
            # and a target sample :
            # with different labels get a infinite
            for j in idx_t[0]:
                M_0[idx_s[0], j] = torch.tensor([limit_max_]) 

    return M_0


def compute_transport(Xs, Xt, 
                      ys=None, yt=None, manifold=None, M=None, reg_ot=1e-2,
                      is_hyperbolic=True, normalization='max', wrapped_function=None,
                      ot_solver='sinkhorn_knopp', limit_max=1e15,
                      detach_x=False, detach_y=True, match_targets=False,
                      verbose=False):
    ns, nt = len(Xs), len(Xt)
    a_np = unif(ns)
    b_np = unif(nt)
    
    if M is None:
        M_0 = compute_cost(Xs=Xs, Xt=Xt, ys=ys, yt=yt, 
                           manifold=manifold, 
                           is_hyperbolic=is_hyperbolic, 
                           normalization=normalization,
                           wrapped_function=wrapped_function, 
                           match_targets=match_targets,
                           detach_x=detach_x, detach_y=detach_y)
        M_np = M_0.data.numpy()
    else:
        M_0 = M.detach()
        M_0 = cost_normalization(M_0, normalization)
        M_np = M_0.data.numpy()

        if ((ys is not None) and (yt is not None)) and match_targets:
            ys_ = ys.data.numpy()
            yt_ = yt.data.numpy()
            classes = [c for c in np.unique(ys_) if c != -1]
            # assumes labeled source samples occupy the first rows
            # and labeled target samples occupy the first columns
            for c in classes:
                idx_s = np.where((ys_ != c) & (ys_ != -1))
                idx_t = np.where(yt_ == c)
                # all the coefficients corresponding to a source sample
                # and a target sample :
                # with different labels get a infinite
                for j in idx_t[0]:
                    M_np[idx_s[0], j] = limit_max

    if verbose:
        print("Computing initial coupling...")
        
    if reg_ot == 0:
        G_np = emd(a_np, b_np, M_np)
    else: 
        if ot_solver == 'greenkhorn':
            G_np = greenkhorn(a_np, b_np, M_np, reg=reg_ot)
        elif ot_solver == 'sinkhorn':
            G_np = sinkhorn_stabilized(a_np, b_np, M_np, reg=reg_ot)    
        elif ot_solver == 'sinkhorn_knopp':
            G_np = sinkhorn_knopp(a_np, b_np, M_np, reg=reg_ot)   
        else:
            raise ValueError
    if verbose:
        print("Coupling done")

    a = torch.FloatTensor(a_np).detach_()
    b = torch.FloatTensor(b_np).detach_()        
    G = torch.FloatTensor(G_np).detach_()
    M = torch.FloatTensor(M_np).detach_()
    return a, b, M, G