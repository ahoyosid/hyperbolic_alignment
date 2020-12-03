import numpy as np
import os
from os.path import join as pjoin
from copy import deepcopy
import torch
import torch.optim as optim
from torch import nn
from torch.optim.optimizer import required

from collections import defaultdict
from scipy.optimize.linesearch import scalar_search_armijo

from ot.lp import emd
from ot.utils import unif, cost_normalization
from ot.bregman import (
    sinkhorn_stabilized, 
    greenkhorn, 
    sinkhorn_epsilon_scaling,
    sinkhorn_knopp
)
from torch.autograd import Variable
from torch.optim import Adam
from geoopt.optim import RiemannianAdam
from gaussian_transport import plan_gaussian_transport

from torch.utils.data import (
    DataLoader, 
    TensorDataset, 
    RandomSampler
)
from transport_utils import (    
    cost_matrix,
    cost_matrix_hyperbolic,
    compute_cost, 
    compute_transport
)
from manifold import Mobius, Euclidean

from mobius_neural_network import (
    MobiusClassificationLayer, 
    MobiusLinear, 
    MobiusTanh
)
from sklearn.metrics import (
    accuracy_score, 
    f1_score,
    precision_recall_fscore_support, 
    matthews_corrcoef, 
    roc_auc_score
)
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
import joblib


### Aux functions
def to_np(v): 
    return v.data.numpy()


def to_float(v): 
    return float(to_np(v))


### Hyperbolic Mapping estimation
def _linear_oracle(f_grad, a, b, reg_ot, ot_solver):
    """
        <M, G + reg_fidelity \nabla f(G)> + reg_ot R(G) 
    """
    ## problem linearization
    Mi = f_grad
    ## set M positive
    Mi = Mi + Mi.min()
    
    a_np = to_np(a)
    b_np = to_np(b)
    Mi_np = to_np(Mi)

    if reg_ot > 0:
        if ot_solver == 'sinkhorn':
            Gc = sinkhorn_stabilized(a_np, b_np, Mi_np, reg=reg_ot)
        elif ot_solver == 'greenkhorn':
            Gc = greenkhorn(a_np, b_np, Mi_np, reg=reg_ot)
        elif ot_solver == 'sinkhorn_knopp':
            Gc = sinkhorn_knopp(a_np, b_np, Mi_np, reg=reg_ot)
        elif ot_solver == 'sinkhorn_epsilon_scaling':
            Gc = sinkhorn_epsilon_scaling(a_np, b_np, Mi_np, reg=reg_ot)
    else:
        # solve linear program
        Gc = emd(a_np, b_np, Mi_np)        
    return Mi, torch.FloatTensor(Gc)


class LossModule(nn.Module):
    
    def __init__(self, xs, xt, M, manifold, a=None, b=None,
                 reg_fidelity=1., reg_ot=1e-3, eps=1e-15, 
                 ot_solver='sinkhorn'):
        
        super(LossModule, self).__init__()
        self.M = deepcopy(M).detach_() 
        self.xs_ = xs
        self.xt_ = xt
        self.ns_ = len(xs)
        self.nt_ = len(xt)
        self.manifold = manifold
        self.reg_fidelity = reg_fidelity
        self.reg_ot = reg_ot 
        self.eps = eps
        self.ot_solver = ot_solver
        self.a = torch.FloatTensor(unif(self.ns_)) if a is None else a
        self.b = torch.FloatTensor(unif(self.nt_)) if b is None else b

    def similarity(self, xst, x_bary):
        #return self.manifold.distance(xst, x_bary).pow(2).sum()
        return self.manifold.distance(xst, x_bary).abs().mean()
    
    def similarity_coupling_fix(self, transport_map, coupling):
        x_bary = self.manifold.barycenter_mapping(self.xt_, 
                                                  coupling=coupling).detach()
        x_transp = transport_map(self.xs_)
        return  self.similarity(x_transp, x_bary)

    def similarity_map_fix(self, transport_map, coupling):
        x_bary = self.manifold.barycenter_mapping(self.xt_, 
                                                  coupling=coupling)
        with torch.no_grad():
            x_transp = transport_map(self.xs_)
        return  self.similarity(x_transp, x_bary)    

    def transport_cost(self, coupling):
        # Entropy regularization
        coupling_ = coupling + self.eps
        base_transport_cost = (self.M * coupling).sum() 
        regularization = (coupling * coupling_.log()).sum()
        return base_transport_cost + self.reg_ot * regularization
    
    ##### Total cost
    def cost(self, transport_map, coupling):    
        transport_cost = self.transport_cost(coupling)
        x_bary = self.manifold.barycenter_mapping(self.xt_, 
                                                  coupling=coupling)
        x_transp = transport_map(self.xs_)
        fidelity_cost = self.similarity(x_transp, x_bary)
        return transport_cost + self.reg_fidelity * fidelity_cost
    
    def cost_map_fix(self, transport_map, coupling):    
        transport_cost = self.transport_cost(coupling)
        fidelity_cost = self.similarity_map_fix(transport_map, coupling)
        return transport_cost + self.reg_fidelity * fidelity_cost    
    
    @torch.autograd.enable_grad()
    def grad_cost(self, transport_map, coupling):  
        coupling_ = Variable(coupling, requires_grad=True)
        loss = self.cost_map_fix(transport_map, coupling_)
        loss.backward()
        grad_coupling = coupling_.grad
        return grad_coupling
    
    @torch.autograd.no_grad()
    def linear_oracle(self, f_grad):
        return _linear_oracle(f_grad,  
                              self.a, self.b, 
                              reg_ot=self.reg_ot,
                              ot_solver=self.ot_solver)


class CouplingFW(optim.Optimizer):
    """
    (Generalized) Frank-Wolfe implementation
    """
    def __init__(self, params, transport_map=required, 
                 loss=required, c1=1e-4, alpha0=0.99):
        self.k_ = 0
        self.c1 = c1
        self.alpha0 = alpha0
            
        defaults = dict(loss=loss, transport_map=transport_map)
        super(CouplingFW, self).__init__(params, defaults)

    @torch.autograd.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None if closure is None else float(closure())
        for group in self.param_groups:
            loss = group['loss']
            transport_map = group['transport_map']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                G = p.data
                grad_euclidean = p.grad 
                ## Riemannian gradient
                f_grad = loss.manifold.rgrad(p, grad_euclidean)

                _, Gc = loss.linear_oracle(f_grad=f_grad.data)
                delta_p = Gc - G                
                ## <M, G> + reg1 <G, log(G)> + reg2 f(G)
                grad_coupling_loss = loss.grad_cost(transport_map, G)
                
                # Line search
                alpha, _, _ = self.line_search_armijo(
                    loss, G, delta_p, grad_coupling_loss,
                    transport_map=transport_map
                )
                # Setting the step size (naive)
                gamma = (2 / (self.k_ + 2)) if alpha is None else alpha

                # Update: x^(k+1) = x^(k) - g x^(k) + g s
                p.data.add_(gamma * delta_p)                               
                                
        self.k_ += 1

    @torch.autograd.no_grad()
    def line_search_armijo(self, f, xk, pk, gfk, transport_map, old_fval=None):
        """ Addapted from POT 
        """        
        fc = [0]
        
        def phi(alpha1):
            fc[0] += 1
            return to_np(f.cost(transport_map, xk + alpha1 * pk))

        phi0 = phi(0.) if old_fval is None else to_np(old_fval)

        derphi0 = to_np((pk * gfk).sum())

        alpha, phi1 = scalar_search_armijo(
            phi, phi0, derphi0, 
            c1=self.c1, alpha0=self.alpha0)
        return alpha, fc[0], phi1


def solve_map(transport_map, coupling, loss,  
              lr=1e-2, max_iter=2000, 
              stop_thr=1e-4, verbose=False, 
              every_n=200, is_hyperbolic=True, results_path=None):
    
    vloss = [stop_thr]
    # init loop
    loop = 1 if max_iter > 0 else 0
    it = 0
    if is_hyperbolic:
        optimizer_map = RiemannianAdam(transport_map.parameters(), lr=lr)
    else:
        optimizer_map = Adam(transport_map.parameters(), lr=lr)
    
    while loop:
        it += 1
        optimizer_map.zero_grad()
        f_loss = loss.similarity_coupling_fix(transport_map, coupling)
        f_loss.backward()
        optimizer_map.step()
        
        vloss.append(to_float(f_loss))
        relative_error = abs(vloss[-1] - vloss[-2]) / abs(vloss[-2]) 

        if verbose and (it % every_n == 0):
            print("\t \t it: %s similarity loss: %.3f" % (it, vloss[-1]))
        
        if (it >= max_iter) or (np.isnan(vloss[-1])):
            loop = 0
            
        if relative_error < stop_thr:
            loop = 0
    return transport_map


def solve_coupling(transport_map, current_coupling, loss,
                   max_iter=100, stop_thr=1e-4, 
                   verbose=False, every_n=20):
    
    coupling = Variable(current_coupling.clone().detach_(), 
                        requires_grad=True)

    optimizer_coupling = CouplingFW([coupling], loss=loss, 
                                    transport_map=transport_map)
    
    vloss = [stop_thr]
    # init loop
    loop = 1 if max_iter > 0 else 0
    it = 0
    while loop:
        it += 1
        
        optimizer_coupling.zero_grad()
        coupling_loss = loss.similarity_map_fix(transport_map, coupling)
        coupling_loss.backward()
        optimizer_coupling.step()
    
        cost = loss.cost(transport_map, coupling)
        vloss.append(to_float(cost))
        relative_error = abs(vloss[-1] - vloss[-2]) / abs(vloss[-2])    
        
        if verbose and (it % every_n == 0):
            print("\t \t it: %s loss couplin: %.3f" % (it, vloss[-1]))        

        if (it >= max_iter) or (np.isnan(vloss[-1])):
            loop = 0

        if relative_error < stop_thr:
            loop = 0  
            
    return coupling


class WrappedLinearModel(nn.Module):
    
    def __init__(self, manifold):
        super(WrappedLinearModel, self).__init__()
        self.manifold = manifold

    def fit(self, xs, xt):    
        self.ns_ = xs.shape[0]
        self.nt_ = xt.shape[0]
        self.d_ = xt.shape[1]
        self.bias_source_ = self.manifold.mean(xs)
        self.bias_target_ = self.manifold.mean(xt)
        ## Centering
        xs_centered = self.manifold.add(-self.bias_source_, xs) 
        xt_centered = self.manifold.add(-self.bias_target_, xt) 
        ## Covariances
        xs_logm = self.manifold.logm_zero(xs_centered) 
        xt_logm = self.manifold.logm_zero(xt_centered)
        self.cov_s_ = xs_logm.t().mm(xs_logm) / self.ns_   
        self.cov_t_ = xt_logm.t().mm(xt_logm) / self.nt_           
        ## Transport plan
        self.T_ = plan_gaussian_transport(self.cov_s_, self.cov_t_)          
        
    def forward(self, xs):
        xs_centered = self.manifold.add(-self.bias_source_, xs) 
        xst = self.manifold.mat_mul(self.T_, xs_centered)
        return self.manifold.add(self.bias_target_, xst)


def solve_wrapped_model(coupling, loss):
    manifold = loss.manifold
    xt = loss.xt_
    xs = loss.xs_
    x_bary = manifold.barycenter_mapping(xt, coupling)
    transport_map = WrappedLinearModel(manifold=manifold)
    transport_map.fit(xs, x_bary)
    return transport_map


def solve_hyperbolic_nn_model(transport_map, coupling, loss, 
                              max_iter=100, stop_thr=1e-3, lr=1e-2,
                              display_every=100, verbose=False, 
                              is_hyperbolic=True):
    if is_hyperbolic:
        optimizer = RiemannianAdam(transport_map.parameters(), lr=lr)
    else:
        optimizer = Adam(transport_map.parameters(), lr=lr)

    vloss = [stop_thr]
    # init loop
    loop = 1 if max_iter > 0 else 0
    it = 0
    while loop:
        it += 1
        optimizer.zero_grad()
        l = loss.similarity_coupling_fix(transport_map, coupling)
        vloss.append(to_float(l))

        if (it >= max_iter) or (np.isnan(vloss[-1])):
            loop = 0

        relative_error = abs(vloss[-1] - vloss[-2]) / abs(vloss[-2])    

        if relative_error < stop_thr:
            loop = 0  
            
        if (it % display_every == 0) and verbose :
            print("\t\t it: %s loss map: %.4f" % (it, to_float(l)))
            
        l.backward()
        optimizer.step()  

    return transport_map


def mapping_estimation(Xs, Xt, transport_model=None, manifold=None, 
                       ys=None, yt=None,
                       lr=1e-3, 
                       reg_ot=0, 
                       ot_solver='sinkhorn_knopp',
                       display_every_map=50,
                       display_every_coupling=10,
                       reg_fidelity=1e-3,
                       stop_thr_coupling=1e-5,
                       stop_thr_map=1e-5,
                       stop_thr_global=1e-5,
                       max_iter_map=300,
                       max_iter_coupling=20, 
                       wrapped_function=None,
                       normalization='max',
                       max_iter=10, 
                       match_targets=False,
                       is_hyperbolic=True, 
                       verbose=False, 
                       results_path=None,
                       is_wrapped_linear=False, 
                       save=False):

    results_path = '' if results_path is None else results_path
    transport_map = deepcopy(transport_model)
    
    if manifold is None:
        manifold = Mobius() if is_hyperbolic else Euclidean()
        
    ### Compute cost
    a, b, M, coupling = compute_transport(
        Xs=Xs, Xt=Xt, 
        ys=ys, yt=yt,
        reg_ot=reg_ot, 
        ot_solver=ot_solver,
        wrapped_function=wrapped_function,
        normalization=normalization,
        match_targets=match_targets,
        is_hyperbolic=is_hyperbolic)

    loss = LossModule(xs=Xs, xt=Xt, 
                      M=M, a=a, b=b, 
                      manifold=manifold, 
                      reg_fidelity=reg_fidelity,
                      ot_solver=ot_solver,
                      reg_ot=reg_ot)

    vloss = [stop_thr_global]
    ### Initialize transport map
    if is_wrapped_linear:
        transport_map = solve_wrapped_model(coupling, loss)
    else:
        transport_map = solve_hyperbolic_nn_model(
            transport_map, coupling, loss, 
            max_iter=max_iter_map, stop_thr=stop_thr_map, lr=lr, 
            display_every=display_every_map, verbose=verbose, 
            is_hyperbolic=is_hyperbolic)

    ### Save initial model
    if save:
        joblib.dump(transport_map, pjoin(
            results_path, 
            f"{reg_ot}_{reg_fidelity}_{is_hyperbolic}_{match_targets}_initial.model")
        )

    vloss.append(to_float(loss.cost(transport_map, coupling)))
    loop = 1 if max_iter > 0 else 0
    it = 0
    if verbose:
        print("global it: %s cost: %.4f" % (it, vloss[-1]))

    #### Block coordinate descent
    while loop:
        it += 1
        # update coupling
        if verbose:
            print("\t update coupling")    
        coupling = solve_coupling(transport_map, coupling, 
                                  loss=loss, max_iter=max_iter_coupling, 
                                  stop_thr=stop_thr_coupling, 
                                  verbose=verbose, every_n=display_every_coupling)    

        # update transport map
        if verbose:
            print("\t update transport map")  
            
        if is_wrapped_linear:
            transport_map = solve_wrapped_model(coupling, loss)
        else:
            transport_map = solve_hyperbolic_nn_model(
                transport_map, coupling, loss, 
                max_iter=max_iter_map, stop_thr=stop_thr_map, lr=lr, 
                display_every=display_every_map, verbose=verbose, 
                is_hyperbolic=is_hyperbolic)

        if save:
            joblib.dump(transport_map, pjoin(
                results_path, 
                f"{reg_ot}_{reg_fidelity}_{is_hyperbolic}_{match_targets}_{it}.model")
            )

        cost = to_float(loss.cost(transport_map, coupling))
        vloss.append(cost)

        if verbose:
            print("global it: %s cost: %.4f" % (it, vloss[-1]))

        if (it >= max_iter) or (np.isnan(vloss[-1])):
            loop = 0

        relative_error = abs(vloss[-1] - vloss[-2]) / abs(vloss[-2])    

        if relative_error < stop_thr_global:
            loop = 0  
    return transport_map, coupling


##############################################################################
### Hyperbolic Sinkhorn Transport
##############################################################################
class HyperbolicSinkhornTransport(BaseEstimator):
    
    def __init__(self, reg_ot=1e-1, ot_solver='sinkhorn_knopp', 
                batch_size=128, wrapped_function=None,
                normalization='max', is_hyperbolic=True, match_targets=False):
        self.reg_ot = reg_ot
        self.ot_solver = ot_solver
        self.batch_size = batch_size
        self.wrapped_function = wrapped_function
        self.normalization = normalization
        self.is_hyperbolic = is_hyperbolic
        self.match_targets = match_targets
        self.manifold = Mobius() if is_hyperbolic else Euclidean() 
    
    def fit(self, Xs, Xt, ys=None, yt=None):
        self.Xs_train_ = deepcopy(Xs)
        self.Xt_train_ = deepcopy(Xt) 
        
        _, _, _, coupling = compute_transport(
            Xs=Xs, Xt=Xt, 
            ys=ys, yt=yt, 
            reg_ot=self.reg_ot, 
            ot_solver=self.ot_solver,
            wrapped_function=self.wrapped_function,
            normalization=self.normalization,
            is_hyperbolic=self.is_hyperbolic,
            detach_x=False, detach_y=False, 
            match_targets=self.match_targets,
        ) 
        self.coupling_ = coupling
        return self

    def transform(self, Xs):
        indices = np.arange(Xs.shape[0])
        batch_ind = [
            indices[i:i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)]  

        transp_Xs = []
        X_bary = self.manifold.barycenter_mapping(self.Xt_train_, 
                                                  self.coupling_)
        #print(X_bary)
        for bi in batch_ind:
            # get the nearest neighbor in the source domain
            M0 = compute_cost(Xs[bi], self.Xs_train_, 
                              normalization=self.normalization, 
                              wrapped_function=lambda x: x, # using the hyperbolic distance
                              is_hyperbolic=self.is_hyperbolic)
            idx = M0.argmin(dim=1)
            # define the transported points
            diff = self.manifold.add(-self.Xs_train_[idx, :], Xs[bi]) 
            ####transp_Xs_ = self.manifold.add(diff, X_bary[idx, :])
            transp_Xs_ = self.manifold.add(X_bary[idx, :], diff)
            transp_Xs.append(transp_Xs_)

        return torch.cat(transp_Xs, dim=0)
    
    def fit_transport(self, Xs, Xt, ys=None, yt=None):
        self.fit(Xs=Xs, Xt=Xt, ys=ys, yt=yt)
        return self.transform(Xs)


###############################################################################
### Sinkhorn alignment
###############################################################################
from transport_utils import sinkhorn_cost, sinkhorn_normalized
from scipy.linalg import orthogonal_procrustes


def minimize_sinkhorn(Xs, Xt, model, ys=None, yt=None, 
                      lr=1e-2, reg_ot=1e-2, is_hyperbolic=True, 
                      match_targets=True, n_iter_sinkhorn=100,
                      max_iter=10000, stop_thr=1e-5, max_iter_map=1000,
                      is_sinkhorn_normalized=False, every_n=100, rate=1e-2,
                      is_projected_output=False, type_ini='bary',
                      is_init_map=False, verbose=False):
   
    mobius = Mobius()
    # Initialize: Barycenter approximation
    if is_init_map:
        
        model = deepcopy(model)
        
        if is_hyperbolic:
            optimizer_init = RiemannianAdam(model.parameters(), lr=lr)
            manifold = Mobius() 
        else:
            optimizer_init = Adam(model.parameters(), lr=lr)
            manifold = Euclidean() 

        ### Compute cost
        _, _, _, coupling = compute_transport(
            Xs=Xs, Xt=Xt,         
            ys=ys, yt=yt,
            reg_ot=reg_ot, 
            match_targets=match_targets,
            manifold=manifold,
            is_hyperbolic=is_hyperbolic
        )    

        if type_ini == 'bary':
            x_approx = manifold.barycenter_mapping(Xt, coupling)
        elif (type_ini == 'rot_s2t') or (type_ini == 'rot_t2s'):
            xs_np = Xs.data.numpy()
            xt_np = Xt.data.numpy()
            xs_mean = xs_np.mean(0)
            xt_mean = xt_np.mean(0)
            xs_centered = xs_np - xs_mean
            xt_centered = xt_np - xt_mean
            if type_ini == 'rot_s2t':
                P, _ = orthogonal_procrustes(xs_centered, xt_centered)
                x_approx = torch.FloatTensor(xs_centered.dot(P) + xt_mean)
            else:
                P, _ = orthogonal_procrustes(xt_centered, xs_centered)
                x_approx = torch.FloatTensor(xt_centered.dot(P) + xs_mean)                

        elif type_ini == 'id':
            x_approx = Xs

        loop_map = 1 if max_iter_map > 0 else 0
        vloss_map = [stop_thr]
        it = 0
        while loop_map:
            it += 1
            optimizer_init.zero_grad()
            X_pred = mobius.proj2ball(model(Xs)) if is_projected_output else model(Xs)
            loss_map = manifold.distance(X_pred, x_approx).mean()
            vloss_map.append(loss_map.item())
            relative_error = abs(vloss_map[-1] - vloss_map[-2]) / abs(vloss_map[-2]) 
            if (it >= max_iter_map) or (np.isnan(vloss_map[-1])):
                loop_map = 0

            if relative_error < stop_thr:
                loop_map = 0
                
            loss_map.backward()    
            optimizer_init.step()
            
    this_model = deepcopy(model)
    lr_mapping = lr * rate
    
    if is_hyperbolic:
        optimizer = RiemannianAdam(this_model.parameters(), lr=lr_mapping)
    else:
        optimizer = Adam(this_model.parameters(), lr=lr_mapping)

    vloss = [stop_thr]

    loop = 1 if max_iter > 0 else 0
    it = 0
    while loop:
        it += 1
        optimizer.zero_grad()
        X_pred = mobius.proj2ball(this_model(Xs)) if is_projected_output else this_model(Xs)

        if is_sinkhorn_normalized:

            loss = sinkhorn_normalized(
                X_pred, Xt, 
                reg_ot=reg_ot, 
                n_iter=n_iter_sinkhorn, 
                ys=ys, yt=yt,
                match_targets=match_targets,
                is_hyperbolic=is_hyperbolic)
        else:
            G, loss = sinkhorn_cost(
                X_pred, Xt,
                reg_ot=reg_ot, 
                n_iter=n_iter_sinkhorn, 
                match_targets=match_targets,
                #wrapped_function=lambda x: -torch.cosh(x),
                ys=ys, yt=yt,
                is_hyperbolic=is_hyperbolic)

        vloss.append(loss.item())

        relative_error = (abs(vloss[-1] - vloss[-2]) / abs(vloss[-2]) 
                          if vloss[-2] != 0 else 0) 

        if verbose and (it % every_n == 0):
            print("\t \t it: %s similarity loss: %.3f" % (it, vloss[-1]))

        if (it >= max_iter) or (np.isnan(vloss[-1])):
            loop = 0

        if relative_error < stop_thr:
            loop = 0

        loss.backward()
        optimizer.step()
        
    return this_model